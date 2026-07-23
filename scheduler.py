"""VLIW dependency DAG + list scheduler.

Two components:

  - ``DAG``: a dependency graph with built-in frontier management. It
    answers the single question the scheduler cares about each cycle:
    *which instructions have all data dependencies resolved and are ready
    for scheduling?*  All dependency bookkeeping (remaining blockers,
    buffered resolutions, committed flags, the frontier set) lives inside
    the DAG; nothing leaks onto the nodes.

  - ``schedule()``: a list scheduler. Each cycle it takes the DAG's ready
    set as its working set and greedily fills engine slots, spilling
    spillable vector ops to ALU when VALU is saturated.

Dependency model (the machine is read-before-write within a cycle: every
slot reads pre-cycle state, every write commits at end of cycle):

  - RAW (edge weight 1): producer -> consumer. The consumer becomes ready
    the cycle *after* the producer commits. Resolutions are *deferred* -
    buffered during the cycle and applied at ``advance()``.
  - WAR (edge weight 0): old reader -> new writer. The writer becomes
    ready the *same* cycle the reader commits. Resolutions are *immediate*
    - applied at ``commit()`` and reported back to the scheduler.
  - No WAW edges: consecutive same-address writes are bridged transitively
    through any intervening reader (a RAW chain); a write with no
    intervening reader is dead and is warned, not eliminated.
"""

from dataclasses import dataclass, field
import heapq
import random

from problem import VLEN, SLOT_LIMITS

# Flow ops that modify the PC - the DAG cannot represent control flow.
# Hitting one means a jump leaked into the body.
FLOW_PANIC = {"cond_jump", "cond_jump_rel", "jump", "jump_indirect", "halt", "pause"}


@dataclass
class DNode:
    """A node in the dependency DAG - static graph data only.

    Dynamic dependency state (remaining blockers, committed flag, frontier
    membership) is owned by the ``DAG``; per-cycle scheduling state (lanes
    landed, sticky engine choice) is owned by the scheduler's
    ``_Placement`` records. DNode itself never mutates after construction.
    """
    idx: int
    engine: str
    slot: tuple
    reads: list[tuple[int, bool]] = field(default_factory=list)
    writes: list[tuple[int, bool]] = field(default_factory=list)
    in_edges: list[tuple[int, int]] = field(default_factory=list)   # (src_idx, weight)
    out_edges: list[tuple[int, int]] = field(default_factory=list)  # (dst_idx, weight)


def slot_io(engine: str, slot: tuple) -> tuple[list[tuple[int, bool]], list[tuple[int, bool]]]:
    """Return (reads, writes) for a slot.

    Each entry is (addr, is_vector).  is_vector=True means the full 8-lane
    vector [addr..addr+7]; is_vector=False means the single word at addr.

    Memory accesses are NOT dependency edges (the body only reads the
    read-only tree; scratch has no indirect reads).
    """
    reads: list[tuple[int, bool]] = []
    writes: list[tuple[int, bool]] = []

    if engine == "alu":
        # ("op", dest, a1, a2) - binary scalar
        _, dest, a1, a2 = slot
        reads.append((a1, False))
        reads.append((a2, False))
        writes.append((dest, False))

    elif engine == "valu":
        op = slot[0]
        if op == "vbroadcast":
            # ("vbroadcast", dest, src) - dest is vec, src is scalar
            _, dest, src = slot
            reads.append((src, False))
            writes.append((dest, True))
        elif op == "multiply_add":
            # ("multiply_add", dest, a, b, c) - all vec (VLEN=8)
            _, dest, a, b, c = slot
            reads.append((a, True))
            reads.append((b, True))
            reads.append((c, True))
            writes.append((dest, True))
        else:
            # (op, dest, a1, a2) - elementwise over VLEN=8 lanes
            _, dest, a1, a2 = slot
            reads.append((a1, True))
            reads.append((a2, True))
            writes.append((dest, True))

    elif engine == "load":
        op = slot[0]
        if op == "load":
            # ("load", dest, addr) - scalar; addr holds mem ptr
            _, dest, addr = slot
            reads.append((addr, False))
            writes.append((dest, False))
        elif op == "load_offset":
            # ("load_offset", dest, addr, offset) - offset is a literal int
            _, dest, addr, offset = slot
            reads.append((addr + offset, False))
            writes.append((dest + offset, False))
        elif op == "vload":
            # ("vload", dest, addr) - addr is scalar (mem base ptr)
            _, dest, addr = slot
            reads.append((addr, False))
            writes.append((dest, True))
        elif op == "const":
            # ("const", dest, val) - val is literal, no reads
            _, dest, _val = slot
            writes.append((dest, False))
        else:
            raise NotImplementedError(f"Unknown load op: {op}")

    elif engine == "store":
        op = slot[0]
        if op == "store":
            # ("store", addr, src) - reads addr (mem ptr) + src (data); no writes
            _, addr, src = slot
            reads.append((addr, False))
            reads.append((src, False))
        elif op == "vstore":
            # ("vstore", addr, src) - reads addr (scalar) + src..src+7 (vec)
            _, addr, src = slot
            reads.append((addr, False))
            reads.append((src, True))
        else:
            raise NotImplementedError(f"Unknown store op: {op}")

    elif engine == "flow":
        op = slot[0]
        if op in FLOW_PANIC:
            raise NotImplementedError(
                f"Flow op '{op}' at slot {slot} modifies PC - "
                f"cannot be represented in the DAG")
        elif op == "select":
            # ("select", dest, cond, a, b) - scalar
            _, dest, cond, a, b = slot
            reads.extend([(cond, False), (a, False), (b, False)])
            writes.append((dest, False))
        elif op == "vselect":
            # ("vselect", dest, cond, a, b) - vec
            _, dest, cond, a, b = slot
            reads.extend([(cond, True), (a, True), (b, True)])
            writes.append((dest, True))
        elif op == "add_imm":
            # ("add_imm", dest, a, imm) - imm is literal
            _, dest, a, _imm = slot
            reads.append((a, False))
            writes.append((dest, False))
        elif op == "coreid":
            # ("coreid", dest) - writes dest, no reads
            _, dest = slot
            writes.append((dest, False))
        elif op == "trace_write":
            # ("trace_write", val) - reads val
            _, val = slot
            reads.append((val, False))
        else:
            raise NotImplementedError(f"Unknown flow op: {op}")

    elif engine == "debug":
        op = slot[0]
        if op == "compare":
            # ("compare", loc, key) - reads loc (scalar)
            _, loc, _key = slot
            reads.append((loc, False))
        elif op == "vcompare":
            # ("vcompare", loc, keys) - reads loc..loc+7 (vec)
            _, loc, _keys = slot
            reads.append((loc, True))
        elif op == "comment":
            pass  # no deps
        else:
            pass  # unknown debug ops are no-ops for dependency purposes

    else:
        raise NotImplementedError(f"Unknown engine: {engine}")

    return reads, writes


class DAG:
    """Dependency graph with built-in frontier management.

    Answers: *which instructions have all data dependencies resolved and
    are ready for scheduling?*

    Construction builds nodes + edges from a slot list (RAW weight-1 /
    WAR weight-0, deduped per (src, dst, weight), no WAW). The frontier is
    the set of node indices with zero unresolved in-edges.

    Scheduling protocol (one cycle)::

        working  = dag.ready()          # snapshot of ready node indices
        ...place nodes from working...
        unlocked = dag.commit(idx)      # WAR unlocks returned (same cycle)
        working.update(unlocked)        # add newly-ready nodes
        ...repeat until no progress...
        dag.advance()                   # RAW unlocks become ready next cycle

    ``commit()`` is the only place WAR blockers decrease (immediately).
    ``advance()`` is the only place RAW blockers decrease (end of cycle).
    """

    def __init__(self, slots: list[tuple[str, tuple]]):
        self.nodes: list[DNode] = self._build_nodes(slots)
        n = len(self.nodes)
        self._raw = [0] * n          # remaining RAW (weight-1) in-edges
        self._war = [0] * n          # remaining WAR (weight-0) in-edges
        self._pending = [0] * n      # RAW resolutions buffered this cycle
        self._committed = [False] * n
        self._frontier: set[int] = set()
        for node in self.nodes:
            self._raw[node.idx] = sum(1 for _, w in node.in_edges if w == 1)
            self._war[node.idx] = sum(1 for _, w in node.in_edges if w == 0)
            assert self._raw[node.idx] + self._war[node.idx] == len(node.in_edges), (
                f"Node {node.idx}: raw+war blockers != len(in_edges)")
        for i in range(n):
            if self._raw[i] == 0 and self._war[i] == 0:
                self._frontier.add(i)

    # -- queries ----------------------------------------------------------

    def ready(self) -> set[int]:
        """Snapshot of node indices ready to schedule this cycle."""
        return set(self._frontier)

    def is_committed(self, idx: int) -> bool:
        return self._committed[idx]

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, idx: int) -> DNode:
        return self.nodes[idx]

    # -- mutations --------------------------------------------------------

    def commit(self, idx: int) -> list[int]:
        """Mark node ``idx`` as scheduled and relax its out-edges.

        WAR (weight 0) children resolve immediately and may become ready
        this same cycle - their indices are returned so the scheduler can
        add them to its working set.  RAW (weight 1) resolutions are
        buffered in ``_pending`` and applied by ``advance()`` (so the child
        becomes ready next cycle, reflecting read-before-write latency).
        """
        self._committed[idx] = True
        self._frontier.discard(idx)
        unlocked: list[int] = []
        for dst, w in self.nodes[idx].out_edges:
            if self._committed[dst]:
                continue
            if w == 0:                       # WAR - same-cycle-safe
                self._war[dst] -= 1
                if self._unblocked(dst) and dst not in self._frontier:
                    self._frontier.add(dst)
                    unlocked.append(dst)
            else:                            # RAW - deferred
                self._pending[dst] += 1
        return unlocked

    def advance(self) -> list[int]:
        """Apply end-of-cycle RAW resolutions.

        Buffered resolutions are subtracted from RAW blockers; nodes that
        become fully unblocked are added to the frontier (ready next
        cycle).  Returns the newly-ready node indices.
        """
        unlocked: list[int] = []
        for i in range(len(self.nodes)):
            if self._pending[i] == 0:
                continue
            self._raw[i] -= self._pending[i]
            self._pending[i] = 0
            if (not self._committed[i]
                    and self._unblocked(i)
                    and i not in self._frontier):
                self._frontier.add(i)
                unlocked.append(i)
        return unlocked

    def _unblocked(self, idx: int) -> bool:
        """Node has zero unresolved in-edges (frontier-eligible)."""
        return self._raw[idx] == 0 and self._war[idx] == 0

    # -- construction -----------------------------------------------------

    @staticmethod
    def _build_nodes(slots: list[tuple[str, tuple]]) -> list[DNode]:
        """Build nodes + deduped bidirectional edges from a slot list.

        Region/lane bookkeeping (keyed by ``region = addr >> 3``,
        ``lane = addr & 7``) avoids false WAR edges and false dead-write
        warnings for scalar writes to different lanes of the same region
        (e.g. the 8 scalar gather loads landing into one vector plane).

        ``last_writer[region]`` is a tagged union: a single ``DNode`` when
        one op wrote all 8 lanes (vector write), or a list of 8 per-lane
        writers (``DNode | None``) when scalar writes landed into different
        lanes of the same region.
        """
        nodes: list[DNode] = []
        last_writer: dict[int, DNode | list] = {}
        readers_since: dict[tuple[int, int], list[int]] = {}

        for idx, (engine, slot) in enumerate(slots):
            reads, writes = slot_io(engine, slot)
            node = DNode(idx=idx, engine=engine, slot=slot,
                         reads=reads, writes=writes)
            nodes.append(node)

            # ---- RAW: each read depends on its lane's last writer ----
            read_lanes: set[tuple[int, int]] = set()
            for addr, is_vec in reads:
                r = addr >> 3
                lanes = range(VLEN) if is_vec else (addr & 7,)
                for lane in lanes:
                    key = (r, lane)
                    if key in read_lanes:
                        continue
                    read_lanes.add(key)
                    lw = last_writer.get(r)
                    if lw is None:
                        continue
                    if isinstance(lw, list):
                        src = lw[lane]
                        if src is None:
                            continue
                        src_idx = src.idx
                    else:
                        src_idx = lw.idx
                    if (src_idx, 1) not in node.in_edges:   # dedup
                        node.in_edges.append((src_idx, 1))
                        nodes[src_idx].out_edges.append((idx, 1))

            # ---- record self as a reader of each read lane ----
            for r, lane in read_lanes:
                readers_since.setdefault((r, lane), []).append(idx)

            # ---- WAR: each write gets edges from prior readers of that lane ----
            write_lanes: list[tuple[int, int]] = []
            for addr, is_vec in writes:
                r = addr >> 3
                if is_vec:
                    write_lanes.extend((r, lane) for lane in range(VLEN))
                else:
                    write_lanes.append((r, addr & 7))

            war_seen: set[int] = set()   # dedup WAR sources across lanes
            for r, lane in write_lanes:
                for old_idx in readers_since.get((r, lane), []):
                    if old_idx == idx or old_idx in war_seen:
                        continue
                    war_seen.add(old_idx)
                    node.in_edges.append((old_idx, 0))
                    nodes[old_idx].out_edges.append((idx, 0))

            # ---- dead-write warning (self-reads count as keeping it alive) ----
            for r, lane in write_lanes:
                lw = last_writer.get(r)
                if lw is None:
                    continue
                old = lw[lane] if isinstance(lw, list) else lw
                if old is None:
                    continue
                if not readers_since.get((r, lane)):
                    print(f"WARN: dead write - node {idx} '{engine} {slot[0]}' "
                          f"overwrites unread writer {old.idx} at region {r} lane {lane}")

            # ---- clear readers, update last_writer for written lanes ----
            for r, lane in write_lanes:
                readers_since[(r, lane)] = []
            write_regions: dict[int, bool] = {}   # region -> any vector write?
            for addr, is_vec in writes:
                r = addr >> 3
                write_regions[r] = write_regions.get(r, False) or is_vec
            for r, is_vec in write_regions.items():
                if is_vec:
                    last_writer[r] = node
                else:
                    for addr, wv in writes:
                        if (addr >> 3) == r and not wv:
                            lane = addr & 7
                            lw = last_writer.get(r)
                            if lw is None or isinstance(lw, DNode):
                                lw = ([lw] * VLEN) if lw is not None else ([None] * VLEN)
                            lw[lane] = node
                            last_writer[r] = lw

        # ---- bidirectional invariant asserts (edge structure) ----
        for n in nodes:
            for src_idx, w in n.in_edges:
                assert (n.idx, w) in nodes[src_idx].out_edges, (
                    f"Node {n.idx}: in-edge ({src_idx},{w}) not in src out_edges")
            for dst_idx, w in n.out_edges:
                assert (n.idx, w) in nodes[dst_idx].in_edges, (
                    f"Node {n.idx}: out-edge ({dst_idx},{w}) not in dst in_edges")

        return nodes


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

# Node classification tags.
_KIND_ATOMIC_SCALAR = "alu_scalar"
_KIND_LOAD = "load"
_KIND_STORE = "store"
_KIND_FLOW = "flow"
_KIND_DEBUG = "debug"
_KIND_VEC_FMA = "vec_fma"
_KIND_VEC_ELEM = "vec_elem"

# Priority for the default "fma_first" picker (lower = scheduled first).
_KIND_PRIORITY = {
    _KIND_VEC_FMA: 0,        # valu-rigid: needs a valu slot or it stalls
    _KIND_VEC_ELEM: 1,       # spillable: can fall back to alu
    _KIND_LOAD: 2,
    _KIND_STORE: 2,
    _KIND_FLOW: 2,
    _KIND_ATOMIC_SCALAR: 2,
    _KIND_DEBUG: 3,          # free; schedule last so real work fills first
}


@dataclass
class _Placement:
    """Per-node scheduling state: classification + spill progress.

    Owned by the scheduler (persists across cycles); mutated by the
    ``FuncUnitPool`` during placement.
    """
    kind: str
    lanes_total: int                  # 1 atomic; 8 spillable vec_elem; 0 debug
    native_engine: str                # alu / load / store / flow / valu / debug
    lanes_done: int = 0               # lanes landed so far
    engine_choice: str | None = None  # sticky once the first lane lands


def _classify(n: DNode) -> _Placement:
    """Classify a node for placement (kind, lanes, native engine)."""
    eng = n.engine
    op = n.slot[0] if n.slot else ""
    if eng == "alu":
        return _Placement(_KIND_ATOMIC_SCALAR, 1, "alu")
    if eng == "load":
        return _Placement(_KIND_LOAD, 1, "load")
    if eng == "store":
        return _Placement(_KIND_STORE, 1, "store")
    if eng == "flow":
        return _Placement(_KIND_FLOW, 1, "flow")
    if eng == "debug":
        return _Placement(_KIND_DEBUG, 0, "debug")
    if eng == "valu":
        if op == "multiply_add":
            return _Placement(_KIND_VEC_FMA, 1, "valu")    # rigid: no scalar fma
        return _Placement(_KIND_VEC_ELEM, VLEN, "valu")    # spillable to alu
    raise NotImplementedError(f"Unknown engine: {eng}")


def _vec_slot_to_alu_lanes(slot: tuple, lanes) -> list[tuple]:
    """Materialise one elementwise ``valu`` slot as per-lane ``alu`` tuples.

    (op, dest, a1, a2) elementwise -> lane j: (op, dest+j, a1+j, a2+j)
    (vbroadcast, dest, src)         -> lane j: ("+", dest+j, src, 0)
    multiply_add cannot spill (no scalar fma in the ISA) and raises.
    """
    op = slot[0]
    if op == "multiply_add":
        raise NotImplementedError("multiply_add cannot spill to alu (no scalar fma)")
    if op == "vbroadcast":
        _, dest, src = slot
        return [("+", dest + j, src, 0) for j in lanes]
    _, dest, a1, a2 = slot
    return [(op, dest + j, a1 + j, a2 + j) for j in lanes]


class FuncUnitPool:
    """Per-cycle functional-unit pool: assigns nodes to units and tracks
    slot occupation.

    The scheduler feeds it ready nodes one at a time; for each, the pool
    answers whether it fits and where, updating its own occupation state.
    The scheduler resets the pool at the start of every cycle and reads
    back the assembled bundle at cycle end - it never reasons about ports
    or realisation itself.

    Lifecycle::

        pool = FuncUnitPool()
        while ...:
            pool.reset()                          # cycle begin
            ...
            finished = pool.place(node, placement)  # yes / partial / no
            ...
            bundles.append(pool.bundle)           # cycle end - all placements
    """

    # Per-engine slot budgets (refreshed by reset() each cycle).
    _CAPACITY = {e: SLOT_LIMITS[e] for e in ("alu", "valu", "load", "store", "flow")}

    def __init__(self):
        self.free: dict[str, int] = {}
        self.bundle: dict[str, list] = {}
        self.reset()

    def reset(self) -> None:
        """Clear slot budgets and the bundle for a new cycle."""
        self.free = dict(self._CAPACITY)
        self.bundle = {}

    @property
    def has_work(self) -> bool:
        """True if any non-debug slot was emitted this cycle."""
        return any(e != "debug" for e in self.bundle)

    def place(self, node: DNode, p: _Placement) -> bool | None:
        """Try to place a single node this cycle.

        Updates occupation state and ``p`` (lanes landed / sticky engine)
        when the node is placed. Returns:

          True  - yes: placed and complete (caller commits to the DAG)
          False - partial: placed some lanes, needs more cycles
          None  - no: no unit had room; nothing changed
        """
        if p.kind == _KIND_DEBUG:
            self.bundle.setdefault("debug", []).append(node.slot)
            return True

        if p.kind == _KIND_VEC_ELEM:
            if p.lanes_done > 0:              # sticky alu continuation
                return self._spill_alu(node, p)
            if self.free["valu"] > 0:         # fresh: prefer one valu slot
                self.bundle.setdefault("valu", []).append(node.slot)
                self.free["valu"] -= 1
                p.lanes_done = p.lanes_total
                p.engine_choice = "valu"
                return True
            return self._spill_alu(node, p)   # else spill to alu

        # Atomic: alu / load / store / flow / vec_fma
        eng = p.native_engine
        if self.free[eng] == 0:
            return None
        self.bundle.setdefault(eng, []).append(node.slot)
        self.free[eng] -= 1
        p.lanes_done = p.lanes_total
        p.engine_choice = eng
        return True

    def _spill_alu(self, node: DNode, p: _Placement) -> bool | None:
        """Land as many remaining vec_elem lanes as fit on the alu unit."""
        take = min(p.lanes_total - p.lanes_done, self.free["alu"])
        if take == 0:
            return None
        for s in _vec_slot_to_alu_lanes(
                node.slot, range(p.lanes_done, p.lanes_done + take)):
            self.bundle.setdefault("alu", []).append(s)
        self.free["alu"] -= take
        p.lanes_done += take
        p.engine_choice = "alu"
        return p.lanes_done == p.lanes_total


def _make_picker(picker: str, placements: list[_Placement], rng: random.Random):
    """Return a sort key for a node index (lower = higher priority)."""
    if picker == "idx":
        return lambda idx: idx
    if picker == "random":
        return lambda idx: rng.random()
    # "fma_first" (default): vec_fma < vec_elem < rest < debug, then idx.
    def _key(idx):
        return (_KIND_PRIORITY.get(placements[idx].kind, 9), idx)
    return _key


def schedule(dag: DAG, *, seed: int | None = None,
             cap: int | None = None, picker: str = "fma_first") -> list[dict]:
    """List-schedule a ``DAG`` into VLIW bundles.

    Each cycle:
      1. Seed a priority queue (the working set) from the DAG's ready set.
      2. Pop nodes in priority order and place each via the ``FuncUnitPool``,
         which assigns units and fills slots. A fully-placed node commits
         immediately; the same-cycle WAR unlocks it produces are pushed
         straight back onto the queue (no re-scan of the working set).
      3. Append the cycle's bundle and advance (deferred RAW resolutions
         become ready next cycle).

    Spillable elementwise ``valu`` ops fall back to ``alu`` scalar lanes
    (sticky, splittable across cycles) when ``valu`` is saturated.
    ``vec_fma`` (``multiply_add``) is rigid - no scalar fma exists.

    A node that doesn't fully place this cycle - no unit had room
    (``place`` returns None) or a partial vec_elem spill (False) - is left
    uncommitted in the frontier; ``dag.ready()`` returns it next cycle with
    ``lanes_done`` carried over (placements persist across cycles). Slots
    never free mid-cycle, so neither case is retried within the same cycle.

    ``picker`` selects node ordering within a cycle:
      ``"fma_first"`` - vec_fma < vec_elem < rest < debug (default)
      ``"idx"``       - program order
      ``"random"``    - random key per node (uses ``seed``)

    Returns a list of bundles (``dict[engine, list[slot]]``), one per cycle
    that placed at least one slot. Debug-only bundles cost 0 cycles in the
    simulator (only bundles with a non-debug engine advance ``cycle``), so a
    trailing debug flush contributes nothing to the cycle count.
    """
    rng = random.Random(seed)
    placements = [_classify(n) for n in dag.nodes]

    if cap is None:
        cap = len(dag) + 1

    key_fn = _make_picker(picker, placements, rng)

    pool = FuncUnitPool()
    bundles: list[dict] = []
    C = 0
    committed = 0
    total = len(dag)

    while committed < total:
        ready = dag.ready()
        if not ready:
            raise RuntimeError(
                f"scheduler: frontier empty with {total - committed} "
                f"uncommitted nodes at C={C} - cyclic DAG or counter bug")

        pool.reset()
        # Working set for this cycle: a priority queue of the ready nodes.
        working: list[tuple] = []
        for idx in ready:
            heapq.heappush(working, (key_fn(idx), idx))

        # Pop in priority order; place each. A fully-placed node commits and
        # pushes its same-cycle WAR unlocks back onto the queue. Nodes that
        # don't fully place (None / False) fall through - they stay in the
        # frontier and come back next cycle with carried-over lanes_done.
        while working:
            _, idx = heapq.heappop(working)
            if pool.place(dag[idx], placements[idx]):
                committed += 1
                for u in dag.commit(idx):
                    heapq.heappush(working, (key_fn(u), u))

        # A cycle that placed nothing (ready was non-empty but no node fit any
        # unit) is stuck. A debug-only flush is fine - it costs 0 cycles.
        if not pool.bundle and committed < total:
            raise RuntimeError(
                f"scheduler: empty cycle at C={C} - stuck "
                f"({len(ready)} ready nodes but none were placeable)")

        bundles.append(pool.bundle)
        dag.advance()
        C += 1
        if C > cap:
            raise RuntimeError(
                f"scheduler: cycle count {C} exceeded cap {cap} - "
                f"regressed below unscheduled baseline")

    return bundles
