"""VLIW dependency DAG + scheduling infrastructure.

This module holds the DAG and the functional-unit/picker machinery shared by
the register allocator (``regalloc.py``), which builds the DAG and drives the
schedule. Two components live here:

  - ``DAG``: a dependency graph with built-in frontier management. It
    answers the single question the scheduler cares about each cycle:
    *which instructions have all data dependencies resolved and are ready
    for scheduling?*  All dependency bookkeeping (remaining blockers,
    buffered resolutions, committed flags, the frontier set) lives inside
    the DAG; nothing leaks onto the nodes. Nodes/edges are built by the
    caller (``regalloc.build_dag``) and handed in via ``DAG.from_nodes``.
  - ``FuncUnitPool`` / ``_make_picker`` / ``_classify``: the per-cycle
    slot-filling and node-ordering machinery the allocating scheduler uses.

Dependency model (the machine is read-before-write within a cycle: every
operand reads pre-cycle state, every write commits at end of cycle):

  - RAW (edge weight 1): producer -> consumer. The consumer becomes ready
    the cycle *after* the producer commits. Resolutions are *deferred* -
    buffered during the cycle and applied at ``advance()``.
  - WAR (edge weight 0): old reader -> new writer. The writer becomes
    ready the *same* cycle the reader commits. Resolutions are *immediate*
    - applied at ``commit()`` and reported back to the scheduler.

The regalloc DAG is RAW-only: tags are unique per write (SSA), so there are
no WAR/WAW anti-dependencies. The WAR machinery remains for generality but
carries no edges in practice.
"""

from dataclasses import dataclass, field
import heapq
import random
from typing import NamedTuple

from ir import Instr, Pause, VecElem, VecFma, VBroadcast, RegId, TaggedSlot, Gather
from problem import VLEN, SLOT_LIMITS

# A register id: (base_addr, is_vector), produced by Instr.reads()/writes().
# is_vector=True covers the 8-lane vector [addr..addr+7]; is_vector=False is
# the single word at addr. A vector register aliases the 8 scalars in its
# 8-word region (the gather's per-lane scalar writes fill one vector).
Reg = RegId

# Sentinel for "no downstream load" in dist_to_load (before normalization).
NO_LOAD = 10**6


class NodeProps(NamedTuple):
    """Static per-node scheduling properties, normalized to 0..1.
    Higher = more urgent for sink/raw/war/idx; for load LOWER = more urgent
    (0 = this node is a load; 1 = no downstream load)."""
    sink: float   # dist_to_sink / max  (longest RAW=1/WAR=0 path to a sink)
    load: float   # dist_to_load / max  (cycle-distance to nearest downstream load)
    raw: float    # #RAW dependents / max  (unblocked next cycle)
    war: float    # #WAR dependents / max  (unblocked same cycle)
    idx: float    # program order / total  (locality / determinism)


class Weights(NamedTuple):
    """Multiplier for each NodeProps term in the weighted picker's score
    (score = sink*props.sink - load*props.load + raw*props.raw
             + war*props.war + rigid*is_rigid_now + idx*props.idx;
    higher = scheduled first). load is subtracted because low dist_to_load =
    urgent."""
    sink: float
    load: float
    raw: float
    war: float
    rigid: float
    idx: float

# Flow ops that modify the PC - the DAG cannot represent control flow.
# The only such op in the IR is Pause (prologue/epilogue barrier); hitting
# one in the DAG means a barrier leaked into the body.
FLOW_PANIC = (Pause,)


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
    instr: Instr
    in_edges: list[tuple[int, int]] = field(default_factory=list)   # (src_idx, weight)
    out_edges: list[tuple[int, int]] = field(default_factory=list)  # (dst_idx, weight)


class DAG:
    """Dependency graph with built-in frontier management.

    Answers: *which instructions have all data dependencies resolved and
    are ready for scheduling?*

    Nodes/edges are built by the caller (``regalloc.build_dag``, RAW
    weight-1 only) and handed in via ``DAG.from_nodes``. The frontier is
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

    @classmethod
    def from_nodes(cls, nodes: list[DNode]) -> "DAG":
        """Build a DAG directly from a pre-built node list (with edges already
        wired), re-deriving dynamic scheduling state and static props. This
        is the only construction path: callers (``regalloc.build_dag`` and
        compaction passes like ``prune_to_stores``) wire nodes/edges
        themselves and hand the list in."""
        dag = cls.__new__(cls)
        dag.nodes = nodes
        dag._finish_init()
        return dag

    def _finish_init(self) -> None:
        """(Re)derive dynamic scheduling state + static props from the current
        node/edge lists. Called by ``from_nodes`` after construction and by
        compaction passes after filtering."""
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

        self.props: list[NodeProps] = self._compute_props()

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

    def reset(self) -> None:
        """Reset dynamic scheduling state for re-scheduling the same DAG
        (e.g. sweeping picker weights). Static graph + props are untouched."""
        n = len(self.nodes)
        for node in self.nodes:
            self._raw[node.idx] = sum(1 for _, w in node.in_edges if w == 1)
            self._war[node.idx] = sum(1 for _, w in node.in_edges if w == 0)
        self._pending = [0] * n
        self._committed = [False] * n
        self._frontier = {i for i in range(n)
                          if self._raw[i] == 0 and self._war[i] == 0}

    # -- construction -----------------------------------------------------

    def _compute_props(self) -> list[NodeProps]:
        """Static per-node scheduling properties (normalized to 0..1), derived
        once from the DAG (reverse program order is a topological order since
        all edges go low->high idx):
          sink - dist_to_sink: longest cycle-weighted path (RAW=1, WAR=0) to a
                 sink. Higher = feeds a longer chain = more urgent.
          load - dist_to_load: cycle-distance to the nearest downstream load
                 (0 for loads). Lower = feeds the gather sooner = more urgent;
                 1.0 = no downstream load.
          raw  - #RAW dependents (unblocked next cycle). Higher = more urgent.
          war  - #WAR dependents (unblocked same cycle). Higher = more urgent.
        """
        n = len(self.nodes)
        n_raw = [sum(1 for _, w in node.out_edges if w == 1) for node in self.nodes]
        n_war = [sum(1 for _, w in node.out_edges if w == 0) for node in self.nodes]
        dist_to_sink = [0] * n
        dist_to_load = [NO_LOAD] * n
        for node in reversed(self.nodes):
            best_sink = 0
            best_load = NO_LOAD
            for dst, w in node.out_edges:
                d = w + dist_to_sink[dst]
                if d > best_sink:
                    best_sink = d
                dl = dist_to_load[dst]
                if dl != NO_LOAD:
                    d2 = w + dl
                    if d2 < best_load:
                        best_load = d2
            dist_to_sink[node.idx] = best_sink
            dist_to_load[node.idx] = 0 if node.engine == "load" else best_load
        max_sink = max(dist_to_sink) or 1
        max_load = max((d for d in dist_to_load if d != NO_LOAD), default=1) or 1
        max_raw = max(n_raw) or 1
        max_war = max(n_war) or 1
        return [NodeProps(dist_to_sink[i] / max_sink,
                          1.0 if dist_to_load[i] == NO_LOAD else dist_to_load[i] / max_load,
                          n_raw[i] / max_raw,
                          n_war[i] / max_war,
                          i / (n - 1) if n > 1 else 0.0)
                for i in range(n)]


# ---------------------------------------------------------------------------
# Dead-code pruning
# ---------------------------------------------------------------------------

def prune_to_stores(dag: DAG) -> DAG:
    """Prune nodes that do not contribute to the final stores, returning a new
    compacted DAG (the input is unchanged).

    Pass 1: backward walk from the store sinks following RAW (weight-1,
    true data dependency) edges only, marking nodes "useful". WAR edges are
    anti-dependencies (register-reuse ordering), not data flow, so they
    neither mark nor are walked.

    Pass 2: drop every unmarked node and its attached edges. Debug nodes
    inherit the usefulness of their producers: a debug node is kept iff all
    its RAW producers are kept (vacuously true when it has none - e.g. it
    reads prologue state outside the body DAG), preserving the dev oracle
    exactly where the asserted value is still computed.

    The kept subgraph needs no dependency re-analysis: kept-kept edges are
    unchanged (they are exactly the induced subgraph), and only edges
    incident to removed nodes disappear. WAW safety is preserved: a kept
    writer W1 of a lane is useful, so it has a kept reader R' on a RAW path
    to a store, and R' necessarily precedes the next kept writer W2 of that
    lane - the bridge W1 ->RAW R' ->WAR W2 survives pruning. (A writer with
    no kept reader before the next writer has no RAW path to a store and is
    pruned.)

    Counter/frontier/props are re-derived from the filtered edge lists via
    ``_finish_init`` (so ``dist_to_sink`` is re-anchored on the real sinks).
    """
    stores = [n.idx for n in dag.nodes if n.engine == "store"]
    assert stores, "prune_to_stores: no store nodes - nothing to anchor on"

    useful: set[int] = set(stores)
    stack = list(stores)
    while stack:
        i = stack.pop()
        for src, w in dag.nodes[i].in_edges:
            if w == 1 and src not in useful:
                useful.add(src)
                stack.append(src)

    keep = set(useful)
    for n in dag.nodes:
        if n.engine == "debug" and n.idx not in keep:
            if all(src in useful for src, w in n.in_edges if w == 1):
                keep.add(n.idx)

    remap: dict[int, int] = {}
    new_nodes: list[DNode] = []
    for old in dag.nodes:
        if old.idx not in keep:
            continue
        nn = DNode(idx=len(new_nodes), engine=old.engine, instr=old.instr)
        remap[old.idx] = nn.idx
        new_nodes.append(nn)
    for old in dag.nodes:
        if old.idx not in keep:
            continue
        nn = new_nodes[remap[old.idx]]
        nn.in_edges = [(remap[src], w) for src, w in old.in_edges if src in keep]
        nn.out_edges = [(remap[dst], w) for dst, w in old.out_edges if dst in keep]

    new = DAG.from_nodes(new_nodes)
    return new


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

# Node classification tags.
_KIND_ATOMIC_SCALAR = "alu_scalar"
_KIND_LOAD = "load"
_KIND_GATHER = "gather"               # VLEN scalar loads, partial completion
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
    instr = n.instr
    eng = instr.engine
    if eng == "alu":
        return _Placement(_KIND_ATOMIC_SCALAR, 1, "alu")
    if eng == "load":
        if isinstance(instr, Gather):
            return _Placement(_KIND_GATHER, VLEN, "load")  # 8 scalar loads
        return _Placement(_KIND_LOAD, 1, "load")
    if eng == "store":
        return _Placement(_KIND_STORE, 1, "store")
    if eng == "flow":
        return _Placement(_KIND_FLOW, 1, "flow")
    if eng == "debug":
        return _Placement(_KIND_DEBUG, 0, "debug")
    if eng == "valu":
        if isinstance(instr, VecFma):
            return _Placement(_KIND_VEC_FMA, 1, "valu")    # rigid: no scalar fma
        return _Placement(_KIND_VEC_ELEM, VLEN, "valu")    # spillable to alu
    raise NotImplementedError(f"Unknown engine: {eng}")


def _vec_instr_to_alu_lanes(instr: Instr, lanes) -> list[tuple]:
    """Materialise one elementwise ``valu`` instruction as per-lane ``alu`` tuples.

    VecElem(op, dest, a1, a2)  -> lane j: (op, dest+j, a1+j, a2+j)
    VBroadcast(dest, src)      -> lane j: ("+", dest+j, src, 0)
    VecFma cannot spill (no scalar fma in the ISA) and raises.

    Each lane is a ``TaggedSlot`` carrying the source instruction's stable id
    so a tracing simulator can attribute the spilled lane to its origin.
    """
    if isinstance(instr, VecFma):
        raise NotImplementedError("multiply_add cannot spill to alu (no scalar fma)")
    rid = instr.rid
    if isinstance(instr, VBroadcast):
        return [TaggedSlot(("+", instr.dest.addr + j, instr.src.addr_of(), 0), rid)
                for j in lanes]
    assert isinstance(instr, VecElem)
    return [TaggedSlot(
                (instr.op, instr.dest.addr + j, instr.a1.addr + j, instr.a2.addr + j),
                rid)
            for j in lanes]


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

    def place(self, node: DNode, p: _Placement) -> bool | None:
        """Try to place a single node this cycle.

        Updates occupation state and ``p`` (lanes landed / sticky engine)
        when the node is placed. Returns:

          True  - yes: placed and complete (caller commits to the DAG)
          False - partial: placed some lanes, needs more cycles
          None  - no: no unit had room; nothing changed
        """
        if p.kind == _KIND_DEBUG:
            self.bundle.setdefault("debug", []).append(node.instr)
            return True

        if p.kind == _KIND_VEC_ELEM:
            if p.lanes_done > 0:              # sticky alu continuation
                return self._spill_alu(node, p)
            if self.free["valu"] > 0:         # fresh: prefer one valu slot
                self.bundle.setdefault("valu", []).append(node.instr)
                self.free["valu"] -= 1
                p.lanes_done = p.lanes_total
                p.engine_choice = "valu"
                return True
            return self._spill_alu(node, p)   # else spill to alu

        # Atomic: alu / load / store / flow / vec_fma
        eng = p.native_engine
        if self.free[eng] == 0:
            return None
        self.bundle.setdefault(eng, []).append(node.instr)
        self.free[eng] -= 1
        p.lanes_done = p.lanes_total
        p.engine_choice = eng
        return True

    def _spill_alu(self, node: DNode, p: _Placement) -> bool | None:
        """Land as many remaining vec_elem lanes as fit on the alu unit."""
        take = min(p.lanes_total - p.lanes_done, self.free["alu"])
        if take == 0:
            return None
        for s in _vec_instr_to_alu_lanes(
                node.instr, range(p.lanes_done, p.lanes_done + take)):
            self.bundle.setdefault("alu", []).append(s)
        self.free["alu"] -= take
        p.lanes_done += take
        p.engine_choice = "alu"
        return p.lanes_done == p.lanes_total


def _make_picker(picker: str, placements: list[_Placement], rng: random.Random,
                 props: list[NodeProps] | None = None,
                 weights: Weights | None = None):
    """Return a sort key for a node index (lower = higher priority)."""
    if picker == "idx":
        return lambda idx: idx
    if picker == "random":
        return lambda idx: rng.random()
    if picker == "fma_first":
        # vec_fma < vec_elem < rest < debug, then idx.
        def _key(idx):
            return (_KIND_PRIORITY.get(placements[idx].kind, 9), idx)
        return _key
    if picker == "weighted":
        # score = sink*sink - load*load + raw*raw + war*war + rigid*is_rigid_now
        #         + idx*idx;
        # higher = scheduled first (max-heap via negation). is_rigid_now is
        # mutable placement state: a node is rigid unless it's a fresh
        # (un-spilled) vec_elem. Computed at push time, so partial vec_elem
        # (lanes_done>0 -> sticky alu) read as rigid when re-pushed next cycle.
        w = weights
        def _key(idx):
            p = props[idx]
            pl = placements[idx]
            rigid = (pl.kind != _KIND_VEC_ELEM) or (pl.lanes_done > 0)
            score = (w.sink * p.sink - w.load * p.load
                     + w.raw * p.raw + w.war * p.war
                     + w.rigid * (1 if rigid else 0)
                     + w.idx * p.idx)
            return -score
        return _key
    raise ValueError(f"Unknown picker: {picker}")
