"""Two-phase register allocation: SSA tag chains + schedule-time allocation.

This is the kernel's register-allocation path (it replaced a one-pass FIFO
rename engine that assigned physical addresses up front, forcing false
WAR/WAW sharing of recycled homes). It:

  1. ``tag_raw_chains`` - rewrites each symbolic operand into a unique version
     tag, one per RAW chain (a write and all reads of that written value).
     Tags are unique by construction, so the DAG needs only RAW edges - no
     WAR, no WAW, no false dependencies. Also records ``read_count[tag]`` =
     how many reads consume the version.

  2. ``build_dag`` - a read of tag T gets a weight-1 (RAW) edge from T's
     unique writer. Pinned symbols (const_vec_0) have no writer and no edge.

  3. ``RegisterAllocator`` + ``schedule`` - physical registers are assigned
     when a RAW chain starts (its writer is placed) and freed when all its
     reads have committed. A write is placeable only when a functional unit
     slot AND a free register are both available.

Gather stays a single DAG node (vector read of addr, vector write of nv,
cost VLEN load slots with partial completion, like the vec_elem spill) so
the "grouped" gather stays atomic-ish: once started, it finishes before
another gather on the same unit.
"""

from collections import deque

from ir import Sym, Reg, LaneRef, Instr, TaggedSlot
from problem import VLEN, SCRATCH_SIZE
from scheduler import (DAG, DNode, FuncUnitPool, _classify, _make_picker,
                       _Placement, _KIND_DEBUG, _KIND_VEC_ELEM,
                       _KIND_ATOMIC_SCALAR, _KIND_GATHER, Weights)


# ===========================================================================
# Step 1: SSA tag chains
# ===========================================================================

def _base(op):
    """The base symbol of an operand (LaneRef views its vector)."""
    return op.vec if isinstance(op, LaneRef) else op


def tag_raw_chains(instrs, pinned=()):
    """Rewrite symbolic operands into unique RAW-chain version tags.

    Forward program-order scan (read-before-write per instruction, matching
    the machine's read-before-write semantics):

      - read of sym  -> current version tag of sym; read_count[tag] += 1
      - write of sym -> a fresh version tag; current[sym] = it

    A tag is a fresh ``Sym`` named ``"<sym>#<n>"`` (unique per RAW chain).
    ``read_count`` counts read *operand occurrences* (a self-read like
    ``a + a`` counts 2), matching the scheduler's per-operand decrement.

    Pinned symbols (const_vec_0) are left untagged - they are read-only and
    resolve to their pinned address with no dependency.

    Returns (tagged_instrs, read_count).
    """
    pinned = set(pinned)
    current: dict[Sym, Sym] = {}
    read_count: dict[Sym, int] = {}
    counter = [0]
    tagged = []

    def fresh(sym: Sym) -> Sym:
        n = counter[0]
        counter[0] += 1
        tag = Sym(f"{sym.name}#{n}", sym.is_vec)
        read_count[tag] = 0
        return tag

    def is_pinned(b) -> bool:
        return b.name in pinned if isinstance(b, Sym) else b in pinned

    for instr in instrs:
        rd_ops = instr.read_operands()
        wr_ops = instr.write_operands()
        # Reads first (read-before-write): resolve to the current version.
        new_rd = []
        for op in rd_ops:
            b = _base(op)
            if is_pinned(b):
                new_rd.append(op)
                continue
            tag = current[b]
            read_count[tag] += 1
            new_rd.append(LaneRef(tag, op.j) if isinstance(op, LaneRef) else tag)
        # Then writes: a fresh version per write.
        new_wr = []
        for op in wr_ops:
            b = _base(op)
            if is_pinned(b):
                new_wr.append(op)
                continue
            tag = fresh(b)
            current[b] = tag
            new_wr.append(LaneRef(tag, op.j) if isinstance(op, LaneRef) else tag)
        tagged.append(instr.rebuild(new_rd, new_wr))

    return tagged, read_count


def recompute_read_count(instrs, pinned=()):
    """Recompute read_count over a (possibly pruned) instruction list.

    After pruning drops nodes (e.g. debug or dead code), the read counts from
    tag_raw_chains (computed over the full stream) overcount for tags whose
    readers were dropped - their registers would never free. Recompute counts
    from the surviving instructions' read operands so freeing matches what
    will actually commit.
    """
    counts: dict[Sym, int] = {}
    for instr in instrs:
        for op in instr.read_operands():
            b = _base(op)
            if b.name in pinned:
                continue
            counts[b] = counts.get(b, 0) + 1
    return counts


# ===========================================================================
# Step 2: RAW-only DAG
# ===========================================================================

def build_dag(tagged, pinned=()):
    """Build a RAW-only DAG from tagged instructions.

    Each tag has exactly one writer, so a read operand maps to a weight-1
    edge from that writer. Pinned reads (const_vec_0) have no writer and no
    edge (always ready). No WAR, no WAW - tags are unique by construction.
    """
    pinned = set(pinned)
    nodes: list[DNode] = []
    writer: dict[Sym, int] = {}
    for idx, instr in enumerate(tagged):
        node = DNode(idx=idx, engine=instr.engine, instr=instr)
        nodes.append(node)
        seen: set[int] = set()
        for op in instr.read_operands():
            b = _base(op)
            if b.name in pinned:
                continue
            # A tag with no writer in this list was written externally (e.g.
            # the linear prologue, which commits before the DAG-scheduled
            # body) - its value is already available, so no edge.
            src = writer.get(b)
            if src is None:
                continue
            if src not in seen:
                seen.add(src)
                node.in_edges.append((src, 1))
                nodes[src].out_edges.append((idx, 1))
        for op in instr.write_operands():
            b = _base(op)
            if b.name in pinned:
                continue
            writer[b] = idx
    return DAG.from_nodes(nodes)


# ===========================================================================
# Step 3: register allocator + allocating scheduler
# ===========================================================================

# Debug: dump per-cycle live-register pressure to find leaks.
_DEBUG_PRESSURE = False
_pressure_log = open("pressure.log", "w") if _DEBUG_PRESSURE else None

# Below this many free vector granules, the pressured_key priority kicks in
# to prefer register-freeing nodes over register-allocating ones.
PRESSURE_THRESHOLD = 32


class RegisterAllocator:
    """Assigns physical registers to RAW-chain tags at schedule time.

    A tag's register is allocated when its writer is placed (sticky across a
    partially-completed node like a gather or a spilled vec_elem) and freed
    when all its reads have committed. Register availability is checked
    alongside functional-unit availability when placing a write.

    Dynamic scalar/vector allocation: there is ONE vector free list (all of
    scratch as 8-aligned granules, const_vec_0 pinned at granule 0). The
    scalar engine holds a free-WORD list; when it runs out it claims a
    vector granule and splits it into 8 scalar words, and when all 8 words
    of a claimed granule are free again the granule returns to the vector
    list. No static scalar reserve: the split is elastic in both directions.
    """

    def __init__(self, read_count, pinned_addr0=True):
        self._read_count = read_count
        # const_vec_0 occupies address 0 (granule 0); all other granules
        # (VLEN..SCRATCH_SIZE step VLEN) are the vector pool.
        self.free_vec = deque(range(VLEN, SCRATCH_SIZE, VLEN))
        # Scalar engine: free words + per-granule bookkeeping for claimed
        # (vector->scalar converted) granules.
        self._scalar_words: deque[int] = deque()     # free scalar words
        self._word_granule: dict[int, int] = {}      # word -> its granule base
        self._granule_free: dict[int, int] = {}      # claimed granule -> #free words
        self._claim_of: dict[int, int] = {}          # word allocated by a fresh claim -> granule
        # Pool sizes (for pressure features / normalization). The scalar
        # pool is elastic: its ceiling is the whole granule space in words.
        self.vec_pool_size = len(self.free_vec)
        self.scalar_pool_size = self.vec_pool_size * VLEN
        self.assigned: dict[Sym, int] = {}     # tag -> base addr
        self.remaining: dict[Sym, int] = {}    # tag -> reads left before free
        self.exhaustion_warnings = 0
        self.n_alloc = 0
        self.n_free = 0
        # addr -> (base name, length) for the simulator's debug scratch map.
        # Recorded at allocation time; tags strip the "#n" version suffix.
        self._names: dict[int, tuple[str, int]] = {0: ("const_vec_0", VLEN)}
        # Undo log for checkpoint()/rollback() (None = not recording).
        self._log: list | None = None

    @property
    def free_scalar(self):
        """The scalar free list (words), as a read-only view."""
        return self._scalar_words

    def can_write(self, tag: Sym) -> bool:
        """A register for ``tag`` is already assigned or one is available."""
        if tag in self.assigned:
            return True
        if tag.is_vec:
            return len(self.free_vec) > 0
        return len(self._scalar_words) > 0 or len(self.free_vec) > 0

    def write(self, tag: Sym) -> int:
        """Allocate (or return the sticky) register for ``tag``; returns addr."""
        if tag in self.assigned:
            return self.assigned[tag]
        claimed = False
        if tag.is_vec:
            if not self.free_vec:
                raise RuntimeError(
                    f"register exhaustion: no free vector register for "
                    f"{tag.name} ({len(self.assigned)} live)")
            addr = self.free_vec.popleft()
        else:
            if not self._scalar_words:
                # Scalar pool empty: claim a vector granule, split into words.
                if not self.free_vec:
                    raise RuntimeError(
                        f"register exhaustion: no free scalar word for "
                        f"{tag.name} ({len(self.assigned)} live)")
                g = self.free_vec.popleft()
                for w in range(g, g + VLEN):
                    self._scalar_words.append(w)
                    self._word_granule[w] = g
                self._granule_free[g] = VLEN
                claimed = True
            addr = self._scalar_words.popleft()
            g = self._word_granule[addr]
            self._granule_free[g] -= 1
            if claimed:
                self._claim_of[addr] = g
        self.assigned[tag] = addr
        self.remaining[tag] = self._read_count[tag]
        self.n_alloc += 1
        # Record the base name (strip the "#n" version suffix) for debug_map.
        old_name = self._names.get(addr)
        self._names[addr] = (tag.name.split("#")[0], VLEN if tag.is_vec else 1)
        if self._log is not None:
            self._log.append(("W" if tag.is_vec else "Ws",
                              tag, addr, old_name, claimed))
        return addr

    def unwrite(self, tag: Sym) -> None:
        """Roll back a speculative ``write`` whose node landed no lane this
        cycle. Only valid if the tag was just allocated (no reads consumed
        yet); a sticky tag allocated by an earlier placement is left alone."""
        if self.remaining.get(tag) == self._read_count[tag] and tag in self.assigned:
            addr = self.assigned.pop(tag)
            del self.remaining[tag]
            was_claim = addr in self._claim_of
            path = None
            if tag.is_vec:
                self.free_vec.appendleft(addr)
            else:
                path = self._scalar_dealloc(addr)
            if self._log is not None:
                self._log.append(("U", tag, addr, was_claim, path))

    def _scalar_dealloc(self, addr: int) -> str:
        """Return a scalar word to the free engine (used by unwrite and by
        rollback of a scalar allocation). Two self-contained paths:

          "rev" - full claim reversal: the word came from a fresh claim and
                  all 7 sibling words are still free (the only checked-out
                  word is this one) -> the granule never happened.
          "ret" - plain word return: bump the granule's free count and put
                  the word back. Granule RECLAMATION IS NOT DONE HERE - it
                  is deferred to cycle end (collect_scalar_garbage),
                  outside the trial/checkpoint window, so it never needs
                  to be rolled back.
        """
        g = self._word_granule[addr]
        if self._claim_of.get(addr) == g and self._granule_free[g] == VLEN - 1:
            del self._claim_of[addr]
            for w in range(g + 1, g + VLEN):
                self._scalar_words.remove(w)
                del self._word_granule[w]
            del self._word_granule[g]
            del self._granule_free[g]
            self.free_vec.appendleft(g)
            return "rev"
        self._claim_of.pop(addr, None)
        self._granule_free[g] += 1
        self._scalar_words.appendleft(addr)
        return "ret"

    def read(self, tag: Sym) -> None:
        """One read of ``tag`` committed; free its register when all are done."""
        self.remaining[tag] -= 1
        if self._log is not None:
            self._log.append(("R", tag))
        if self.remaining[tag] == 0:
            addr = self.assigned.pop(tag)
            del self.remaining[tag]
            self.n_free += 1
            if tag.is_vec:
                self.free_vec.append(addr)
                if self._log is not None:
                    self._log.append(("F", tag, addr))
            else:
                g = self._word_granule[addr]
                self._granule_free[g] += 1
                self._scalar_words.append(addr)
                was_claim = self._claim_of.pop(addr, None) is not None
                if self._log is not None:
                    self._log.append(("Fs", tag, addr, was_claim))
                # NB: no granule reclaim here - deferred to cycle end.

    def collect_scalar_garbage(self) -> None:
        """Cycle-end scalar GC: return fully-free claimed granules to the
        vector pool. Called by the scheduler AFTER the cycle's decisions are
        committed (outside any checkpoint/rollback window), so it needs no
        undo support."""
        reclaimable = [g for g, cnt in self._granule_free.items()
                       if cnt == VLEN]
        if not reclaimable:
            return
        reclaim = set(reclaimable)
        self._scalar_words = deque(
            w for w in self._scalar_words if self._word_granule[w] not in reclaim)
        for g in reclaimable:
            for w in range(g, g + VLEN):
                del self._word_granule[w]
            del self._granule_free[g]
            self.free_vec.append(g)

    # -- checkpoint / rollback --------------------------------------------
    # Same interface as DAG.checkpoint()/rollback(): mutations above append
    # inverse records to ``_log`` while it is open; rollback(token) undoes
    # everything past the token in reverse. Rolling back to token 0 closes
    # the log. Diagnostic counters (n_alloc/n_free/exhaustion_warnings) are
    # intentionally NOT rolled back (monotonic stats; per-trial deltas are
    # computed by subtraction around a trial).

    def checkpoint(self) -> int:
        """Start recording mutations; returns an opaque rollback token."""
        if self._log is None:
            self._log = []
        return len(self._log)

    def rollback(self, token: int) -> None:
        """Undo all mutations recorded since ``token`` (reverse order)."""
        log = self._log
        assert log is not None, "rollback without an open checkpoint"
        while len(log) > token:
            entry = log.pop()
            kind = entry[0]
            if kind == "W":                     # undo a vector allocation
                _, tag, addr, old_name, _ = entry
                del self.assigned[tag]
                del self.remaining[tag]
                self.free_vec.appendleft(addr)
                self._restore_name(addr, old_name)
            elif kind == "Ws":                  # undo a scalar allocation
                _, tag, addr, old_name, claimed = entry
                del self.assigned[tag]
                del self.remaining[tag]
                if claimed:
                    g = self._claim_of.pop(addr)
                    for w in range(g + 1, g + VLEN):
                        self._scalar_words.remove(w)
                        del self._word_granule[w]
                    del self._word_granule[g]
                    del self._granule_free[g]
                    self.free_vec.appendleft(g)
                else:
                    g = self._word_granule[addr]
                    self._granule_free[g] += 1
                    self._scalar_words.appendleft(addr)
                self._restore_name(addr, old_name)
            elif kind == "R":                   # undo a consumed read
                self.remaining[entry[1]] += 1
            elif kind == "F":                   # undo a vector free
                _, tag, addr = entry
                self.free_vec.remove(addr)
                self.assigned[tag] = addr
                self.remaining[tag] = 0
            elif kind == "Fs":                  # undo a scalar free
                _, tag, addr, was_claim = entry
                g = self._word_granule[addr]
                self._granule_free[g] -= 1
                self._scalar_words.remove(addr)
                if was_claim:
                    self._claim_of[addr] = g
                self.assigned[tag] = addr
                self.remaining[tag] = 0
            elif kind == "Rc":                  # unreachable (deferred GC)
                raise AssertionError("Rc entries must not exist: granule "
                                     "reclaim is deferred to cycle end")
            elif kind == "U":                   # undo an unwrite (redo write)
                _, tag, addr, was_claim, path = entry
                if tag.is_vec:
                    self.free_vec.remove(addr)
                elif path == "rev":
                    g = addr            # a claim hands out the granule base
                    self._claim_of[addr] = g
                    self.free_vec.remove(g)
                    for w in range(g, g + VLEN):
                        self._scalar_words.append(w)
                        self._word_granule[w] = g
                    self._granule_free[g] = VLEN - 1
                    self._scalar_words.remove(addr)
                else:                   # "ret": plain word return was made
                    g = self._word_granule[addr]
                    self._granule_free[g] -= 1
                    self._scalar_words.remove(addr)
                    if was_claim:
                        self._claim_of[addr] = g
                self.assigned[tag] = addr
                self.remaining[tag] = self._read_count[tag]
            else:
                raise AssertionError(f"unknown allocator log entry: {entry}")
        if token == 0:
            self._log = None

    def _restore_name(self, addr, old_name):
        if old_name is None:
            self._names.pop(addr, None)
        else:
            self._names[addr] = old_name

    # -- queries for the scheduler ----------------------------------------

    def addr_of(self, op):
        """The physical base address assigned to an operand's tag. Pinned
        symbols (const_vec_0) resolve to address 0 (no allocation)."""
        if isinstance(op, LaneRef):
            base = op.vec
            if base not in self.assigned:
                return 0 + op.j      # pinned const_vec_0 lane
            return self.assigned[base] + op.j
        if isinstance(op, Sym) and op not in self.assigned:
            return 0                   # pinned const_vec_0
        return self.assigned[op]

    def debug_map(self) -> dict[int, tuple[str, int]]:
        """addr -> (base name, length) for the simulator's debug scratch map."""
        return dict(self._names)


def _resolve_operands(instr, allocator):
    """Rebuild an instruction with physical Reg operands from the allocator."""
    rd = [Reg(allocator.addr_of(o), getattr(o, "is_vec", False))
          for o in instr.read_operands()]
    wr = [Reg(allocator.addr_of(o), getattr(o, "is_vec", False))
          for o in instr.write_operands()]
    return instr.rebuild(rd, wr)


def schedule(dag: DAG, read_count, *, seed: int | None = None,
             picker: str = "fma_first", weights: Weights | None = None,
             cap: int | None = None, allocator: RegisterAllocator | None = None,
             prio=None):
    """List-schedule a RAW-only DAG, allocating registers at schedule time.

    A standard list-scheduling loop over the DAG's ready set (driven by the
    shared ``FuncUnitPool``/picker), but resolving tags to physical registers
    via a ``RegisterAllocator`` as nodes are placed. A write node is placeable
    only when a unit slot AND a free register are both available; reads are
    consumed (and their register freed) on commit. Pass ``allocator`` to
    continue an existing allocation (e.g. from a linearly-emitted prologue);
    a fresh one is created otherwise.

    ``prio``: optional priority-fn factory ``(allocator, base_key_fn) ->
    key_fn`` to override node ordering (default: pressure-aware freeing bias).
    """
    import heapq
    import random

    rng = random.Random(seed)
    if allocator is None:
        allocator = RegisterAllocator(read_count)
    for n in dag.nodes:
        _classify(n)
    if cap is None:
        cap = len(dag) + 1
    key_fn = _make_picker(picker, rng, weights)

    # Register-freeing priority: a node that is the LAST reader of a tag
    # frees that tag's register when it commits. ALWAYS prefer such nodes
    # (better than threshold-gated: chains finish, freeing registers, instead
    # of piling up un-freed temps). This bias is also required for the
    # schedule to complete at all under register pressure.
    def freeing_read(idx) -> int:
        """#registers this node's commit would free (reads with 1 remaining)."""
        n = 0
        for op in dag[idx].instr.read_operands():
            b = _base(op)
            if allocator.remaining.get(b) == 1:
                n += 1
        return n

    def pressured_key(idx):
        base = key_fn(dag[idx])
        b0 = base[0] if isinstance(base, tuple) else base
        fw = weights.freeing if weights is not None else 1000.0
        return b0 - freeing_read(idx) * fw

    active_key = prio(allocator, key_fn, freeing_read) if prio else pressured_key

    pool = FuncUnitPool()
    bundles: list[dict] = []
    C = 0
    committed = 0
    total = len(dag)

    while committed < total:
        ready = dag.ready()
        if not ready:
            raise RuntimeError(
                f"regalloc schedule: frontier empty with {total - committed} "
                f"uncommitted nodes at C={C} - cyclic DAG or counter bug")
        pool.reset()
        working = [(active_key(i), i) for i in ready]
        heapq.heapify(working)

        while working:
            _, idx = heapq.heappop(working)
            node = dag[idx]
            p = node.placement
            if p.kind == _KIND_DEBUG:
                # Debug nodes write nothing but DO read a tag (their loc):
                # resolve it and consume the read (free at 0) like any reader.
                resolved_node = DNode(idx=node.idx, engine=node.engine,
                                      instr=_resolve_operands(node.instr, allocator))
                pool.place(resolved_node, p)
                committed += 1
                for op in node.instr.read_operands():
                    b = _base(op)
                    if b in allocator.assigned:
                        allocator.read(b)
                for u in dag.commit(idx):
                    heapq.heappush(working, (active_key(u), u))
                continue
            if p.kind == _KIND_GATHER:
                # Gather = one node emitting VLEN scalar loads (partial
                # completion). Allocate nv (sticky), read addr, land as many
                # lanes as free load slots allow; commit reads on completion.
                wr_tags = [_base(o) for o in node.instr.write_operands()]
                if not all(allocator.can_write(t) for t in wr_tags):
                    allocator.exhaustion_warnings += 1
                    continue
                dest = allocator.write(wr_tags[0])     # nv vector granule
                addr = allocator.addr_of(node.instr.addr)
                take = min(p.lanes_total - p.lanes_done, pool.free["load"])
                if take == 0:
                    continue                           # no load slot this cycle
                for j in range(p.lanes_done, p.lanes_done + take):
                    pool.bundle.setdefault("load", []).append(
                        ("load", dest + j, addr + j))
                pool.free["load"] -= take
                p.lanes_done += take
                if p.lanes_done == p.lanes_total:      # all 8 lanes landed
                    committed += 1
                    for op in node.instr.read_operands():
                        b = _base(op)
                        if b in allocator.assigned:
                            allocator.read(b)
                    for u in dag.commit(idx):
                        heapq.heappush(working, (active_key(u), u))
                continue
            # Register check: every write tag must be allocatable.
            wr_tags = [_base(o) for o in node.instr.write_operands()]
            if not all(allocator.can_write(t) for t in wr_tags):
                allocator.exhaustion_warnings += 1
                continue   # leave for next cycle (a read may free a register)
            # Allocate write registers, then resolve operands to physical Regs
            # NOW - while every read tag is still assigned (they free only
            # after this node commits). The pool then works with resolved
            # operands (.addr / spill / lower).
            for t in wr_tags:
                allocator.write(t)
            resolved_instr = _resolve_operands(node.instr, allocator)
            resolved_node = DNode(idx=node.idx, engine=node.engine,
                                  instr=resolved_instr)
            placed = pool.place(resolved_node, p)
            if placed is None:
                # No lane landed this cycle: roll back the speculative allocs -
                # but ONLY if nothing was ever landed (lanes_done==0). A node
                # with lanes already written into dest keeps its register.
                if p.lanes_done == 0:
                    for t in wr_tags:
                        allocator.unwrite(t)
                continue
            if placed:   # fully placed -> commit reads, free at 0
                committed += 1
                for op in node.instr.read_operands():
                    b = _base(op)
                    if b in allocator.assigned:
                        allocator.read(b)
                for u in dag.commit(idx):
                    heapq.heappush(working, (active_key(u), u))

        # Emit the cycle's bundle (slots are already resolved/lowered).
        bundle = {eng: [s.tagged_lower() if isinstance(s, Instr) else s
                        for s in slots]
                  for eng, slots in pool.bundle.items()}
        if _DEBUG_PRESSURE:
            _pressure_log.write(
                f"cycle {C}: live_vec={len(allocator.assigned)} "
                f"free_vec={len(allocator.free_vec)} committed={committed}\n")
        if not bundle and committed < total:
            raise RuntimeError(
                f"regalloc schedule: empty cycle at C={C} - stuck "
                f"({len(ready)} ready nodes but none were placeable; "
                f"register exhaustion warnings so far: "
                f"{allocator.exhaustion_warnings})")
        bundles.append(bundle)
        dag.advance()
        # Cycle-end scalar GC: return fully-free claimed granules to the
        # vector pool (committed state, outside any checkpoint).
        allocator.collect_scalar_garbage()
        C += 1
        if C > cap:
            raise RuntimeError(
                f"regalloc schedule: cycle count {C} exceeded cap {cap}")

    if allocator.exhaustion_warnings:
        print(f"regalloc: {allocator.exhaustion_warnings} register-exhaustion "
              f"stalls during scheduling")
    return bundles
