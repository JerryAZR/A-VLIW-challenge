"""Rollout scheduler: per-cycle trial-and-score list scheduling.

Instead of committing to a single global priority order (the greedy
``regalloc.schedule``), this scheduler treats each cycle as a small search
problem:

  1. checkpoint the three mutable structures (DAG / allocator / placements)
     via their uniform ``checkpoint()`` / ``rollback(token)`` interface,
  2. run K trial orderings of the (fixed, RAW-only) ready set - trial 0 is
     the incumbent weighted-greedy order, the rest are uniform shuffles -
     each trial simulating *placement decisions only* (no operand
     resolution, no slot emission; the pool is per-cycle scratch and needs
     no rollback),
  3. score each trial's post-``advance()`` state (slots filled per engine,
     register pressure, next-cycle frontier size, commits),
  4. roll every trial back, then replay the winner's ordering with emission
     on. Placement is deterministic given (state, order), so the replay
     reproduces the winning end state exactly - and runs unlogged, since
     all trial checkpoints were rolled back to token 0.

The point: register-pressure deadlocks (ready nodes that all need a
register none can free) are invisible to a greedy priority but visible one
cycle ahead in the scored state, so the search naturally staggers
register-hungry work (gathers, hash chains) before the pool drains.

The greedy path in ``regalloc.schedule`` is untouched; this module only
*reads* DAG/allocator/placement internals for features and drives their
public mutation + rollback interfaces.
"""

import random
from dataclasses import dataclass

from ir import Instr
from scheduler import (DNode, FuncUnitPool, Weights, _classify, _make_picker,
                       _Placement, _KIND_DEBUG, _KIND_GATHER)
from regalloc import RegisterAllocator, _base, _resolve_operands


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for the per-trial state score (dot product with the feature
    dict from ``_features``). Deliberately minimal to start: throughput
    (alu_work), pressure (reg_delta), flexibility (frontier). Raw counts,
    no normalization; meant to be trained/swept once the structure works.

    Default: reads=+2, reg_delta=-1, K=6 - the sweep_rollout winner (1413
    cyc at 1536 scratch, correct on all rounds). reads (reward consuming
    read obligations) is the metric that meters in-flight chains and avoids
    the free_vec=0 deadlock; reg_delta is the pressure tiebreaker.
    reg_delta=0 or reads=1/3/4 are all worse; K=4 deadlocks, K=10 is not
    better (more trials can pick flashier-but-doomed orders).

    Obligation metrics (0 by default): reads counts obligations CONSUMED
    (each placed read decrements a tag's remaining count); obligations
    counts obligations CREATED (sum of read_count over freshly written
    tags). Their difference is the delta of total outstanding read
    obligations - a leading indicator of future frees, vs reg_delta's
    lagging one. Kept as separate features: a negative weight on
    obligations pushes high-fanout writes back, which may or may not be
    desirable under pressure."""
    alu_work: float = 0.0
    reg_delta: float = -1.0
    frontier: float = 0.0
    reads: float = 2.0
    obligations: float = 0.0


def _features(dag, allocator, pool, stats, ready_n):
    """Post-advance state features for one trial (raw counts, no caps):

      alu_work    - (v)alu slots used this cycle, lane-weighted:
                    scalar alu slots + 8 x valu slots (a valu slot does 8
                    lanes of work). Load/store/flow slots are not scored.
      reg_delta   - register allocations minus frees this cycle (pressure;
                    lower is better, weight should be negative).
      frontier    - next-cycle ready set size (post-advance; raw count).
      reads       - read operands consumed this cycle (obligations -1 each).
      obligations - sum of read_count over freshly allocated write tags
                    (obligations created this cycle).
    """
    cap = FuncUnitPool._CAPACITY
    return {
        "alu_work": (cap["alu"] - pool.free["alu"])
                    + 8 * (cap["valu"] - pool.free["valu"]),
        "reg_delta": stats["allocs"] - stats["frees"],
        "frontier": dag.frontier_size(),
        "reads": stats["reads"],
        "obligations": stats["obligations"],
    }


def _score(feats, w: ScoreWeights) -> float:
    return (w.alu_work * feats["alu_work"]
            + w.reg_delta * feats["reg_delta"]
            + w.frontier * feats["frontier"]
            + w.reads * feats["reads"]
            + w.obligations * feats["obligations"])


def _run_cycle(order, dag, allocator, placements, pool, emit):
    """Run one cycle's placements in ``order``; mirrors the placement
    semantics of ``regalloc.schedule`` exactly.

    emit=False (trial): decisions only - pool budgets, allocator and
        placement state, DAG commits. No operand resolution, no slot
        materialisation (pool.place dry mode).
    emit=True (winner replay): identical decisions plus resolved operands
        and emitted slots into ``pool.bundle``.

    Returns a stats dict: committed (nodes), allocs/frees (register
    allocate/free counts from the allocator's monotonic counters), reads
    (read operands consumed), obligations (sum of read_count over freshly
    allocated write tags).
    """
    n_committed = 0
    n_reads = 0
    n_obligations = 0
    a0, f0 = allocator.n_alloc, allocator.n_free

    def _commit_reads(instr):
        nonlocal n_reads
        for op in instr.read_operands():
            b = _base(op)
            if b in allocator.assigned:
                allocator.read(b)
                n_reads += 1

    def _alloc_writes(instr):
        nonlocal n_obligations
        wr_tags = [_base(o) for o in instr.write_operands()]
        if not all(allocator.can_write(t) for t in wr_tags):
            allocator.exhaustion_warnings += 1
            return None
        fresh = []
        for t in wr_tags:
            if t not in allocator.assigned:     # fresh allocation (not sticky)
                n_obligations += allocator._read_count[t]
                fresh.append(t)
            allocator.write(t)
        return wr_tags, fresh

    for idx in order:
        node = dag[idx]
        p = placements[idx]
        instr = node.instr
        if p.kind == _KIND_DEBUG:
            # Debug nodes write nothing but DO read a tag (their loc):
            # consume the read (free at 0) like any reader.
            if emit:
                resolved_node = DNode(idx=idx, engine=node.engine,
                                      instr=_resolve_operands(instr, allocator))
                pool.place(resolved_node, p)
            _commit_reads(instr)
            dag.commit(idx)
            n_committed += 1
            continue
        if p.kind == _KIND_GATHER:
            # Gather = one node emitting VLEN scalar loads (partial
            # completion). Allocate nv (sticky), land as many lanes as free
            # load slots allow; commit reads on completion.
            aw = _alloc_writes(instr)
            if aw is None:
                continue
            wr_tags, _ = aw
            dest = allocator.addr_of(wr_tags[0])     # nv vector granule
            take = min(p.lanes_total - p.lanes_done, pool.free["load"])
            if take == 0:
                continue                           # no load slot this cycle
            if emit:
                addr = allocator.addr_of(instr.addr)
                for j in range(p.lanes_done, p.lanes_done + take):
                    pool.bundle.setdefault("load", []).append(
                        ("load", dest + j, addr + j))
            pool.free["load"] -= take
            p.lanes_done += take
            if p.lanes_done == p.lanes_total:      # all 8 lanes landed
                _commit_reads(instr)
                dag.commit(idx)
                n_committed += 1
            continue
        # Regular write node: placeable only with a unit slot AND registers.
        aw = _alloc_writes(instr)
        if aw is None:
            continue   # leave for next cycle (a read may free a register)
        wr_tags, fresh = aw
        if emit:
            resolved_node = DNode(idx=idx, engine=node.engine,
                                  instr=_resolve_operands(instr, allocator))
            placed = pool.place(resolved_node, p)
        else:
            placed = pool.place(node, p, dry=True)
        if placed is None:
            # No lane landed this cycle: roll back the speculative allocs -
            # but ONLY if nothing was ever landed (lanes_done==0). A node
            # with lanes already written into dest keeps its register.
            if p.lanes_done == 0:
                for t in wr_tags:
                    allocator.unwrite(t)
                for t in fresh:      # no lasting obligation was created
                    n_obligations -= allocator._read_count[t]
            continue
        if placed:   # fully placed -> commit reads, free at 0
            _commit_reads(instr)
            dag.commit(idx)
            n_committed += 1
    return {"committed": n_committed, "allocs": allocator.n_alloc - a0,
            "frees": allocator.n_free - f0, "reads": n_reads,
            "obligations": n_obligations}


def schedule_rollout(dag, read_count, *, seed: int = 42, trials: int = 6,
                     greedy_trials: int = 1,
                     score_weights: ScoreWeights | None = None,
                     weights: Weights | None = None,
                     cap: int | None = None,
                     allocator: RegisterAllocator | None = None):
    """List-schedule a RAW-only DAG with per-cycle rollout search.

    Same contract as ``regalloc.schedule`` (returns the body's instruction
    bundles; continues an existing ``allocator`` from the linear prologue),
    but each cycle picks its placement order by scoring K candidate
    orderings on the post-cycle state they produce.

    ``trials``: candidate orderings evaluated per cycle (trial 0..greedy_trials-1
        are the incumbent weighted-greedy order, the rest uniform shuffles).
    ``score_weights``: linear weights over the state features.
    ``weights``: base picker weights for the greedy candidate ordering.
    """
    rng = random.Random(seed)
    if allocator is None:
        allocator = RegisterAllocator(read_count)
    if score_weights is None:
        score_weights = ScoreWeights()
    placements = [_classify(n) for n in dag.nodes]
    if cap is None:
        cap = len(dag) + 1

    # Greedy candidate ordering: the incumbent weighted picker + always-on
    # freeing bias (same priority as regalloc.schedule's pressured_key).
    key_fn = _make_picker("weighted", placements, rng, dag.props, weights)

    def freeing_read(idx) -> int:
        n = 0
        for op in dag[idx].instr.read_operands():
            b = _base(op)
            if allocator.remaining.get(b) == 1:
                n += 1
        return n

    def pressured_key(idx):
        base = key_fn(idx)
        b0 = base[0] if isinstance(base, tuple) else base
        return b0 - freeing_read(idx) * 1000

    pool = FuncUnitPool()
    bundles: list[dict] = []
    C = 0
    committed = 0
    total = len(dag)

    while committed < total:
        ready = sorted(dag.ready())
        if not ready:
            raise RuntimeError(
                f"rollout schedule: frontier empty with {total - committed} "
                f"uncommitted nodes at C={C} - cyclic DAG or counter bug")

        # Candidate orderings: greedy incumbent first, then shuffles.
        orders = []
        for _ in range(min(greedy_trials, trials)):
            orders.append(sorted(ready, key=pressured_key))
        for _ in range(trials - len(orders)):
            o = ready[:]
            rng.shuffle(o)
            orders.append(o)

        best_score = float("-inf")
        best_order = None
        for o in orders:
            td = dag.checkpoint()
            ta = allocator.checkpoint()
            tp = _Placement.checkpoint()
            pool.reset()
            stats = _run_cycle(o, dag, allocator, placements, pool,
                               emit=False)
            dag.advance()
            slots_used = sum(FuncUnitPool._CAPACITY[e] - pool.free[e]
                             for e in FuncUnitPool._CAPACITY)
            if slots_used == 0 and stats["committed"] == 0:
                # Nothing placed with work remaining: the same empty cycle
                # would repeat forever - a hard deadlock, not a stall.
                score = float("-inf")
            else:
                score = _score(_features(dag, allocator, pool, stats,
                                         len(ready)),
                               score_weights)
            _Placement.rollback(tp)
            allocator.rollback(ta)
            dag.rollback(td)
            if score > best_score:
                best_score = score
                best_order = o

        if best_order is None:
            raise RuntimeError(
                f"rollout schedule: deadlock at C={C} - all {len(orders)} "
                f"trials stuck ({len(ready)} ready nodes, none placeable; "
                f"free_vec={len(allocator.free_vec)} "
                f"free_scalar={len(allocator.free_scalar)})")

        # Winner replay: same order, emission on, no logging (checkpoints
        # all rolled back to token 0, so the logs are closed).
        pool.reset()
        stats = _run_cycle(best_order, dag, allocator, placements, pool,
                           emit=True)
        dag.advance()
        committed += stats["committed"]
        bundle = {eng: [s.tagged_lower() if isinstance(s, Instr) else s
                        for s in slots]
                  for eng, slots in pool.bundle.items()}
        bundles.append(bundle)
        C += 1
        if C > cap:
            raise RuntimeError(
                f"rollout schedule: cycle count {C} exceeded cap {cap}")

    if allocator.exhaustion_warnings:
        print(f"rollout: {allocator.exhaustion_warnings} register-exhaustion "
              f"stalls during scheduling")
    return bundles
