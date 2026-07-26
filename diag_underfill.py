"""Diagnose valu-underfilled cycles: register pressure vs work starvation.

Replaces rollout.schedule_rollout with a per-cycle logging K=1 equivalent
(monkeypatched, so KernelBuilder.build_kernel runs unchanged otherwise)
and classifies every cycle where valu was not filled:

  pressure    - more valu-native nodes were ready than placed; since valu
                had a free slot, what blocked them was register exhaustion
                (every vec op is a writer; allocation precedes placement).
  starvation  - every valu-native ready node was placed; the frontier simply
                had no more valu work to give.

(Approximation: a vec_elem sticky-spilled to alu in an earlier cycle is no
longer valu-eligible but still counts as valu-native here; rare enough not
to flip classifications.)

Usage: python diag_underfill.py [l0off|l0on]
"""

import random
import sys

from problem import Tree, Input
import perf_takehome as pt
import rollout as ro
from rollout import SortCtx, _run_cycle, _classify, make_weighted_greedy
from scheduler import FuncUnitPool
from regalloc import RegisterAllocator

ROWS = []


def logging_schedule(dag, read_count, *, seed=42, trials=1, sort_funcs=None,
                     weights=None, score_weights=None, cap=None,
                     allocator=None):
    if allocator is None:
        allocator = RegisterAllocator(read_count)
    for n in dag.nodes:
        _classify(n)
    if sort_funcs is None:
        sort_funcs = [make_weighted_greedy(weights)]
    pool = FuncUnitPool()
    bundles = []
    committed = 0
    total = len(dag)
    C = 0
    while committed < total:
        ready = [dag.nodes[i] for i in sorted(dag.ready())]
        ready_valu = sum(1 for n in ready
                         if n.placement.native_engine == "valu"
                         and n.placement.lanes_done < n.placement.lanes_total)
        ctx = SortCtx(allocator=allocator, progress=committed / total)
        order = sort_funcs[0](ready, ctx)
        pool.reset()
        w0 = allocator.exhaustion_warnings
        stats = _run_cycle(order, dag, allocator, pool, emit=True)
        exhaust = allocator.exhaustion_warnings - w0
        dag.advance()
        committed += stats["committed"]
        allocator.collect_scalar_garbage()
        ROWS.append(dict(C=C, ready=len(ready), ready_valu=ready_valu,
                         valu=FuncUnitPool._CAPACITY["valu"] - pool.free["valu"],
                         alu=FuncUnitPool._CAPACITY["alu"] - pool.free["alu"],
                         exhaust=exhaust, free_vec=len(allocator.free_vec)))
        bundles.append({eng: [s.tagged_lower() if hasattr(s, "tagged_lower")
                              else s for s in slots]
                        for eng, slots in pool.bundle.items()})
        C += 1
    return bundles


def run(name):
    ROWS.clear()
    ro.schedule_rollout = logging_schedule
    random.seed(123)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    kb = pt.KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)
    n = len(ROWS)
    valu = sum(r["valu"] for r in ROWS)
    floor = -(-valu // 6)
    print(f"\n== {name}: {n} cycles, valu {valu} slots, floor {floor}, "
          f"slack {n - floor}")
    under = [r for r in ROWS if r["valu"] < 6]
    cls = {"pressure": [], "starvation": []}
    for r in under:
        key = "pressure" if r["ready_valu"] > r["valu"] else "starvation"
        cls[key].append(r)
    for key, rs in cls.items():
        phases = (sum(1 for r in rs if r["C"] < 60),
                  sum(1 for r in rs if 60 <= r["C"] < n - 40),
                  sum(1 for r in rs if r["C"] >= n - 40))
        lost = sum(6 - r["valu"] for r in rs)
        print(f"  {key:10s}: {len(rs):3d} cycles (lost {lost:3d} valu slots) "
              f" ramp/mid/tail = {phases[0]}/{phases[1]}/{phases[2]}")
    print("  pressure cycles:", [r["C"] for r in cls["pressure"]][:40])
    print("  first-40 valu fill:", [r["valu"] for r in ROWS[:40]])
    print("  first-40 ready_valu:", [r["ready_valu"] for r in ROWS[:40]])
    print("  first-40 free_vec:", [r["free_vec"] for r in ROWS[:40]])


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("l0off", "all"):
        pt.LEVEL0_DIRECT_TREE0 = False
        pt.ROLLOUT_SORT_FUNCS = None
        run("L0 OFF, shipped weights")
    if which in ("l0on", "all"):
        pt.LEVEL0_DIRECT_TREE0 = True
        from scheduler import Weights
        from rollout import make_interp_greedy
        w1 = Weights(sink=9.1, load=9.9, raw=-0.1, rigid=6.4, group=2.8,
                     freeing=2.4)
        w2 = Weights(sink=4.1, load=-1.6, raw=4.4, rigid=3.3, group=9.7,
                     freeing=4.3)
        pt.ROLLOUT_SORT_FUNCS = [make_interp_greedy(w1, w2)]
        run("L0 ON, interp 1123 weights")
