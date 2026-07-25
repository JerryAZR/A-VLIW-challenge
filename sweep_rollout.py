"""Sweep rollout ScoreWeights / trial counts against the full kernel DAG.

Builds the kernel once per config (build_kernel only, no simulator) and
reports body cycle count or the deadlock cycle. Usage:

    python sweep_rollout.py
"""

import perf_takehome as pt
from rollout import (ScoreWeights, make_random_order, make_weighted_greedy,
                     make_interp_greedy)
from scheduler import Weights
import random


def build_cycles():
    """(body_cycles, error) for the current module config."""
    import random
    from problem import Tree, Input, build_mem_image
    random.seed(123)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    kb = pt.KernelBuilder()
    try:
        kb.build_kernel(forest.height, len(forest.values),
                        len(inp.indices), 16)
    except RuntimeError as e:
        return None, str(e).split("\n")[0][:110]
    # Cycle count = bundles containing any non-debug engine.
    return sum(1 for b in kb.instrs if any(eng != "debug" for eng in b)), None


def try_config(name, trials, sort_funcs, sw):
    pt.SCHEDULER_MODE = "rollout"
    pt.ROLLOUT_TRIALS = trials
    pt.ROLLOUT_SORT_FUNCS = sort_funcs
    pt.ROLLOUT_SCORE_WEIGHTS = sw
    cyc, err = build_cycles()
    status = f"{cyc} cyc" if cyc is not None else f"FAIL: {err}"
    print(f"{name:44s} {status}", flush=True)
    return cyc


if __name__ == "__main__":
    import time
    R = ScoreWeights(reads=2, reg_delta=-1)
    W = pt.REGALLOC_WEIGHTS
    RND = [make_random_order(random.Random(42)) for _ in range(5)]
    configs = [
        ("greedy group=-4", 6, [make_weighted_greedy(W._replace(group=-4))] + RND, R),
        ("greedy group=-6", 6, [make_weighted_greedy(W._replace(group=-6))] + RND, R),
        ("greedy group=-8", 6, [make_weighted_greedy(W._replace(group=-8))] + RND, R),
        ("greedy group=-12", 6, [make_weighted_greedy(W._replace(group=-12))] + RND, R),
        ("greedy group=-20", 6, [make_weighted_greedy(W._replace(group=-20))] + RND, R),
        ("group=-4 K=10", 10, [make_weighted_greedy(W._replace(group=-4))]
         + [make_random_order(random.Random(42)) for _ in range(9)], R),
        ("group=-4 K=3", 3, [make_weighted_greedy(W._replace(group=-4))]
         + [make_random_order(random.Random(42)) for _ in range(2)], R),
    ]
    for name, trials, sf, sw in configs:
        t0 = time.time()
        try_config(name, trials, sf, sw)
        print(f"    ({time.time() - t0:.1f}s)", flush=True)
