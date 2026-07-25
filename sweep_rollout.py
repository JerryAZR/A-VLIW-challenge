"""Sweep rollout ScoreWeights / trial counts against the full kernel DAG.

Builds the kernel once per config (build_kernel only, no simulator) and
reports body cycle count or the deadlock cycle. Usage:

    python sweep_rollout.py
"""

import perf_takehome as pt
from rollout import ScoreWeights


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


def try_config(name, trials, greedy, sw):
    pt.SCHEDULER_MODE = "rollout"
    pt.ROLLOUT_TRIALS = trials
    pt.ROLLOUT_GREEDY_TRIALS = greedy
    pt.ROLLOUT_SCORE_WEIGHTS = sw
    cyc, err = build_cycles()
    status = f"{cyc} cyc" if cyc is not None else f"FAIL: {err}"
    print(f"{name:44s} {status}", flush=True)
    return cyc


if __name__ == "__main__":
    import time
    for rd in (-1, -2, -4, -8, -16):
        t0 = time.time()
        try_config(f"reg_delta={rd} frontier=0.1 K=6", 6, 1,
                   ScoreWeights(alu_work=1.0, reg_delta=rd, frontier=0.1))
        print(f"    ({time.time() - t0:.1f}s)", flush=True)
