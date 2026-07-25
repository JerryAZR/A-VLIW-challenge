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
    R = ScoreWeights(reads=1, reg_delta=-1)
    configs = [
        ("reads=2 rd=-1 K=6", 6, 1, ScoreWeights(reads=2, reg_delta=-1)),
        ("reads=2 rd=-1 K=10", 10, 1, ScoreWeights(reads=2, reg_delta=-1)),
        ("reads=4 rd=-1 K=6", 6, 1, ScoreWeights(reads=4, reg_delta=-1)),
        ("reads=2 rd=0 K=6", 6, 1, ScoreWeights(reads=2, reg_delta=0)),
        ("reads=2 rd=-2 K=6", 6, 1, ScoreWeights(reads=2, reg_delta=-2)),
        ("reads=3 rd=-1 K=6", 6, 1, ScoreWeights(reads=3, reg_delta=-1)),
        ("reads=2 rd=-1 oblig=-0.5 K=6", 6, 1, ScoreWeights(reads=2, reg_delta=-1, obligations=-0.5)),
        ("reads=2 rd=-1 alu=0.1 K=6", 6, 1, ScoreWeights(reads=2, reg_delta=-1, alu_work=0.1)),
    ]
    for name, trials, greedy, sw in configs:
        t0 = time.time()
        try_config(name, trials, greedy, sw)
        print(f"    ({time.time() - t0:.1f}s)", flush=True)
