"""Incremental measurement: which small consts benefit from being computed
on valu vs const+vbroadcast? Each config is one build (no simulator).

    python sweep_consts.py
"""

import random
import time

from problem import Tree, Input
import perf_takehome as pt

ALL = {1, 2, 3, 9, 16, 19, 33, 4097}


def build_cycles():
    random.seed(123)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    kb = pt.KernelBuilder()
    try:
        kb.build_kernel(forest.height, len(forest.values),
                        len(inp.indices), 16)
    except RuntimeError as e:
        return None, str(e).split("\n")[0][:100]
    return sum(1 for b in kb.instrs if any(e != "debug" for e in b)), None


VARIANTS = [
    ("all computed (current)", ALL),
    ("none computed (all bcast)", set()),
    ("only 1,2,3", {1, 2, 3}),
    ("no 4097 (bcast)", ALL - {4097}),
    ("no 19,33 (bcast)", ALL - {19, 33}),
    ("no 16 (+deps): 16 bcast, 19/33 comp", {1, 2, 3, 9, 19, 33}),
    ("16 bcast, 19/33 comp, no 4097", {1, 2, 3, 9, 19, 33}),
    ("4097 only via bcast; 16/19/33 comp", ALL - {4097}),
    ("only 1,2,3,9", {1, 2, 3, 9}),
]

for name, s in VARIANTS:
    pt.COMPUTED_CONSTS = s
    t0 = time.time()
    cyc, err = build_cycles()
    status = f"{cyc} cyc" if cyc is not None else f"FAIL: {err}"
    print(f"{name:42s} {status}   ({time.time() - t0:.1f}s)", flush=True)
