"""Sign-free joint random search over the trimmed Weights space (no war, no
idx). No sign assumptions: every dim sampled uniformly from [-LO, LO].

    python sweep_weights.py [n_samples] [bound]
"""

import random
import sys
import time

from problem import Tree, Input
import perf_takehome as pt
from scheduler import Weights

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
LO = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
DIMS = ("sink", "load", "raw", "rigid", "group", "freeing")


def build_cycles():
    random.seed(123)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    kb = pt.KernelBuilder()
    try:
        kb.build_kernel(forest.height, len(forest.values),
                        len(inp.indices), 16)
    except RuntimeError:
        return None
    return sum(1 for b in kb.instrs if any(e != "debug" for e in b))


rng = random.Random(2024)
results = []
best = None
t0 = time.time()
for i in range(N):
    w = Weights(**{d: round(rng.uniform(-LO, LO), 1) for d in DIMS})
    pt.REGALLOC_WEIGHTS = w
    cyc = build_cycles()
    if cyc is None:
        continue
    results.append((cyc, w))
    if best is None or cyc < best[0]:
        best = (cyc, w)
        print(f"[{time.time()-t0:6.1f}s] #{i}: NEW BEST {cyc} cyc  {w}",
              flush=True)

print(f"\n{len(results)}/{N} configs completed (rest deadlocked)")
results.sort(key=lambda r: r[0])
for cyc, w in results[:10]:
    print(f"  {cyc}  {w}")
