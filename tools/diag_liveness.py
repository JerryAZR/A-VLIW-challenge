"""Measure the vector-register liveness high-water mark of the level-3 kernel.

Runs the rollout schedule at big scratch (4096) and records, at each
committed cycle boundary (post-advance), the number of live VECTOR tags
(tags with a home, counting only is_vec). The peak answers: is a 1536-word
(185-granule) schedule of this DAG feasible at all?

  peak <= 185: the wall is the transition tax / scheduling, not capacity.
  peak >  185: no scheduler fits this DAG at 1536; dataflow must change.
"""

import random

import problem
import regalloc

BIG = 4096
problem.SCRATCH_SIZE = BIG
regalloc.SCRATCH_SIZE = BIG

from problem import Tree, Input
import perf_takehome as pt
from regalloc import RegisterAllocator
from rollout import schedule_rollout, ScoreWeights

random.seed(123)
forest = Tree.generate(10)
inp = Input.generate(forest, 256, 16)

import rollout
_orig_run = rollout._run_cycle

peaks = {"live_vec": 0, "hist": []}


def spy(order, dag, allocator, placements, pool, emit):
    r = _orig_run(order, dag, allocator, placements, pool, emit)
    if emit:   # count committed state only (winner replay), not trials
        live = sum(1 for t in allocator.assigned if t.is_vec)
        peaks["live_vec"] = max(peaks["live_vec"], live)
        peaks["hist"].append(live)
    return r


rollout._run_cycle = spy

kb = pt.KernelBuilder()
kb.build_kernel(10, 2047, 256, 16, prune=True)

hist = peaks["hist"]
print(f"peak live vector granules (committed state): {peaks['live_vec']}")
print(f"pool capacity at 1536 scratch: 185")
print(f"final-cycle live: {hist[-1]}, mean: {sum(hist)/len(hist):.0f}")
# Decile profile of the pressure over the body.
n = len(hist)
for d in range(0, 11):
    i = min(n - 1, d * n // 10)
    print(f"  {10*d:3d}% through body: {sorted(hist)[int(0.9*(n-1))]} (p90) "
          f"live at this point: {hist[i]}")
