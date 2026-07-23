"""Random search over weighted-picker weights.

Builds the kernel body DAG once, then randomly samples weight vectors
(negatives included), re-scheduling each. Prints every new best as it's
found, so a timeout still yields a result. Total cycles = body + 185
(121 prologue + 64 epilogue, fixed). Correctness is guaranteed by the
scheduler (any valid schedule respects all RAW/WAR deps).

Run: python sweep_picker.py [budget_seconds]
"""
import random
import sys
import time

import perf_takehome as pt
from scheduler import DAG, schedule, Weights

# --- capture the body slot list (short-circuit build) ---
cap = {}
_orig = pt.KernelBuilder.build
pt.KernelBuilder.build = lambda self, slots, vliw=False, seed=None, picker="fma_first", weights=None: (
    cap.__setitem__("body", list(slots)), [])[1]
kb = pt.KernelBuilder(); kb.build_kernel(10, 2047, 256, 16)
pt.KernelBuilder.build = _orig
dag = DAG(cap["body"])

PROLOGUE_EPI = 185  # 121 prologue + 64 epilogue (fixed, not scheduled)


def body_cycles(weights):
    dag.reset()
    bundles = schedule(dag, seed=42, picker="weighted", weights=weights)
    return sum(1 for b in bundles if any(e != "debug" for e in b))


def total(weights):
    return body_cycles(weights) + PROLOGUE_EPI


# --- baselines ---
print("=== baselines (total = body + 185) ===")
for pk in ["idx", "fma_first", "random"]:
    dag.reset()
    b = schedule(dag, seed=42, picker=pk)
    c = sum(1 for x in b if any(e != "debug" for e in x))
    print(f"  {pk:10s}: body {c:5d}  total {c + PROLOGUE_EPI}")
# previous grid winner, for reference
ref = Weights(0, 0, 0, 1, 1)
print(f"  grid winner (0,0,0,1,1): body {body_cycles(ref):5d}  total {total(ref)}")

# --- random search ---
# Discrete value set with negatives; 7^5 = 16807 possible combos sampled at
# random so we explore many distinct regions without grinding any one axis.
VALUES = [-4, -2, -1, 0, 1, 2, 4]
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
rng = random.Random(12345)
print(f"\n=== random search: weights from {VALUES}, budget {budget:.0f}s ===")

best = None
results = []
t0 = time.time()
n = 0
last_heartbeat = t0
while time.time() - t0 < budget:
    w = Weights(*(rng.choice(VALUES) for _ in range(5)))
    tot = total(w)
    results.append((tot, w))
    n += 1
    if best is None or tot < best[0]:
        best = (tot, w)
        print(f"  [{time.time()-t0:6.1f}s] #{n:4d}: NEW BEST {tot} cyc  "
              f"sink={w.sink} load={w.load} raw={w.raw} war={w.war} rigid={w.rigid}")
    now = time.time()
    if now - last_heartbeat >= 30:
        last_heartbeat = now
        print(f"  [{now-t0:6.1f}s] #{n:4d} samples; best so far {best[0]} cyc  "
              f"Weights{best[1]}")

print(f"\n=== {n} samples in {time.time()-t0:.1f}s ===")
print(f"BEST: {best[0]} cyc  Weights{best[1]}")
results.sort()
print("\ntop 10:")
for tot, w in results[:10]:
    print(f"  {tot:5d} cyc  sink={w.sink} load={w.load} raw={w.raw} "
          f"war={w.war} rigid={w.rigid}")
