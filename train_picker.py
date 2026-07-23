"""Train the weighted picker's weights via coordinate descent with random
restarts. The loss is body_cycles(schedule(dag, w)) - deterministic and
data-independent (the DAG is structural; all submission seeds give the same
count for fixed weights), so we optimize one deterministic function.

Landscape: piecewise-constant (the schedule changes only when a pairwise score
ordering flips -> gradients are useless, use sampling) and scale-invariant
(only score ORDERING matters -> optimum is a direction, ~4D not 5D). So we
search a weight vector and let magnitude fall out.

Builds the DAG once, reuses dag.reset() per trial. Prints each new best, so a
timeout still yields a result. Dumps the winner as a Weights(...) literal.

Run: python train_picker.py [budget_seconds]
"""
import random
import sys
import time

import numpy as np
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
# Fixed prologue+epilogue offset (vstores overlap the body; only pause 2 + the
# grown prologue remain). Measured once; only used for display, not for ranking.
PROLOGUE_EPI = 124


def body_cycles(w):
    dag.reset()
    bundles = schedule(dag, seed=42, picker="weighted", weights=w)
    return sum(1 for b in bundles if any(e != "debug" for e in b))


def total(w):
    return body_cycles(w) + PROLOGUE_EPI


# --- coordinate descent with random restarts ---
GRID = [-8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8]
FINE = [-1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5]  # refine around best
W = ["sink", "load", "raw", "war", "rigid"]
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
rng = random.Random(2024)
t0 = time.time()

best_w = None
best_c = float("inf")
seen = 0


def eval_w(w):
    global best_w, best_c, seen
    c = body_cycles(w)
    seen += 1
    if c < best_c:
        best_c = c
        best_w = w
        print(f"  [{time.time()-t0:6.1f}s] #{seen:4d}: NEW BEST body={c} total={c+PROLOGUE_EPI}  "
              f" sink={w.sink} load={w.load} raw={w.raw} war={w.war} rigid={w.rigid}", flush=True)
    return c


def random_w():
    return Weights(*(rng.choice(GRID) for _ in range(5)))


# start from the current shipped weights + a few random restarts
starts = [Weights(sink=-1, load=1, raw=-2, war=-2, rigid=-1)]  # current shipped
starts += [random_w() for _ in range(4)]
print(f"=== coordinate descent, budget {budget:.0f}s, {len(starts)} starts ===")

for si, start in enumerate(starts):
    if time.time() - t0 > budget:
        break
    cur = start
    cur_c = eval_w(cur)
    print(f"  -- start {si}: body={cur_c} {cur}", flush=True)
    while time.time() - t0 < budget:
        improved = False
        for ax in range(5):  # cycle through weights
            if time.time() - t0 > budget:
                break
            # coarse sweep this axis, keep best
            trial = list(cur)
            best_ax_val = cur[ax]
            best_ax_c = cur_c
            for v in GRID:
                trial[ax] = v
                c = eval_w(Weights(*trial))
                if c < best_ax_c:
                    best_ax_c = c
                    best_ax_val = v
            trial[ax] = best_ax_val
            if best_ax_c < cur_c:
                cur = Weights(*trial)
                cur_c = best_ax_c
                improved = True
        if not improved:
            break  # converged for this start

# refine around the global best with a fine grid
print(f"  -- refine around best {best_w}", flush=True)
cur = best_w
cur_c = best_c
while time.time() - t0 < budget:
    improved = False
    for ax in range(5):
        if time.time() - t0 > budget:
            break
        trial = list(cur)
        best_ax_val = cur[ax]
        best_ax_c = cur_c
        for v in FINE:
            trial[ax] = cur[ax] + v
            c = eval_w(Weights(*trial))
            if c < best_ax_c:
                best_ax_c = c
                best_ax_val = trial[ax]
        if best_ax_c < cur_c:
            cur = Weights(best_ax_val if ax == 0 else cur[0],
                          best_ax_val if ax == 1 else cur[1],
                          best_ax_val if ax == 2 else cur[2],
                          best_ax_val if ax == 3 else cur[3],
                          best_ax_val if ax == 4 else cur[4])
            cur_c = best_ax_c
            improved = True
    if not improved:
        break

print(f"\n=== {seen} evals in {time.time()-t0:.1f}s ===")
print(f"BEST: body={best_c} total={best_c+PROLOGUE_EPI}")
print(f"  Weights(sink={best_w.sink}, load={best_w.load}, raw={best_w.raw}, "
      f"war={best_w.war}, rigid={best_w.rigid})")
