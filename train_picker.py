"""Train the weighted picker weights via coordinate descent + random restarts,
with checkpointing so runs can be chained across timeouts.

Loss = body_cycles(schedule(dag, w)): deterministic, data-independent (the DAG
is structural; all submission seeds give the same count for fixed weights).
Landscape is piecewise-constant (gradients useless) + scale-invariant (optimum
is a direction) -> sampling-based coordinate descent.

Each new best is printed AND appended to _best_weights.txt (one Weights literal
per line, last line = current best), so a killed run still yields its best, and
the next run loads it as a start point.

Run: python train_picker.py [budget_seconds]
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
PROLOGUE_EPI = 123  # measured: 123 prologue + 0 epilogue (vstores overlap body; pause2 is 0-cyc)


def body_cycles(w):
    dag.reset()
    return sum(1 for b in schedule(dag, seed=42, picker="weighted", weights=w)
               if any(e != "debug" for e in b))


# --- finer grid (smaller steps) + fine refine offsets ---
GRID = [-8, -6, -4, -3, -2, -1.5, -1, -0.75, -0.5, -0.25, 0,
        0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8]
FINE = [-0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5]
CKPT = "_best_weights.txt"

budget = float(sys.argv[1]) if len(sys.argv) > 1 else 500.0
rng = random.Random(int(time.time()))
t0 = time.time()

best_w = None
best_c = float("inf")
seen = 0


def load_ckpt():
    global best_w, best_c
    try:
        line = open(CKPT).read().strip().splitlines()[-1]
        best_w = eval(line, {"Weights": Weights})
        dag.reset()
        best_c = body_cycles(best_w)
        print(f"  loaded checkpoint: body={best_c} {best_w}", flush=True)
    except (FileNotFoundError, Exception):
        pass


def save_ckpt():
    with open(CKPT, "w") as f:
        f.write(f"Weights(sink={best_w.sink}, load={best_w.load}, "
                f"raw={best_w.raw}, war={best_w.war}, rigid={best_w.rigid}, "
                f"idx={best_w.idx})\n")


def eval_w(w):
    global best_w, best_c, seen
    c = body_cycles(w)
    seen += 1
    if c < best_c:
        best_c = c
        best_w = w
        save_ckpt()
        print(f"  [{time.time()-t0:6.1f}s] #{seen:4d}: NEW BEST body={c} "
              f"total={c+PROLOGUE_EPI}  sink={w.sink} load={w.load} "
              f"raw={w.raw} war={w.war} rigid={w.rigid}", flush=True)
    return c


def coord_descent(start):
    """Line-search each axis over GRID, cycle to convergence."""
    cur = start
    cur_c = eval_w(cur)
    while time.time() - t0 < budget:
        improved = False
        for ax in range(6):
            if time.time() - t0 > budget:
                break
            trial = list(cur)
            best_ax_val, best_ax_c = cur[ax], cur_c
            for v in GRID:
                trial[ax] = v
                c = eval_w(Weights(*trial))
                if c < best_ax_c:
                    best_ax_c, best_ax_val = c, v
            trial[ax] = best_ax_val
            if best_ax_c < cur_c:
                cur, cur_c = Weights(*trial), best_ax_c
                improved = True
        if not improved:
            break
    return cur, cur_c


def refine(start):
    """Fine offsets around each axis from the current value."""
    cur = start
    cur_c = best_c
    while time.time() - t0 < budget:
        improved = False
        for ax in range(6):
            if time.time() - t0 > budget:
                break
            base = list(cur)
            best_ax_val, best_ax_c = cur[ax], cur_c
            for off in FINE:
                base[ax] = cur[ax] + off
                c = eval_w(Weights(*base))
                if c < best_ax_c:
                    best_ax_c, best_ax_val = c, base[ax]
            base[ax] = best_ax_val
            if best_ax_c < cur_c:
                cur, cur_c = Weights(*base), best_ax_c
                improved = True
        if not improved:
            break


load_ckpt()

# starts: checkpoint best (if any), current shipped weights, + random restarts
starts = []
if best_w is not None:
    starts.append(best_w)
starts.append(Weights(sink=-3, load=1.5, raw=-1, war=6, rigid=-8, idx=0))  # shipped
print(f"=== coordinate descent, budget {budget:.0f}s ===", flush=True)

si = 0
while time.time() - t0 < budget:
    if si < len(starts):
        start = starts[si]
    else:
        start = Weights(*(rng.choice(GRID) for _ in range(6)))  # random restart
    si += 1
    print(f"  -- start {si}: {start}", flush=True)
    coord_descent(start)

# refine around the global best with the fine grid
if time.time() - t0 < budget:
    print(f"  -- refine around best {best_w}", flush=True)
    refine(best_w)

print(f"\n=== {seen} evals in {time.time()-t0:.1f}s ===", flush=True)
print(f"BEST: body={best_c} total={best_c+PROLOGUE_EPI}", flush=True)
print(f"  Weights(sink={best_w.sink}, load={best_w.load}, raw={best_w.raw}, "
      f"war={best_w.war}, rigid={best_w.rigid})", flush=True)
