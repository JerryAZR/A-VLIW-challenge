"""Global search for picker weights via scipy differential_evolution.

Coordinate descent plateaued at 1514; DE is a global derivative-free optimizer
that maintains a population and explores broadly (good for the piecewise-
constant landscape where gradients are useless). Bounds [-8,8]^5 (normalized
props). Checkpoints each new best to _best_weights.txt.

Run: python train_de.py [budget_seconds]
"""
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution

import perf_takehome as pt
from scheduler import DAG, schedule, Weights

cap = {}
_orig = pt.KernelBuilder.build
pt.KernelBuilder.build = lambda self, slots, vliw=False, seed=None, picker="fma_first", weights=None: (
    cap.__setitem__("body", list(slots)), [])[1]
kb = pt.KernelBuilder(); kb.build_kernel(10, 2047, 256, 16)
pt.KernelBuilder.build = _orig
dag = DAG(cap["body"])
PROLOGUE_EPI = 124
CKPT = "_best_weights.txt"

budget = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
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
        best_c = sum(1 for b in schedule(dag, seed=42, picker="weighted", weights=best_w)
                     if any(e != "debug" for e in b))
        print(f"  loaded checkpoint: body={best_c} {best_w}", flush=True)
    except (FileNotFoundError, Exception):
        pass


def save_ckpt():
    with open(CKPT, "w") as f:
        f.write(f"Weights(sink={best_w.sink}, load={best_w.load}, "
                f"raw={best_w.raw}, war={best_w.war}, rigid={best_w.rigid})\n")


def loss(x):
    global best_w, best_c, seen
    w = Weights(*x)
    dag.reset()
    c = sum(1 for b in schedule(dag, seed=42, picker="weighted", weights=w)
            if any(e != "debug" for e in b))
    seen += 1
    if c < best_c:
        best_c = c
        best_w = w
        save_ckpt()
        print(f"  [{time.time()-t0:6.1f}s] #{seen}: NEW BEST body={c} "
              f"total={c+PROLOGUE_EPI}  {w}", flush=True)
    return c


load_ckpt()

# seed the population with known good points + perturbations (DE needs >4)
x0 = None
if best_w is not None:
    base = [best_w.sink, best_w.load, best_w.raw, best_w.war, best_w.rigid]
    x0 = [base]
    # add perturbed neighbors to seed the population around the known good point
    for d in [-1, 1, -0.5, 0.5]:
        for ax in range(5):
            v = list(base)
            v[ax] += d
            v = [max(-8, min(8, x)) for x in v]
            x0.append(v)

print(f"=== differential_evolution, budget {budget:.0f}s ===", flush=True)

class BudgetStop:
    def __call__(self, xk, convergence):
        return time.time() - t0 > budget

result = differential_evolution(
    loss, bounds=[(-8, 8)] * 5,
    maxiter=1000, popsize=18, tol=0.0, polish=False,
    init=np.array(x0) if x0 else "latinhypercube",
    callback=BudgetStop(),
    seed=42, disp=False,
)

print(f"\n=== {seen} evals in {time.time()-t0:.1f}s ===", flush=True)
print(f"BEST: body={best_c} total={best_c+PROLOGUE_EPI}", flush=True)
print(f"  Weights(sink={best_w.sink}, load={best_w.load}, raw={best_w.raw}, "
      f"war={best_w.war}, rigid={best_w.rigid})", flush=True)
