"""Differential-evolution search over regalloc Weights (always-free bias on).
Escapes the random-search plateau by evolving a population with mutation +
crossover + selection.

    python de_regalloc.py [budget_seconds]
"""
import random
import sys
import time

from problem import Tree, Input, build_mem_image
import perf_takehome as pt
import regalloc
from regalloc import (build_dag, RegisterAllocator, schedule as reg_schedule,
                      recompute_read_count, _base)
from scheduler import prune_to_stores, Weights, _make_picker, _classify

PROLOGUE_EPI = 197

random.seed(123)
forest = Tree.generate(10)
inp = Input.generate(forest, 256, 16)
mem = build_mem_image(forest, inp)

cap = {}
_orig_bd = regalloc.build_dag
def _cap_bd(t, pinned=()):
    cap["body"] = t
    return _orig_bd(t, pinned)
regalloc.build_dag = _cap_bd
pt.KernelBuilder().build_kernel(10, 2047, 256, 16, prune=True)
regalloc.build_dag = _orig_bd

dag = prune_to_stores(build_dag(cap["body"], pinned={"const_vec_0"}))
read_count = recompute_read_count([n.instr for n in dag.nodes],
                                  pinned={"const_vec_0"})
_placements = [_classify(n) for n in dag.nodes]


def mk_weighted_base(weights, free_w):
    def prio(allocator, base_key, freeing_read):
        wf = _make_picker("weighted", _placements, random.Random(42),
                          dag.props, weights)
        def key(idx):
            b = wf(idx)
            b0 = b[0] if isinstance(b, tuple) else b
            return b0 - freeing_read(idx) * free_w
        return key
    return prio


def body_cycles(weights, free_w=1000):
    dag.reset()
    a = RegisterAllocator(read_count)
    try:
        bundles = reg_schedule(dag, read_count, seed=42, picker="weighted",
                               weights=weights, allocator=a,
                               prio=mk_weighted_base(weights, free_w))
    except RuntimeError:
        return None, None
    n = sum(1 for b in bundles if any(e != "debug" for e in b))
    return n, a.exhaustion_warnings


def vec(w):
    return [w.sink, w.load, w.raw, w.war, w.rigid, w.idx]


def to_weights(v):
    return Weights(*[int(round(x)) for x in v])


budget = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
rng = random.Random(2024)

# DE parameters
POP = 12
F = 0.7     # mutation factor
CR = 0.8    # crossover prob
BOUNDS = (-6, 6)

# Init population around the known best + random.
seed_best = vec(Weights(sink=-1, load=5, raw=1, war=1, rigid=1, idx=-4))
pop = [seed_best] + [[rng.uniform(*BOUNDS) for _ in range(6)] for _ in range(POP - 1)]
scores = []
for v in pop:
    c, _ = body_cycles(to_weights(v))
    scores.append(c if c is not None else 9999)

best_i = min(range(POP), key=lambda i: scores[i])
print(f"init best: {scores[best_i]} at {to_weights(pop[best_i])}")

t0 = time.time()
gen = 0
while time.time() - t0 < budget:
    gen += 1
    for i in range(POP):
        a, b, c = rng.sample([j for j in range(POP) if j != i], 3)
        mutant = [pop[a][d] + F * (pop[b][d] - pop[c][d]) for d in range(6)]
        trial = [mutant[d] if rng.random() < CR else pop[i][d] for d in range(6)]
        trial = [max(BOUNDS[0], min(BOUNDS[1], x)) for x in trial]
        cw = to_weights(trial)
        sc, _ = body_cycles(cw)
        sc = sc if sc is not None else 9999
        if sc <= scores[i]:
            pop[i] = trial
            scores[i] = sc
            if sc < min(scores):
                print(f"  [{time.time()-t0:5.1f}s] gen{gen}: NEW BEST {sc}  {cw}")
    best_i = min(range(POP), key=lambda i: scores[i])

print(f"\n{gen} generations; best body={scores[best_i]} total={scores[best_i]+PROLOGUE_EPI}")
print(f"  weights: {to_weights(pop[best_i])}")
