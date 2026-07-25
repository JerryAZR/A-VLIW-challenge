"""Sweep regalloc priority functions to find cycle potential of the RAW-only
DAG. Reuses the built DAG + read_count; re-schedules with different priority
functions and reports body-cycle counts (lower = better).

    python sweep_regalloc.py [budget_seconds]
"""
import random
import sys
import time

from problem import Tree, Input, build_mem_image
import perf_takehome as pt
import regalloc
from regalloc import (tag_raw_chains, build_dag, RegisterAllocator,
                      schedule as reg_schedule, recompute_read_count, _base)
from scheduler import prune_to_stores

PROLOGUE_EPI = 197  # prologue + pause bundles (fixed, not DAG-scheduled)

random.seed(123)
forest = Tree.generate(10)
inp = Input.generate(forest, 256, 16)
mem = build_mem_image(forest, inp)

# Build the tagged body once (capture via the kernel builder path).
cap = {}
_orig_tag = regalloc.tag_raw_chains
def _cap_tag(instrs, pinned=()):
    out = _orig_tag(instrs, pinned)
    cap["tagged"], cap["rc"] = out
    return out
regalloc.tag_raw_chains = _cap_tag
kb = pt.KernelBuilder()
kb.build_kernel(10, 2047, 256, 16, prune=True)
regalloc.tag_raw_chains = _orig_tag

tagged, rc = cap["tagged"], cap["rc"]
n_pro = len([1])  # placeholder; recompute below
# The kernel tags prologue+body; split like _emit_regalloc does.
# We can't easily get n_pro here, so rebuild via the same call the builder made.
# Simpler: re-derive by rebuilding the DAG the same way and pruning.
# Instead, capture tagged_body directly from a fresh build_dag call path.
# (We re-run tag over the full stream and split by the known prologue length.)
# To keep it robust, capture tagged_body from build_dag's input:
cap2 = {}
_orig_bd = regalloc.build_dag
def _cap_bd(t, pinned=()):
    cap2["body"] = t
    return _orig_bd(t, pinned)
regalloc.build_dag = _cap_bd
kb2 = pt.KernelBuilder()
kb2.build_kernel(10, 2047, 256, 16, prune=True)
regalloc.build_dag = _orig_bd

tagged_body = cap2["body"]
dag = prune_to_stores(build_dag(tagged_body, pinned={"const_vec_0"}))
read_count = recompute_read_count([n.instr for n in dag.nodes],
                                  pinned={"const_vec_0"})
from scheduler import _classify
for _n in dag.nodes:
    _classify(_n)


def body_cycles(prio):
    dag.reset()
    a = RegisterAllocator(read_count)
    bundles = reg_schedule(dag, read_count, seed=42, picker="weighted",
                           weights=pt.REGALLOC_WEIGHTS, allocator=a, prio=prio)
    n = sum(1 for b in bundles if any(e != "debug" for e in b))
    return n, a.exhaustion_warnings


def total(prio):
    n, _ = body_cycles(prio)
    return n + PROLOGUE_EPI


# --- priority factories: (allocator, base_key, freeing_read) -> key_fn ------

def mk_always_free(weight):
    def prio(allocator, base_key, freeing_read):
        def key(idx):
            b = base_key(idx)
            b0 = b[0] if isinstance(b, tuple) else b
            return b0 - freeing_read(idx) * weight
        return key
    return prio


def mk_threshold(thresh, weight):
    def prio(allocator, base_key, freeing_read):
        def key(idx):
            b = base_key(idx)
            if len(allocator.free_vec) < thresh:
                b0 = b[0] if isinstance(b, tuple) else b
                return b0 - freeing_read(idx) * weight
            return b
        return key
    return prio


def mk_no_bias():
    def prio(allocator, base_key, freeing_read):
        return base_key
    return prio


def mk_weighted_base(weights, free_w):
    """Always-free bias (weight free_w) on top of an arbitrary Weights base."""
    from scheduler import _make_picker
    import random as _r
    def prio(allocator, base_key, freeing_read):
        wf = _make_picker("weighted", _r.Random(42), weights)
        def key(idx):
            b = wf(dag.nodes[idx])
            b0 = b[0] if isinstance(b, tuple) else b
            return b0 - freeing_read(idx) * free_w
        return key
    return prio


def mk_net_delta(free_w, alloc_w):
    """Net register delta: freeing good (negative), allocating bad (positive)."""
    def prio(allocator, base_key, freeing_read):
        def key(idx):
            b = base_key(idx)
            b0 = b[0] if isinstance(b, tuple) else b
            n_alloc = len(dag[idx].instr.write_operands())
            n_free = freeing_read(idx)
            return b0 + alloc_w * n_alloc - free_w * n_free
        return key
    return prio


def mk_free_then_base(weight):
    """Freeing count as the PRIMARY sort, base as tiebreak."""
    def prio(allocator, base_key, freeing_read):
        def key(idx):
            b = base_key(idx)
            b0 = b[0] if isinstance(b, tuple) else b
            return (-freeing_read(idx) * weight, b0)
        return key
    return prio


variants = [
    ("no bias (pure weighted)", mk_no_bias()),
    ("current (thresh32, w1000)", mk_threshold(32, 1000)),
    ("always free w1", mk_always_free(1)),
    ("always free w1000", mk_always_free(1000)),
    ("net_delta f1000 a1000", mk_net_delta(1000, 1000)),
    ("net_delta f1000 a100", mk_net_delta(1000, 100)),
]

budget = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
print(f"=== regalloc priority sweep (body cycles, lower=better) ===")
results = []
for name, prio in variants:
    t0 = time.time()
    try:
        n, stalls = body_cycles(prio)
        dt = time.time() - t0
        results.append((n, name, stalls, dt))
        print(f"  {name:28s}: body {n:5d}  stalls {stalls:5d}  ({dt:.1f}s)")
    except RuntimeError as e:
        dt = time.time() - t0
        print(f"  {name:28s}: STALL  ({str(e)[:60]})  ({dt:.1f}s)")

results.sort()
print("\n=== best ===")
for n, name, stalls, dt in results[:3]:
    print(f"  {name}: body {n}, total {n + PROLOGUE_EPI}, stalls {stalls}")

# ---- random search over Weights base + always-free ----
from scheduler import Weights
VALUES = [-4, -2, -1, 0, 1, 2, 4]
rng = random.Random(12345)
best = None
t0 = time.time()
n = 0
print(f"\n=== random Weights search (always-free w1000), budget {budget:.0f}s ===")
while time.time() - t0 < budget:
    w = Weights(*(rng.choice(VALUES) for _ in range(6)))
    try:
        cycles, stalls = body_cycles(mk_weighted_base(w, 1000))
    except RuntimeError:
        continue
    n += 1
    if best is None or cycles < best[0]:
        best = (cycles, w, stalls)
        print(f"  [{time.time()-t0:5.1f}s] #{n}: NEW BEST {cycles} cyc  "
              f"sink={w.sink} load={w.load} raw={w.raw} war={w.war} "
              f"rigid={w.rigid} idx={w.idx}  (stalls {stalls})")
print(f"\n{n} samples; best body={best[0]} total={best[0]+PROLOGUE_EPI}")
print(f"  weights: {best[1]}")

# ---- local refinement: perturb the best weights by +/-1 each dim ----
print("\n=== local refinement around best ===")
best_w = list(best[1])
improved = True
while improved:
    improved = False
    for dim in range(6):
        for delta in (-1, 1):
            cand = list(best_w)
            cand[dim] += delta
            w = Weights(*cand)
            try:
                cycles, stalls = body_cycles(mk_weighted_base(w, 1000))
            except RuntimeError:
                continue
            if cycles < best[0]:
                best = (cycles, w, stalls)
                best_w = cand
                improved = True
                print(f"  dim{dim}{delta:+d}: NEW BEST {cycles}  {w}  (stalls {stalls})")
print(f"\nrefined best body={best[0]} total={best[0]+PROLOGUE_EPI}")
print(f"  weights: {best[1]}")
