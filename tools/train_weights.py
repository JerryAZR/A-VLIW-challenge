"""Weight training for the weighted-greedy priority (trimmed space, no
war/idx). Two independent phases, each with its own TIME BUDGET and
external result file; phases chain through files so any phase can be
restarted without rerunning the others:

    python -m tools.train_weights random   600  --mode single
    python -m tools.train_weights finetune 1200 --mode single
    python -m tools.train_weights random   600  --mode interp
    python -m tools.train_weights finetune 1200 --mode interp

  random    - sign-free joint random search until the budget is exhausted;
              every completing config is checkpointed to --out (rewritten
              on every new best + periodically, so early stops keep data).
  finetune  - coordinate descent around the top/diverse candidates loaded
              from --in; time-shared breadth-first across candidates (one
              neighborhood pass each, round-robin) with the results file
              rewritten after every improvement.

Modes:
  single    - 6 dims, one priority fn (make_weighted_greedy).
  interp    - 12 dims (w_early ++ w_late) for make_interp_greedy; random
              seeding (NOT pairs of pre-optimized singles - those were
              optimized for no interpolation and may sit far from the
              interp optimum).

PARALLELISM: candidate evaluation (build_cycles) is farmed out to a
multiprocessing pool (--workers, default min(28, cpu_count)). random is
embarrassingly parallel. finetune evaluates the whole (dims x DELTAS)
neighborhood of a candidate in one parallel batch and applies the single
best improvement - steepest-neighbor descent, NOT the sequential
Gauss-Seidel apply-first-improvement of the serial version (the search
trajectory differs; the checkpoint/best bookkeeping is unchanged).

Search space (v2, data-backed from 4050+166+37 trained completes):

  raw      DROPPED (fixed 0.0): corr +/-0.04 with cycles in every file, no
           sign preference in the top-50 - pure search-space noise.
  bounds   per-prop, sign-biased where the top-50 is unambiguous:
           sink [0,10], freeing [0,10] (98-100% positive),
           load [-3,10], group [-4,10], rigid [-3,6].
  interp   reparameterized as base+tilt: w1 = base + tilt/2 (dominates
           late), w2 = base - tilt/2 (dominates early). The signal lives
           in base (corr -0.33/-0.44) while tilt is weak (+/-0.1), so this
           axis-aligns the strong directions for both random sampling and
           coordinate descent. tilt bounds +/-5.
  deltas   two-stage finetune: coarse (+/-2/1/0.5) to convergence, then
           fine (+/-0.5/0.25) polish until convergence or budget.

Loss: total scheduled cycles of build_kernel at K=1 (deterministic).
Deadlocks fail fast (~0.4s); completes ~1.6s.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

from problem import Tree, Input
import perf_takehome as pt
from rollout import make_interp_greedy
from scheduler import Weights

# Searched properties (raw dropped as inert); per-prop random-search bounds.
PROP_BOUNDS = {
    "sink":    ( 0.0, 10.0),
    "load":    (-3.0, 10.0),
    "rigid":   (-3.0,  6.0),
    "group":   (-4.0, 10.0),
    "freeing": ( 0.0, 16.0),
}
SEARCH_PROPS = tuple(PROP_BOUNDS)
TILT_BOUND = 5.0


def dims_for(mode):
    if mode == "single":
        return list(SEARCH_PROPS)
    return ([f"{p}_base" for p in SEARCH_PROPS]
            + [f"{p}_tilt" for p in SEARCH_PROPS])


def dim_bounds(dims):
    out = []
    for name in dims:
        if name.endswith("_base"):
            out.append(PROP_BOUNDS[name[:-5]])
        elif name.endswith("_tilt"):
            out.append((-TILT_BOUND, TILT_BOUND))
        else:
            out.append(PROP_BOUNDS[name])
    return out


def _mk(d):
    d = dict(d)
    d["raw"] = 0.0                    # inert, fixed
    return Weights(**d)


def weights_from(v, mode):
    """Search vector -> (w1, w2|None): single = one Weights; interp =
    base+tilt pair (w1 = base + tilt/2 dominates late, w2 = base - tilt/2
    dominates early)."""
    if mode == "single":
        return _mk(zip(SEARCH_PROPS, v)), None
    n = len(SEARCH_PROPS)
    b, t = v[:n], v[n:]
    w1 = {p: round(b[i] + t[i] / 2, 2) for i, p in enumerate(SEARCH_PROPS)}
    w2 = {p: round(b[i] - t[i] / 2, 2) for i, p in enumerate(SEARCH_PROPS)}
    return _mk(w1.items()), _mk(w2.items())


def build_cycles(v, mode, knob=0):
    pt.PRELOAD_L4_GROUPS = knob
    if mode == "single":
        w, _ = weights_from(v, mode)
        pt.ROLLOUT_SORT_FUNCS = None
        pt.REGALLOC_WEIGHTS = w
    else:
        w1, w2 = weights_from(v, mode)
        pt.ROLLOUT_SORT_FUNCS = [make_interp_greedy(w1, w2)]
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


# --- parallel evaluation -------------------------------------------------

def _worker_init():
    # per-eval noise (rollout stall reports) would interleave 28-way on the
    # console; the parent prints everything that matters. stderr stays.
    sys.stdout = open(os.devnull, "w")


def _eval_star(arg):
    """Pool worker: (v, mode, knob) -> (cycles|None, v). Never raises - a stray
    failure must not kill a long training run; it is reported on stderr and
    scored as a non-complete."""
    v, mode, knob = arg
    try:
        return (build_cycles(v, mode, knob), v)
    except Exception:
        import traceback
        traceback.print_exc()
        return (None, v)


def default_workers():
    # leave 4 logical processors for the OS/parent; never below 1
    return min(28, max(1, (os.cpu_count() or 1) - 4))


def dump(path, rows):
    tmp = path + ".tmp"
    json.dump(rows, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def phase_random(budget, out, seed, mode, workers, knob):
    dims = dims_for(mode)
    bounds = dim_bounds(dims)
    rng = random.Random(seed)
    found = []
    best = None
    t0 = time.time()
    i = 0
    last_dump = t0

    def candidates():
        while True:
            yield ([round(rng.uniform(lo, hi), 1) for lo, hi in bounds],
                   mode, knob)

    with Pool(workers, initializer=_worker_init) as pool:
        it = pool.imap_unordered(_eval_star, candidates(), chunksize=1)
        for cyc, v in it:
            i += 1
            if time.time() - t0 >= budget:
                break
            if cyc is None:
                continue
            found.append((cyc, v))
            found.sort(key=lambda r: r[0])
            if best is None or cyc < best[0]:
                best = (cyc, v)
                print(f"[{time.time()-t0:6.1f}s] #{i}: NEW BEST {cyc} cyc  "
                      f"{dict(zip(dims, v))}", flush=True)
                dump(out, found)
            elif time.time() - last_dump > 30:
                dump(out, found)
                last_dump = time.time()
    dump(out, found)
    print(f"random({mode}): {len(found)}/{i} completes -> {out} "
          f"({time.time()-t0:.0f}s)")


def pick_diverse(found, k):
    """Greedy max-min distance over L2 in weight space."""
    def dist(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    pts = [found[0]]
    while len(pts) < k and len(pts) < len(found):
        c = max(found, key=lambda r: min(dist(r[1], p[1]) for p in pts))
        if min(dist(c[1], p[1]) for p in pts) < 1.5:
            break                      # remaining completes are all near-dupes
        pts.append(c)
    return pts


COARSE_DELTAS = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
FINE_DELTAS = (-0.5, -0.25, 0.25, 0.5)


def _neighbors(v, dims, deltas):
    for d in range(len(dims)):
        for delta in deltas:
            cand = list(v)
            cand[d] = round(cand[d] + delta, 2)
            if cand != v:
                yield cand


KNOB_DELTAS = (-4, -2, -1, 1, 2, 4)


def _knob_neighbors(knob, n_groups=32):
    for kd in KNOB_DELTAS:
        nk = knob + kd
        if 0 <= nk <= n_groups:
            yield nk


def phase_finetune(budget, infile, out, top, mode, workers, knob):
    dims = dims_for(mode)
    found = json.load(open(infile))
    # rows may carry their own knob (3-elem, from a previous finetune)
    starts = pick_diverse([(r[0], r[1]) for r in found], top)
    knob_of = {(r[0], tuple(r[1])): (r[2] if len(r) > 2 else knob)
               for r in found}
    print(f"{len(starts)} candidates from {infile}:")
    for cyc, v in starts:
        print(f"  {cyc}  {dict(zip(dims, v))}")
    cur = [[cyc, list(v), knob_of.get((cyc, tuple(v)), knob)]
           for cyc, v in starts]
    t0 = time.time()
    dump(out, cur)
    deltas = COARSE_DELTAS
    # Breadth-first round-robin: one steepest-neighborhood pass per
    # candidate (whole neighborhood evaluated in one parallel batch, single
    # best improvement applied), so the budget is shared fairly and every
    # intermediate state is checkpointed. Coarse deltas to convergence,
    # then a fine-delta polish stage.
    with Pool(workers, initializer=_worker_init) as pool:
        while time.time() - t0 < budget:
            improved_any = False
            for ci in range(len(cur)):
                if time.time() - t0 >= budget:
                    break
                batch = [(cand, mode, cur[ci][2])
                         for cand in _neighbors(cur[ci][1], dims, deltas)]
                # the knob is a finetune coordinate too: (same weights,
                # G +/- delta) trials ride the same parallel batch
                batch += [(cur[ci][1], mode, nk)
                          for nk in _knob_neighbors(cur[ci][2])]
                results = pool.map(_eval_star, batch, chunksize=1)
                best = None
                for (cand_v, _, nk), (cyc, _) in zip(batch, results):
                    if cyc is not None and (best is None or cyc < best[0]):
                        best = (cyc, cand_v, nk)
                if best is not None and best[0] < cur[ci][0]:
                    cur[ci] = list(best)
                    improved_any = True
                    dump(out, sorted(cur, key=lambda r: r[0]))
                    print(f"[{time.time()-t0:6.1f}s] cand {ci}: "
                          f"{best[0]} cyc (knob {best[2]}) "
                          f"{dict(zip(dims, best[1]))}",
                          flush=True)
            if not improved_any:
                if deltas is COARSE_DELTAS:
                    deltas = FINE_DELTAS
                    print(f"[{time.time()-t0:6.1f}s] coarse converged; "
                          f"switching to fine deltas", flush=True)
                else:
                    break
    cur.sort(key=lambda r: r[0])
    dump(out, cur)
    print(f"finetune done -> {out} ({time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="phase", required=True)
    pr = sub.add_parser("random", help="sign-free joint random search")
    pr.add_argument("budget", type=float, help="seconds")
    pr.add_argument("--out", default=None)
    pr.add_argument("--seed", type=int, default=2024)
    pr.add_argument("--mode", choices=["single", "interp"], default="single")
    pr.add_argument("--knob", type=int, default=0,
                    help="PRELOAD_L4_GROUPS value during evaluation")
    pr.add_argument("--workers", type=int, default=default_workers())
    pf = sub.add_parser("finetune", help="coordinate descent around candidates")
    pf.add_argument("budget", type=float, help="seconds")
    pf.add_argument("--in", dest="infile", default=None)
    pf.add_argument("--out", default=None)
    pf.add_argument("--top", type=int, default=6)
    pf.add_argument("--mode", choices=["single", "interp"], default="single")
    pf.add_argument("--knob", type=int, default=0,
                    help="PRELOAD_L4_GROUPS value during evaluation")
    pf.add_argument("--workers", type=int, default=default_workers())
    args = ap.parse_args()

    if args.phase == "random":
        out = args.out or f"weights/_weights_found_{args.mode}.json"
        phase_random(args.budget, out, args.seed, args.mode,
                     args.workers, args.knob)
    else:
        infile = args.infile or f"weights/_weights_found_{args.mode}.json"
        out = args.out or f"weights/_weights_refined_{args.mode}.json"
        phase_finetune(args.budget, infile, out, args.top, args.mode,
                       args.workers, args.knob)


if __name__ == "__main__":
    main()
