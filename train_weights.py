"""Weight training for the weighted-greedy priority (trimmed space, no
war/idx). Two independent phases, each with its own TIME BUDGET and
external result file; phases chain through files so any phase can be
restarted without rerunning the others:

    python train_weights.py random   600  --mode single
    python train_weights.py finetune 1200 --mode single
    python train_weights.py random   600  --mode interp
    python train_weights.py finetune 1200 --mode interp

  random    - sign-free joint random search until the budget is exhausted;
              every completing config is checkpointed to --out (rewritten
              on every new best + periodically, so early stops keep data).
  finetune  - coordinate descent around the top/diverse candidates loaded
              from --in; time-shared breadth-first across candidates (one
              coordinate pass each, round-robin) with the results file
              rewritten after every improvement.

Modes:
  single    - 6 dims, one priority fn (make_weighted_greedy).
  interp    - 12 dims (w_early ++ w_late) for make_interp_greedy; random
              seeding (NOT pairs of pre-optimized singles - those were
              optimized for no interpolation and may sit far from the
              interp optimum).

Loss: total scheduled cycles of build_kernel at K=1 (deterministic).
Deadlocks fail fast (~0.4s); completes ~1.6s.
"""

import argparse
import json
import math
import os
import random
import time

from problem import Tree, Input
import perf_takehome as pt
from rollout import make_interp_greedy
from scheduler import Weights

BASE_DIMS = ("sink", "load", "raw", "rigid", "group", "freeing")


def dims_for(mode):
    if mode == "single":
        return list(BASE_DIMS)
    return [f"{d}1" for d in BASE_DIMS] + [f"{d}2" for d in BASE_DIMS]


def build_cycles(v, mode):
    if mode == "single":
        pt.ROLLOUT_SORT_FUNCS = None
        pt.REGALLOC_WEIGHTS = Weights(**dict(zip(BASE_DIMS, v)))
    else:
        w1 = Weights(**dict(zip(BASE_DIMS, v[:6])))
        w2 = Weights(**dict(zip(BASE_DIMS, v[6:])))
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


def dump(path, rows):
    tmp = path + ".tmp"
    json.dump(rows, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def phase_random(budget, out, bound, seed, mode):
    dims = dims_for(mode)
    rng = random.Random(seed)
    found = []
    best = None
    t0 = time.time()
    i = 0
    last_dump = t0
    while time.time() - t0 < budget:
        v = [round(rng.uniform(-bound, bound), 1) for _ in dims]
        i += 1
        cyc = build_cycles(v, mode)
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


DELTAS = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


def phase_finetune(budget, infile, out, top, mode):
    dims = dims_for(mode)
    found = json.load(open(infile))
    starts = pick_diverse([tuple(r) for r in found], top)
    print(f"{len(starts)} candidates from {infile}:")
    for cyc, v in starts:
        print(f"  {cyc}  {dict(zip(dims, v))}")
    cur = [[cyc, list(v)] for cyc, v in starts]
    t0 = time.time()
    dump(out, cur)
    improved_any = True
    # Breadth-first round-robin: one coordinate pass per candidate, so the
    # budget is shared fairly and every intermediate state is checkpointed.
    while improved_any and time.time() - t0 < budget:
        improved_any = False
        for ci in range(len(cur)):
            for d in range(len(dims)):
                if time.time() - t0 >= budget:
                    break
                for delta in DELTAS:
                    cand = list(cur[ci][1])
                    cand[d] = round(cand[d] + delta, 1)
                    if cand == cur[ci][1]:
                        continue
                    cyc = build_cycles(cand, mode)
                    if cyc is not None and cyc < cur[ci][0]:
                        cur[ci] = [cyc, cand]
                        improved_any = True
                        dump(out, sorted(cur, key=lambda r: r[0]))
                        print(f"[{time.time()-t0:6.1f}s] cand {ci}: "
                              f"{cyc} cyc  {dict(zip(dims, cand))}",
                              flush=True)
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
    pr.add_argument("--bound", type=float, default=10.0)
    pr.add_argument("--seed", type=int, default=2024)
    pr.add_argument("--mode", choices=["single", "interp"], default="single")
    pf = sub.add_parser("finetune", help="coordinate descent around candidates")
    pf.add_argument("budget", type=float, help="seconds")
    pf.add_argument("--in", dest="infile", default=None)
    pf.add_argument("--out", default=None)
    pf.add_argument("--top", type=int, default=6)
    pf.add_argument("--mode", choices=["single", "interp"], default="single")
    args = ap.parse_args()

    if args.phase == "random":
        out = args.out or f"_weights_found_{args.mode}.json"
        phase_random(args.budget, out, args.bound, args.seed, args.mode)
    else:
        infile = args.infile or f"_weights_found_{args.mode}.json"
        out = args.out or f"_weights_refined_{args.mode}.json"
        phase_finetune(args.budget, infile, out, args.top, args.mode)


if __name__ == "__main__":
    main()
