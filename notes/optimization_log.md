# Optimization Log

Worked example: Anthropic's VLIW/SIMD performance-engineering take-home.
Canonical workload: `forest_height=10, n_nodes=2047, batch_size=256,
rounds=16`. Scored by simulated cycle count on a frozen copy of the
simulator (`tests/submission_tests.py`).

Each entry records intent, mechanism, cycle count, and PMU corroboration.

---

## Baseline (committed up-stream)               147 734 cyc   1.00×

`KernelBuilder.build_kernel` as shipped, a deliberately-naive scalar program:

- One slot per instruction bundle (no VLIW packing at all — `vliw=False`).
- Fully unrolled `rounds × batch_size` (= 4096) iterations emitted statically.
- Re-reads `idx[i]`, `val[i]` from **mem** every round (one `load` apiece),
  plus one `load` for `tree.values[idx]`.
- Stores `idx[i]` AND `val[i]` back to **mem** every round (the `idx` writes
  are entirely wasted — the grader checks only `val`).
- `myhash` taken literally: 6 stages × 3 alu ops = **18 ops/hash**.
- Idx update `% 2`, `== 0`, `flow select`, `* 2`, `+`, `< n_nodes`,
  `flow select`. Wrap is fully per-lane branchy.
- Utility slot count per lane per round: ~29 alu, 3 load, 2 store, 2 flow.
  All scalar; `valu` untouched.

PMU: alu 118784 (6.7% util), valu 0, load 12564, store 8192, flow 8194.
Every active engine sits at histogram k=1 (one-slot-per-bundle pathology).

---

## v1 — fma via `valu` `multiply_add`         123 165 cyc   1.20×

Commit `648da3d`. First reduction of `myhash`:

- Three linear stages `(a+K) + (a<<s) == a*(1+2^s) + K` collapse from three
  alu slots to **one `valu multiply_add` slot each** (verified bit-exact
  against `myhash`). Using `multiply_add` as a scalar fma on lane 0 of a
  VLEN-word work region; broadcast constants fill lanes 1..7 with junk we
  ignore.
- Three xor/add-shift stages (1, 3, 5) are irreducible at 3 slots each
  — carries block fusing `+K` across the `^` combine (numerically falsified).
- ⇒ `myhash` goes from 18 literal alu ops to **12 slots** (3 fma + 9 alu).
- Everything else still sequential & bloated (per-round mem round-trips,
  branchy `%`/`select`/`<`/`select` idx/wrap, idx stored).

Per-lane per-round slot count: 12 hash (vs 18) + ~6 idx/wrap = ~20 → ~18.
PMU: alu 118784 → 81920, valu 0 → 12296, others unchanged.

---

## v2 — scratch-resident state + branchless idx
                                                77 223 cyc   1.91×
Commit `ff00b76`. Plumbing fixes (no compute parallelism yet):

- `val[256]`, `idx[256]` **resident in scratch** across all 16 rounds.
  Prologue `vload`s `val` once (32 vloads); `idx` starts at 0 (scratch is
  zero-initialized → no init needed). Epilogue `vstore`s `val` once.
  No per-round mem round-trips for lane state.
- `val[i]` doubles as the running hash register via a shared transient
  8-word `work_vec` (lane 0 active for fma). Stage 5's final combine writes
  `v` **directly into `val[i]`** — no separate copy-back op.
- Per-lane stage temps `t1[i]`, `t2[i]` (single-word, never shared across
  lanes — no rename hazards when we later pack VLIW bundles).
- **Branchless idx update**: `base = (idx<<1)|1`; `next = base + (v & 1)`.
  No `%`, no `select`, no flow port consumed.
- **Wrap as build-time per-round decision**: verified that for the canonical
  shape all lanes are at the same tree level in every round and that the
  wrap is therefore uniform and lands exactly on round = `forest_height`
  (=10). On that round we skip the idx update and write `idx[i] := 0` in one
  op. No per-lane wrap test in the hot path.
- Constants properly mapped: scalar (load const) + vector (vbroadcast) at
  prologue; reused across all 4096 hashes.
- Pause-ordering fix: epilogue (vstore) BEFORE pause 2, so machine.mem holds
  the final values when the test compares at the reference's final yield
  (caught by `Incorrect result on round 1` during dev).

PMU before/after:

| engine   | v1 fires | v2 fires | delta |
|----------|----------|----------|-------|
| store    | 8192     | **32**   | −8160 (only final vstore) |
| flow     | 8194     | **2**    | −8192 (branchless idx + build-time wrap) |
| load     | 12565    | 4157     | −8408 (no idx/val re-reads) |
| alu      | 81920    | 60736    | −21184 (dropped %/==/</select/* redundancies) |
| valu     | 12294    | 12296    | unchanged (intrinsic fma count) |

Passes `submission_tests.py` correctness (8 seeds) and the first two speed
tiers: baseline `< 147734`, updated-starting-point `< 18532`. Fails the
`< 2164` tier (expected — that lives in the VLIW-packing + vectorization
regime).

---

## v3 — (next) gather prefetch / VLIW packing / cross-lane vectorization

Status: not started. Design discussion points documented in
`notes/architecture.md`, `notes/scratch_map_canonical.md`, `notes/hash_dag.md`.

Approximate targets from the roofline analysis:
- gather prefetch alone : ~77k → ~50k (modest; load hidden behind compute)
- VLIW packing of the hash : ~50k → ~8–10k (large; ~9-cycle critical path
  per lane, port-pressure bound)
- cross-lane vectorization (8 lanes per valu) : ~8–10k → ~1500 (lands at
  the Opus-4.5 tier; the verified per-lane compute-floor regime)
- cache-swap / scratch-tree tricks : N/A — scratch has no indirect
  addressing, tree must stay in mem; only load-port sharing via broadcast
  of coincident idx values helps (and even then ~450 cyc, mostly hidden)