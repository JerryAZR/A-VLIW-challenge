# VLIW/SIMD Performance Take-Home — 1,110 cycles (133.1×)

[![CI](https://github.com/JerryAZR/A-VLIW-challenge/actions/workflows/ci.yml/badge.svg)](https://github.com/JerryAZR/A-VLIW-challenge/actions/workflows/ci.yml)
[![cycles](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JerryAZR/A-VLIW-challenge/badge/cycles.json)](https://github.com/JerryAZR/A-VLIW-challenge/actions/workflows/ci.yml)

A completed solution to Anthropic's original performance-engineering
take-home (preserved verbatim at `notes/original_challenge.md`):
optimize a 256-lane × 16-round hash/tree-descent kernel for a simulated
single-core VLIW+SIMD machine — alu 12 · valu 6 · load 2 · store 2 ·
flow 1 slots per cycle, a 1,536-word scratch register file, and
read-before-write bundles with unit latency.

| | cycles |
|---|---:|
| Baseline (as shipped) | 147,734 |
| **This repo** | **1,110** (133.1×) |
| Compute-work floor of the final algorithm | 1,077 |
| Claude Opus 4.5 (11.5 h harness) | 1,487 |
| Claude Opus 4.5 (improved harness, best listed) | 1,363 |

All nine performance tiers in `tests/submission_tests.py` pass, including
`opus45-improved-harness < 1363` (253 cycles clear).

## Provenance & attribution

Everything up to and including **1,110 cycles** (step 26, commit
`e728293`) was developed independently, without knowledge of other
solutions — the complete record is `JOURNEY.md` +
`notes/optimization_log.md` (steps 1–26).

Any optimization *below* 1,110 builds on ideas borrowed afterward from
external solutions — principally **rubinownz111's 1,063-cycle repo**
(github.com/rubinownz111/1063-cycles-original-performance-takehome).
Each borrowed idea is credited where it is recorded
(`notes/next_steps.md`, "Prior art" section) and again in the
optimization log when it ships.

## Reproduce

```
python tests/submission_tests.py   # 1,110 cycles, 8 seeds, tier report
python -m tools._check             # per-round correctness oracle
git diff origin/main tests/        # empty - tests untouched
```

## Repo layout

- `problem.py` — the machine/ISA spec + simulator (given, unmodified)
- `perf_takehome.py` — the kernel builder (the deliverable)
- `ir.py` — typed instruction IR with symbolic operands
- `scheduler.py` — dependency DAG, node properties, functional-unit pool
- `regalloc.py` — SSA tag chains, RAW-only DAG, dynamic register allocator
- `rollout.py` — list scheduler with per-cycle checkpoint/rollback
- `tests/` — frozen grading harness (untouched)
- `dev_tests/` — unit tests (19)
- `notes/` — design docs, machine reference, original challenge text
- `tools/` — dev tooling (trainers, checkers, diagnostics);
  run as `python -m tools.x`
- `weights/` — trained priority-weight artifacts

## Where the story lives

- **`JOURNEY.md`** — the curated writeup: eight eras, the final-design
  inventory, the regressions we kept, the graveyard of superseded ideas,
  and what's left on the table. (Covers only the independent era,
  147,734 → 1,110 — see "Provenance & attribution" above.)
- **`notes/optimization_log.md`** — append-only per-step history with
  mechanisms and PMU tables.
- **`notes/original_challenge.md`** — the original challenge README,
  verbatim, including its validation protocol and the contact it lists
  for sub-1,487 submissions.
