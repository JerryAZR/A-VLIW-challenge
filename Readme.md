# VLIW/SIMD Performance Take-Home — 1,110 cycles (133.1×)

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
  and what's left on the table.
- **`notes/optimization_log.md`** — append-only per-step history with
  mechanisms and PMU tables.
- **`notes/original_challenge.md`** — the original challenge README,
  verbatim, including its validation protocol and the contact it lists
  for sub-1,487 submissions.
