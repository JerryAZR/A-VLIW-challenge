# AI Assistant Guide

This optimization challenge is driven by the user. As the assistant, your job is to:
1. Carry out implementation and investigation tasks given by the user.
2. Give suggestions when you see a promising lead.

You should NOT:
1. Implement any "optimizations" not asked for by the user.
2. Overthink a problem when you could just ask the user.

## Testing correctness with a sub-optimal scheduler

The real machine has a fixed `SCRATCH_SIZE` (1536 words = 185 vector granules
after the scalar pool). When a change raises register pressure, the greedy
list scheduler may deadlock there even though the kernel's *dataflow* is
correct - it just can't fit the parallelism into 185 granules.

`_check_big.py` is a correctness testbench that grows the scratch file
(patching `SCRATCH_SIZE` in both the allocator's pool sizing and the
simulator's per-core scratch array, `python _check_big.py [size]`, default
4096) so the schedule completes and the `DebugVCompare` node_val / hashed_val
oracle can validate the kernel's dataflow (select logic, path recompute,
gather) independent of register pressure.

Use it to confirm a change is *correct* before investing in scheduler work to
make it *fit* at 1536. It is NOT a grading harness - `SCRATCH_SIZE` is frozen
by `problem.py`; the patch is validation-only. `_check.py` remains the
real-machine check (1536 scratch).
