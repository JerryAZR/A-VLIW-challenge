"""Correctness check with an enlarged scratch register file.

The real machine has SCRATCH_SIZE=1536 (185 vector granules); the greedy
scheduler deadlocks there once tree7-14 + in-flight hash temps saturate the
pool. This testbench grows the scratch file (patching both the allocator's pool
sizing and the simulator's per-core scratch array) so the schedule completes
and the node_val / hashed_val DebugVCompare oracle can validate the kernel's
DATAFLOW (select logic, path recompute, gather) independent of register
pressure.

NOT a grading harness - SCRATCH_SIZE is frozen by problem.py. Validation only.

    python -m tools._check_big [scratch_size]   # default 4096
"""
import random
import sys

import problem
import regalloc

BIG = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
problem.SCRATCH_SIZE = BIG
regalloc.SCRATCH_SIZE = BIG

from problem import (Tree, Input, build_mem_image, Machine, N_CORES,
                     reference_kernel2)
import perf_takehome as pt

random.seed(123)
forest = Tree.generate(10)
inp = Input.generate(forest, 256, 16)
mem = build_mem_image(forest, inp)

ref_mem = list(mem)
vt = {}
round_refs = []
for s in reference_kernel2(ref_mem, vt):
    round_refs.append(list(s[s[6]:s[6] + len(inp.values)]))
print(f"reference rounds: {len(round_refs)}  (scratch_size={BIG})")

kb = pt.KernelBuilder()
kb.build_kernel(10, 2047, 256, 16, prune=True)
machine = Machine(list(mem), kb.instrs, kb.debug_info(), n_cores=N_CORES,
                  value_trace=vt, trace=False, scratch_size=BIG)
machine.enable_debug = False

inp_values_p = mem[6]
ok = True
for i in range(len(round_refs)):
    machine.run()
    mine = machine.mem[inp_values_p:inp_values_p + len(inp.values)]
    if list(mine) != round_refs[i]:
        bad = [j for j, (a, b) in enumerate(zip(mine, round_refs[i])) if a != b]
        print(f"round {i}: INCORRECT, {len(bad)} mismatches at {bad[:5]}")
        ok = False
        break
print("CORRECT - all rounds match" if ok else "FAILED")
print("CYCLES:", machine.cycle)
