"""Correctness check mirroring do_kernel_test, but with the reference
generator materialised first (avoids slow interleaving) and run() called
once per round (Machine halts at each pause)."""
import random
import time
from problem import Tree, Input, build_mem_image, Machine, N_CORES, reference_kernel2
import perf_takehome as pt

random.seed(123)
forest = Tree.generate(10)
inp = Input.generate(forest, 256, 16)
mem = build_mem_image(forest, inp)

# Materialise the reference rounds first AND capture the value_trace the
# debug vcompare oracle reads.
ref_mem = list(mem)
vt = {}
round_refs = []
for ref_state in reference_kernel2(ref_mem, vt):
    p = ref_state[6]
    round_refs.append(list(ref_state[p : p + len(inp.values)]))
print(f"reference rounds: {len(round_refs)}")

kb = pt.KernelBuilder()
kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)
machine = Machine(list(mem), kb.instrs, kb.debug_info(), n_cores=N_CORES,
                  value_trace=vt, trace=False)

inp_values_p = mem[6]
ok = True
for i in range(len(round_refs)):
    machine.run()
    mine = machine.mem[inp_values_p : inp_values_p + len(inp.values)]
    if list(mine) != round_refs[i]:
        bad = [j for j, (a, b) in enumerate(zip(mine, round_refs[i])) if a != b]
        print(f"round {i}: INCORRECT, {len(bad)} mismatches at {bad[:5]}")
        ok = False
        break
print("CORRECT - all rounds match" if ok else "FAILED")
print("CYCLES:", machine.cycle)
