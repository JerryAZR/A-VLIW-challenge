"""Generate a consistent set of debugging logs from a single kernel build.

One-line entry point:

    python gen_logs.py [outdir]

Produces, in `outdir` (default "logs"), a cross-referenceable set of logs all
from the SAME build (so rename ids line up across files):

    autofree.txt  - post-auto-free instruction stream (input instrs + auto
                    Free directives), each tagged with its stable rid.
    alloc.txt     - every home allocation (BIRTH/REHOME old->new) and free,
                    tied to the triggering instruction's rid.
    gather.txt    - every Gather decomposition into per-lane loads.
    trace.txt     - full execution trace from the TracingMachine: every
                    executed slot with cycle, rid, engine, encoding, and the
                    addresses/values it read and wrote.
    build.txt     - build metadata: cycle count, output correctness summary.

The rename-pass logs (autofree/alloc/gather) are emitted by the rename engine
when rename.DEBUG_DIR is set; the execution trace is emitted by the
TracingMachine reading the end-to-end rid on each TaggedSlot.
"""

import os
import random
import sys

import rename
from problem import (Tree, Input, build_mem_image, N_CORES,
                     reference_kernel2)
import perf_takehome as pt
from trace_sim import TracingMachine


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "logs"
    os.makedirs(outdir, exist_ok=True)

    # Same problem instance the kernel tests use.
    random.seed(123)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem = build_mem_image(forest, inp)

    # Reference result + value_trace (the debug oracle's source of truth).
    ref_mem = list(mem)
    vt = {}
    for _ in reference_kernel2(ref_mem, vt):
        pass

    # Point the rename engine's logs at outdir, then build the kernel.
    # prune=False: keep the debug vcompare nodes so the build matches what the
    # oracle-checker sees.
    rename.DEBUG_DIR = outdir

    kb = pt.KernelBuilder()
    kb.build_kernel(10, 2047, 256, 16, prune=False)

    # Run the tracing simulator over the built program.
    trace_path = os.path.join(outdir, "trace.txt")
    machine = TracingMachine(list(mem), kb.instrs, kb.debug_info(),
                             n_cores=N_CORES, value_trace=vt,
                             resolved_body=kb.resolved_body,
                             log_path=trace_path)
    machine.enable_debug = False
    machine.run()
    crashed = False
    try:
        machine.run()
    except IndexError:
        crashed = True
    machine.close()

    # Correctness summary against the reference.
    p = mem[6]
    mine = machine.mem[p : p + 256]
    ref = ref_mem[p : p + 256]
    wrong = [(i, a, b) for i, (a, b) in enumerate(zip(mine, ref)) if a != b]
    bad_groups = sorted({i // 8 for i, _, _ in wrong})

    with open(os.path.join(outdir, "build.txt"), "w") as f:
        f.write(f"cycles: {machine.cycle}\n")
        f.write(f"crashed: {crashed}\n")
        f.write(f"wrong outputs: {len(wrong)}/256\n")
        f.write(f"affected groups: {bad_groups}\n")

    print(f"logs written to {outdir}/")
    print(f"  cycles={machine.cycle} crashed={crashed} "
          f"wrong={len(wrong)}/256 groups={bad_groups}")
    for name in ("autofree", "alloc", "gather", "trace", "build"):
        path = os.path.join(outdir, name + ".txt")
        if os.path.exists(path):
            n = sum(1 for _ in open(path))
            print(f"  {name}.txt: {n} lines")


if __name__ == "__main__":
    main()
