"""Diagnose the rollout deadlock: what registers are live when it stalls,
and what do the stuck ready nodes want?"""

from collections import Counter
import random

from problem import Tree, Input
import perf_takehome as pt
from regalloc import (tag_raw_chains, build_dag, recompute_read_count,
                      RegisterAllocator)
from scheduler import prune_to_stores
from rollout import schedule_rollout, ScoreWeights

random.seed(123)
forest = Tree.generate(10)
inp = Input.generate(forest, 256, 16)

kb = pt.KernelBuilder.__new__(pt.KernelBuilder)
kb.instrs = []
kb.allocator = None

# Reproduce _emit_regalloc's pipeline up to the body schedule.
kb.build_kernel  # noqa - just to show intent; we call the real one below

kb2 = pt.KernelBuilder()
import types

orig_schedule = schedule_rollout
captured = {}


def spy(dag, read_count, **kw):
    captured["allocator"] = kw.get("allocator")
    captured["dag"] = dag
    return orig_schedule(dag, read_count, **kw)


import rollout
rollout.schedule_rollout = spy
try:
    kb2.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)
except RuntimeError as e:
    print("DEADLOCK:", str(e).split("\n")[0])
finally:
    rollout.schedule_rollout = orig_schedule

alloc = captured["allocator"]
dag = captured["dag"]

# Live tag histogram by base name.
names = Counter()
for tag in alloc.assigned:
    names[tag.name.split("#")[0].split("[")[0]] += 1
print(f"\nlive vector registers: {len(alloc.assigned)}")
print("live by base name:", names.most_common(15))

# What are the uncommitted ready nodes?
ready = sorted(dag.ready())
kinds = Counter(type(dag[i].instr).__name__ for i in ready)
print(f"\nready nodes: {len(ready)}", kinds.most_common())

# How many reads would each live tag still need (remaining)?
rem = Counter()
for tag, r in alloc.remaining.items():
    rem[r] += 1
print("\nreads-remaining histogram (live tags):", sorted(rem.items()))
