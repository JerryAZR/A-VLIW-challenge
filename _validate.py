"""Validate the regalloc schedule against the RAW-only DAG.

The emitted body bundles carry TaggedSlots (with rid), so we derive the cycle
each DAG node committed directly from the built program - no re-running the
scheduler. Then we check every RAW in-edge: a parent must commit STRICTLY
earlier than its consumer (read-before-write / 1-cycle latency).

The regalloc DAG is RAW-only (all edges weight 1): tags are unique per write,
so there are no WAR/WAW anti-dependencies to validate.
"""
import perf_takehome as pt

kb = pt.KernelBuilder()
kb.build_kernel(10, 2047, 256, 16)
dag = kb.body_dag

# Cycle each node committed, keyed by rid (from the emitted body bundles).
# A node's commit cycle = the LAST cycle any of its slots appears (a partial
# node like a gather or spilled vec_elem commits only when its final lane
# lands; reads of its register happen at/after that).
cycle_of_rid = {}
pause_idxs = [i for i, instr in enumerate(kb.instrs)
              if "flow" in instr and instr["flow"][0][0] == "pause"]
body_start = pause_idxs[0] + 1
for rel_cyc, bundle in enumerate(kb.instrs[body_start:]):
    for slot in bundle.values():
        for s in slot:
            rid = getattr(s, "rid", -1)
            if rid >= 0:
                cycle_of_rid[rid] = rel_cyc  # keep last occurrence

cycle_of = {n.idx: cycle_of_rid[n.instr.rid] for n in dag.nodes
            if n.instr.rid in cycle_of_rid}
print(f"placed {len(cycle_of)}/{len(dag)} nodes over "
      f"{max(cycle_of.values(), default=-1) + 1} body cycles")

viol = 0
for n in dag.nodes:
    cn = cycle_of.get(n.idx)
    if cn is None:
        continue
    for src, w in n.in_edges:
        cs = cycle_of.get(src)
        if cs is None:
            continue
        assert w == 1, f"unexpected non-RAW edge weight {w}"
        if not (cn > cs):
            if viol < 12:
                print(f"RAW viol: node {n.idx}(rid={n.instr.rid},cyc{cn}) "
                      f"<= parent {src}(rid={dag.nodes[src].instr.rid},cyc{cs})"
                      f"  {n.instr}")
            viol += 1
print(f"violations: {viol}")
