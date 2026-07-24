"""Validate the schedule against the DAG, matching nodes to cycles by NODE
INDEX (not lowered tuple - many distinct nodes lower to identical tuples).

We re-run schedule() with a hook that records the cycle each node index is
placed, then check every in-edge: RAW parents must be strictly earlier, WAR
parents same-or-earlier.
"""
import heapq
import perf_takehome as pt
from scheduler import DAG, FuncUnitPool, _make_picker, _classify, Weights
from rename import rid_of

# The resolved body is kept on the builder after build_kernel (no monkeypatch).
kb = pt.KernelBuilder(); kb.build_kernel(10, 2047, 256, 16)
body = kb.resolved_body
dag = DAG(body)

placements = [_classify(n) for n in dag.nodes]
key_fn = _make_picker("weighted", placements, __import__('random').Random(42),
                      dag.props, Weights(sink=-3, load=-1.5, raw=-0.25, war=6,
                                         rigid=0.25, idx=-4))

# Re-run the schedule loop, recording the cycle each node commits.
cycle_of = {}
pool = FuncUnitPool()
C = 0
committed = 0
total = len(dag)
cap_cycles = total + 1
while committed < total:
    ready = dag.ready()
    pool.reset()
    working = [(key_fn(i), i) for i in ready]
    heapq.heapify(working)
    while working:
        _, idx = heapq.heappop(working)
        if pool.place(dag[idx], placements[idx]):
            cycle_of[idx] = C
            committed += 1
            for u in dag.commit(idx):
                heapq.heappush(working, (key_fn(u), u))
    dag.advance()
    C += 1
    if C > cap_cycles:
        raise RuntimeError("stuck")

print(f"scheduled {committed}/{total} nodes in {C} cycles")

viol = 0
for n in dag.nodes:
    cn = cycle_of.get(n.idx)
    if cn is None:
        continue
    for src, w in n.in_edges:
        cs = cycle_of.get(src)
        if cs is None:
            continue
        if w == 1 and not (cn > cs):
            if viol < 12:
                print(f"RAW viol: node {n.idx}(rid={rid_of(n.instr)},cyc{cn}) "
                      f"<= parent {src}(rid={rid_of(dag.nodes[src].instr)},cyc{cs})  {n.instr}")
            viol += 1
        elif w == 0 and not (cn >= cs):
            if viol < 12:
                print(f"WAR viol: node {n.idx}(rid={rid_of(n.instr)},cyc{cn}) "
                      f"< parent {src}(rid={rid_of(dag.nodes[src].instr)},cyc{cs})  {n.instr}")
            viol += 1
print(f"violations: {viol}")
