"""Unit tests for the rollout scheduler (rollout.py).

Key properties tested on a small synthetic DAG (fast, no simulator):
  - golden equivalence: rollout with a single greedy trial must reproduce
    regalloc.schedule's bundles exactly (same placement semantics),
  - determinism: same seed -> byte-identical bundles across runs,
  - allocator hygiene: no register leaks / pool corruption after a full
    trial-heavy pass,
  - deadlock handling: an unplaceable cycle in every trial raises.

Run: python -m unittest dev_tests.test_rollout -v
"""

import unittest

from ir import (Sym, Alu, VecElem, VecFma, VBroadcast, Const, VStore, VSelect,
                DebugVCompare, Gather, Reg)
from regalloc import (tag_raw_chains, build_dag, RegisterAllocator,
                      schedule as greedy_schedule)
from rollout import (schedule_rollout, ScoreWeights, SortCtx, _features,
                     _score, make_random_order, make_weighted_greedy,
                     make_interp_greedy)
from scheduler import (Weights, DNode, DAG, FuncUnitPool, _classify,
                       _KIND_VEC_ELEM, _KIND_VEC_FMA)
from problem import VLEN

W = Weights(sink=-1, load=5, raw=1, war=1, rigid=1, idx=-4)


def _synthetic():
    """A small program exercising every placement kind: atomic alu, vec_elem
    (spillable), vec_fma (rigid), gather (partial), store, flow, debug."""
    a = Sym("a", True); b = Sym("b", True); c = Sym("c", True); d = Sym("d", True)
    s = Sym("s"); o = Sym("o")
    return [
        Const(s, 5),
        Const(o, 40),
        VBroadcast(a, s),
        VecElem("+", b, a, a),
        VecFma(c, b, a, a),
        VecElem("&", d, c, a),
        Gather(b, d),                # b[j] = mem[scratch[d+j]]
        VecElem("^", a, a, b),
        VSelect(c, d, a, b),
        Alu("+", s, s, o),
        VecElem("+", d, c, a),
        VecFma(a, d, b, c),
        VStore(s, a),
        DebugVCompare(a, [("k0", 0)]),
        VecElem("-", b, a, d),
        VecElem("*", c, b, a),
        VStore(o, c),
        DebugVCompare(c, [("k1", 1)]),
    ]


def _tagged_dag():
    tagged, read_count = tag_raw_chains(_synthetic())
    return tagged, read_count, build_dag(tagged)


class TestGreedyEquivalence(unittest.TestCase):
    """rollout(trials=1, greedy) must be byte-identical to greedy schedule."""

    def test_same_bundles(self):
        tagged, rc, dag = _tagged_dag()
        alloc1 = RegisterAllocator(rc)
        b_greedy = greedy_schedule(dag, rc, seed=42, picker="weighted",
                                   weights=W, allocator=alloc1)
        dag.reset()
        alloc2 = RegisterAllocator(rc)
        b_roll = schedule_rollout(dag, rc, seed=42, trials=1,
                                  weights=W, allocator=alloc2)
        self.assertEqual(b_greedy, b_roll)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_bundles(self):
        results = []
        for _ in range(2):
            _, rc, dag = _tagged_dag()
            b = schedule_rollout(dag, rc, seed=42, trials=4,
                                 weights=W, allocator=RegisterAllocator(rc))
            results.append(b)
        self.assertEqual(results[0], results[1])

    def test_pure_random_sort_funcs_complete(self):
        """An explicit sort_funcs trial set is honoured: 4 pure shuffles."""
        import random as _r
        _, rc, dag = _tagged_dag()
        b = schedule_rollout(
            dag, rc, seed=7,
            sort_funcs=[make_random_order(_r.Random(7)) for _ in range(4)],
            weights=W, allocator=RegisterAllocator(rc))
        self.assertTrue(any(b for b in b if b))   # some non-empty bundle


class TestAllocatorHygiene(unittest.TestCase):
    """After a full trial-heavy pass, the allocator must be consistent:
    pools hold unique addresses and no live register aliases a free one."""

    def test_no_leaks_or_pool_corruption(self):
        _, rc, dag = _tagged_dag()
        alloc = RegisterAllocator(rc)
        schedule_rollout(dag, rc, seed=42, trials=6,
                         weights=W, allocator=alloc)
        for pool in (alloc.free_vec, alloc.free_scalar):
            self.assertEqual(len(pool), len(set(pool)))
        live = set(alloc.assigned.values())
        for pool in (alloc.free_vec, alloc.free_scalar):
            self.assertTrue(live.isdisjoint(pool))
        # Every remaining live tag still expects future reads.
        for tag, rem in alloc.remaining.items():
            self.assertGreater(rem, 0)


class TestDeadlock(unittest.TestCase):
    def test_all_trials_stuck_raises(self):
        tagged, rc = tag_raw_chains(_synthetic())
        dag = build_dag(tagged)
        alloc = RegisterAllocator(rc)
        # Drain the entire vector pool with live pads (1 read each, never
        # read -> never freed), so no vector write can ever allocate.
        for i in range(alloc.vec_pool_size):
            pad = Sym(f"pad{i}", True)
            rc[pad] = 1
            alloc.write(pad)
        with self.assertRaises(RuntimeError):
            schedule_rollout(dag, rc, seed=42, trials=3,
                             weights=W, allocator=alloc)


class TestPoolSwap(unittest.TestCase):
    """Rigid valu (vec_fma) with a full valu unit may evict a valu-placed
    vec_elem down to the alu (1 valu slot <-> VLEN alu lanes), if the alu
    can take all lanes. Slot-mapping only: the evicted node stays complete."""

    def _elem(self, i):
        n = DNode(idx=i, engine="valu",
                  instr=VecElem("+", Reg(100 + 8 * i, True),
                                Reg(200, True), Reg(208, True)))
        _classify(n)
        return n

    def _fma(self, i):
        n = DNode(idx=100 + i, engine="valu",
                  instr=VecFma(Reg(300, True), Reg(308, True),
                               Reg(316, True), Reg(324, True)))
        _classify(n)
        return n

    def _fill(self, pool, n_elems, n_alu_used, dry=False):
        for _ in range(n_alu_used):
            pool.free["alu"] -= 1
        elems = [self._elem(i) for i in range(n_elems)]
        for e in elems:
            assert pool.place(e, e.placement, dry) is True
        return elems

    def test_swap_fits_fma(self):
        pool = FuncUnitPool()
        pool.reset()
        elems = self._fill(pool, 6, n_alu_used=4)   # valu full, alu 8 free
        self.assertEqual(pool.free["valu"], 0)
        fma = self._fma(0)
        self.assertIs(pool.place(fma, fma.placement), True)
        # fma on valu; evicted elem moved wholly to alu.
        self.assertEqual(pool.free["valu"], 0)
        self.assertEqual(pool.free["alu"], 0)
        self.assertIn(fma.instr, pool.bundle["valu"])
        self.assertEqual(len(pool.bundle["valu"]), 6)      # 5 elems + fma
        self.assertEqual(len(pool.bundle["alu"]), VLEN)    # evicted lanes
        evicted = elems[-1].placement                     # LIFO eviction
        self.assertEqual(evicted.engine_choice, "alu")
        self.assertEqual(evicted.lanes_done, VLEN)         # still complete

    def test_no_swap_when_alu_short(self):
        pool = FuncUnitPool()
        pool.reset()
        self._fill(pool, 6, n_alu_used=5)      # alu has only 7 < VLEN free
        fma = self._fma(0)
        self.assertIsNone(pool.place(fma, fma.placement))
        self.assertEqual(len(pool.bundle["valu"]), 6)      # nothing evicted

    def test_swap_in_dry_mode(self):
        pool = FuncUnitPool()
        pool.reset()
        self._fill(pool, 6, n_alu_used=4, dry=True)
        fma = self._fma(0)
        self.assertIs(pool.place(fma, fma.placement, dry=True), True)
        self.assertEqual(pool.free["valu"], 0)
        self.assertEqual(pool.free["alu"], 0)
        self.assertEqual(pool.bundle, {})                  # dry: no emission


class TestScoring(unittest.TestCase):
    def test_score_is_linear_dot(self):
        w = ScoreWeights(alu_work=2.0, reg_delta=-3.0, frontier=0.5,
                         reads=1.5, obligations=-2.0)
        feats = {"alu_work": 10, "reg_delta": 4, "frontier": 30,
                 "reads": 6, "obligations": 12}
        self.assertAlmostEqual(_score(feats, w),
                               2 * 10 - 3 * 4 + 0.5 * 30 + 1.5 * 6 - 2 * 12)


class TestGroupProps(unittest.TestCase):
    """DAG-derived sink groups: sinks sorted by depth get ids 1..N; every
    node inherits its DEEPEST descendant sink's id (ties -> lower id);
    nodes reaching no non-debug sink get 0."""

    def _dag(self):
        # shared source 0 -> chain A: 0 -> 1 -> 3(store)    depths 0,1,2
        #                 -> chain B: 0 -> 2 -> 4 -> 5(store) depths 0,1,2,3
        # debug sink: 6
        def n(i, eng):
            return DNode(idx=i, engine=eng, instr=None)
        nodes = [n(0, "alu"), n(1, "alu"), n(2, "alu"), n(3, "store"),
                 n(4, "alu"), n(5, "store"), n(6, "debug")]
        for src, dst in [(0, 1), (1, 3), (0, 2), (2, 4), (4, 5)]:
            nodes[dst].in_edges.append((src, 1))
            nodes[src].out_edges.append((dst, 1))
        return DAG.from_nodes(nodes)

    def test_group_assignment(self):
        dag = self._dag()
        g = [node.props.group for node in dag.nodes]
        # Sinks by depth: node 3 (depth 2) -> id 1, node 5 (depth 3) -> id 2.
        self.assertEqual(g[3], 1 / 2)
        self.assertEqual(g[5], 1.0)
        # Chain members inherit their sink.
        self.assertEqual(g[1], 1 / 2)
        self.assertEqual(g[2], 1.0)
        self.assertEqual(g[4], 1.0)
        # Shared source gets the DEEPER sink's group (5, not 3).
        self.assertEqual(g[0], 1.0)
        # Debug sink is ungrouped.
        self.assertEqual(g[6], 0.0)


class TestInterpGreedy(unittest.TestCase):
    """progress=0 must reproduce make_weighted_greedy(w2); progress=1 -> w1."""

    def test_endpoints(self):
        _, rc, dag = _tagged_dag()
        for n in dag.nodes:
            from scheduler import _classify
            _classify(n)
        alloc = RegisterAllocator(rc)
        w1 = Weights(sink=10, load=0, raw=0, war=0, rigid=0, idx=0)
        w2 = Weights(sink=0, load=0, raw=0, war=0, rigid=0, idx=-10)
        ready = [dag.nodes[i] for i in sorted(dag.ready())]
        interp = make_interp_greedy(w1, w2)
        greedy1 = make_weighted_greedy(w1)
        greedy2 = make_weighted_greedy(w2)
        ctx0 = SortCtx(allocator=alloc, progress=0.0)
        ctx1 = SortCtx(allocator=alloc, progress=1.0)
        self.assertEqual([n.idx for n in interp(ready, ctx0)],
                         [n.idx for n in greedy2(ready, ctx0)])
        self.assertEqual([n.idx for n in interp(ready, ctx1)],
                         [n.idx for n in greedy1(ready, ctx1)])


if __name__ == "__main__":
    unittest.main()
