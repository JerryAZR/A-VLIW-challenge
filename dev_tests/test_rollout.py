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
                DebugVCompare, Gather)
from regalloc import (tag_raw_chains, build_dag, RegisterAllocator,
                      schedule as greedy_schedule)
from rollout import schedule_rollout, ScoreWeights, _features, _score
from scheduler import Weights

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
        b_roll = schedule_rollout(dag, rc, seed=42, trials=1, greedy_trials=1,
                                  weights=W, allocator=alloc2)
        self.assertEqual(b_greedy, b_roll)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_bundles(self):
        results = []
        for _ in range(2):
            _, rc, dag = _tagged_dag()
            b = schedule_rollout(dag, rc, seed=42, trials=4, greedy_trials=1,
                                 weights=W, allocator=RegisterAllocator(rc))
            results.append(b)
        self.assertEqual(results[0], results[1])

    def test_different_seed_may_differ_but_completes(self):
        _, rc, dag = _tagged_dag()
        b = schedule_rollout(dag, rc, seed=7, trials=4, greedy_trials=0,
                             weights=W, allocator=RegisterAllocator(rc))
        self.assertTrue(any(b for b in b if b))   # some non-empty bundle


class TestAllocatorHygiene(unittest.TestCase):
    """After a full trial-heavy pass, the allocator must be consistent:
    pools hold unique addresses and no live register aliases a free one."""

    def test_no_leaks_or_pool_corruption(self):
        _, rc, dag = _tagged_dag()
        alloc = RegisterAllocator(rc)
        schedule_rollout(dag, rc, seed=42, trials=6, greedy_trials=1,
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
            schedule_rollout(dag, rc, seed=42, trials=3, greedy_trials=1,
                             weights=W, allocator=alloc)


class TestScoring(unittest.TestCase):
    def test_score_is_linear_dot(self):
        w = ScoreWeights(alu_work=2.0, reg_delta=-3.0, frontier=0.5,
                         reads=1.5, obligations=-2.0)
        feats = {"alu_work": 10, "reg_delta": 4, "frontier": 30,
                 "reads": 6, "obligations": 12}
        self.assertAlmostEqual(_score(feats, w),
                               2 * 10 - 3 * 4 + 0.5 * 30 + 1.5 * 6 - 2 * 12)


if __name__ == "__main__":
    unittest.main()
