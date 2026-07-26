"""Unit tests for the checkpoint/rollback interfaces (DAG, RegisterAllocator,
_Placement). Each structure owns its own rollback; these tests exercise them
in isolation so a corruption bug is caught here, not in a full sim run.

Run: python -m unittest dev_tests.test_rollback -v
"""

import random
import unittest
from collections import deque

from ir import Sym
from regalloc import RegisterAllocator
from scheduler import DAG, DNode, _Placement, _KIND_VEC_ELEM, _KIND_ATOMIC_SCALAR


def _random_dag(rng: random.Random, n: int, edge_p: float = 0.15) -> DAG:
    """A random low->high idx DAG (topological by construction)."""
    nodes = [DNode(idx=i, engine="alu", instr=None) for i in range(n)]
    for dst in range(n):
        for src in range(dst):
            if rng.random() < edge_p:
                w = 1 if rng.random() < 0.7 else 0
                nodes[dst].in_edges.append((src, w))
                nodes[src].out_edges.append((dst, w))
    return DAG.from_nodes(nodes)


def _dag_snapshot(dag: DAG):
    return (list(dag._raw), list(dag._war), list(dag._pending),
            list(dag._committed), set(dag._frontier))


def _assert_dag_matches(tc: unittest.TestCase, dag: DAG, snap):
    raw, war, pending, committed, frontier = snap
    tc.assertEqual(dag._raw, raw)
    tc.assertEqual(dag._war, war)
    tc.assertEqual(dag._pending, pending)
    tc.assertEqual(dag._committed, committed)
    tc.assertEqual(dag._frontier, frontier)


class TestDAGRollback(unittest.TestCase):
    def test_random_commit_advance_rollbacks(self):
        rng = random.Random(7)
        for trial in range(50):
            dag = _random_dag(rng, n=rng.randint(2, 40))
            snap = _dag_snapshot(dag)
            token = dag.checkpoint()
            # Random sequence of commits (from the frontier) and advances.
            for _ in range(rng.randint(1, 30)):
                action = rng.random()
                if action < 0.7 and dag._frontier:
                    dag.commit(rng.choice(sorted(dag._frontier)))
                else:
                    dag.advance()
            dag.rollback(token)
            _assert_dag_matches(self, dag, snap)

    def test_nested_tokens(self):
        """A mid-sequence token rolls back only its own suffix."""
        rng = random.Random(11)
        dag = _random_dag(rng, n=30)
        t0 = dag.checkpoint()
        for _ in range(5):
            if dag._frontier:
                dag.commit(rng.choice(sorted(dag._frontier)))
        mid = _dag_snapshot(dag)
        t1 = dag.checkpoint()
        for _ in range(5):
            if dag._frontier:
                dag.commit(rng.choice(sorted(dag._frontier)))
            dag.advance()
        dag.rollback(t1)
        _assert_dag_matches(self, dag, mid)
        dag.rollback(t0)
        self.assertFalse(any(dag._committed))

    def test_rollback_to_zero_closes_log(self):
        dag = _random_dag(random.Random(3), n=10)
        t = dag.checkpoint()
        dag.advance()
        dag.rollback(t)
        self.assertIsNone(dag._log)
        # Post-rollback mutations are unrecorded (no error, no growth).
        dag.advance()


def _alloc_snapshot(a: RegisterAllocator):
    return (dict(a.assigned), dict(a.remaining), deque(a.free_vec),
            a.free_scalar_count(), dict(a._names), dict(a._word_granule),
            {g: list(ws) for g, ws in a._granule_words.items()},
            dict(a._granule_free), dict(a._claim_of))


def _assert_alloc_matches(tc, a, snap):
    (assigned, remaining, free_vec, n_free_scalar, names,
     word_granule, granule_words, granule_free, claim_of) = snap
    tc.assertEqual(a.assigned, assigned)
    tc.assertEqual(a.remaining, remaining)
    tc.assertEqual(a.free_vec, free_vec)
    tc.assertEqual(a.free_scalar_count(), n_free_scalar)
    tc.assertEqual(a._names, names)
    tc.assertEqual(a._word_granule, word_granule)
    tc.assertEqual(a._granule_words, granule_words)
    tc.assertEqual(a._granule_free, granule_free)
    tc.assertEqual(a._claim_of, claim_of)


class TestAllocatorRollback(unittest.TestCase):
    def _make(self):
        tags = [Sym(f"v{i}", True) for i in range(12)] + \
               [Sym(f"s{i}") for i in range(6)]
        read_count = {t: 3 for t in tags}
        return RegisterAllocator(read_count), tags

    def test_random_op_rollbacks(self):
        rng = random.Random(19)
        for _ in range(50):
            a, tags = self._make()
            snap = _alloc_snapshot(a)
            token = a.checkpoint()
            for _ in range(rng.randint(1, 60)):
                t = rng.choice(tags)
                op = rng.random()
                if op < 0.45:
                    if a.can_write(t):
                        a.write(t)
                elif op < 0.85:
                    if t in a.assigned and a.remaining[t] > 0:
                        a.read(t)
                else:
                    a.unwrite(t)   # internally a no-op unless fresh
            a.rollback(token)
            _assert_alloc_matches(self, a, snap)

    def test_free_and_realloc_same_addr(self):
        """alloc -> free -> realloc of the same granule within one window
        must restore the debug name and pool order exactly."""
        a, tags = self._make()
        a.free_vec = deque([8, 16, 24])        # shrink the pool for the test
        t0, t1, t2, t3 = tags[:4]
        snap = _alloc_snapshot(a)
        token = a.checkpoint()
        addr0 = a.write(t0)                     # takes 8 (leftmost)
        self.assertEqual(addr0, 8)
        for _ in range(3):
            a.read(t0)                          # frees 8 (to the right end)
        self.assertEqual(a.write(t1), 16)
        self.assertEqual(a.write(t2), 24)
        self.assertEqual(a.write(t3), 8)        # granule 8 reused by t3
        a.rollback(token)
        _assert_alloc_matches(self, a, snap)

    def test_unwrite_undo(self):
        a, tags = self._make()
        snap = _alloc_snapshot(a)
        token = a.checkpoint()
        a.write(tags[0])
        a.unwrite(tags[0])                  # returns the granule
        a.rollback(token)
        _assert_alloc_matches(self, a, snap)


class TestPlacementRollback(unittest.TestCase):
    def test_field_restore(self):
        ps = [_Placement(_KIND_VEC_ELEM, 8, "valu") for _ in range(5)]
        token = _Placement.checkpoint()
        ps[0].lanes_done = 3
        ps[0].engine_choice = "alu"
        ps[4].lanes_done = 8
        ps[4].engine_choice = "valu"
        _Placement.rollback(token)
        for p in ps:
            self.assertEqual(p.lanes_done, 0)
            self.assertIsNone(p.engine_choice)
        self.assertIsNone(_Placement._log)

    def test_partial_token(self):
        p = _Placement(_KIND_ATOMIC_SCALAR, 1, "alu")
        _Placement.checkpoint()
        p.lanes_done = 1
        t = _Placement.checkpoint()
        p.engine_choice = "alu"
        _Placement.rollback(t)
        self.assertEqual(p.lanes_done, 1)        # kept: before the token
        self.assertIsNone(p.engine_choice)       # undone: after the token
        _Placement.rollback(0)
        self.assertEqual(p.lanes_done, 0)


if __name__ == "__main__":
    unittest.main()
