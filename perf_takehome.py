"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False,
              seed: int | None = None, picker: str = "fma_first"):
        """Convert a slot list into instruction bundles.

        vliw=False: one slot per bundle (the original sequential packing).
        vliw=True:  DAG-driven VLIW scheduler - builds a dependency DAG from
                    the slots and packs multiple independent slots per cycle
                    respecting per-engine slot limits and read-before-write.
                    picker selects node ordering ("fma_first", "idx", "random").
        """
        if not vliw:
            return [{engine: [slot]} for engine, slot in slots]
        from scheduler import DAG, schedule
        dag = DAG(slots)
        cap = len(slots)  # worst case: 1 slot/cycle
        return schedule(dag, seed=seed, cap=cap, picker=picker)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_vec_hash(self, val_vec, t1_vec, t2_vec, r, base_i,
                        fma_vec_consts, irr_vec_consts):
        """Emit the 12-slot reduced myhash fully on the `valu` unit, operating
        elementwise on all VLEN=8 lanes of `val_vec` in parallel.

        `val_vec` is both the input `a = val ^ node_val` and the persistent
        lane-state vector: each stage writes back into it, and the final
        stage leaves the new `val` there (no copy-out needed).
        t1_vec / t2_vec : 8-word stage-scratch vectors. Live only within a
            single irreducible xor/add-shift stage (2 parallel transforms +
            `^` combine), dead between stages. After the hash they are reused
            for the branchless idx update (they're dead post-final-combine).
        fma_vec_consts : {value: addr} of broadcast vectors for the 3 linear
            stages (0/2/4): keys are the multiplier (1+2^s) and the addend K.
        irr_vec_consts : {value: addr} of broadcast vectors for the 3
            irreducible stages (1/3/5): keys are K (val1) and the shift amount
            (val3). Kept separate from fma_vec_consts so the literal `9` does
            not collide (mult 9 in stage 4 vs shift 9 in stage 3).
        """
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+":
                # Linear stage: (a + K) + (a << s) == a*(1+2^s) + K, one fma.
                mult = (1 << val3) + 1
                slots.append(("valu", ("multiply_add", val_vec, val_vec,
                                       fma_vec_consts[mult], fma_vec_consts[val1])))
            else:
                # Irreducible xor/add-shift stage: 2 parallel elementwise
                # transforms of val_vec, then a `^` (or `+`) combine.
                slots.append(("valu", (op1, t1_vec, val_vec, irr_vec_consts[val1])))
                slots.append(("valu", (op3, t2_vec, val_vec, irr_vec_consts[val3])))
                slots.append(("valu", (op2, val_vec, t1_vec, t2_vec)))
            keys = [(r, base_i + j, "hash_stage", hi) for j in range(VLEN)]
            slots.append(("debug", ("vcompare", val_vec, keys)))
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """v3a: 8-lane-per-group kernel. Cross-lane vectorization over the
        `valu` unit (VLEN=8 lanes per group, 32 groups), still one slot per
        bundle (vliw=False) -- sophisticated VLIW packing is deliberately
        postponed; this pass is the functional 8-lane-vectorization chunk.

        Layout (all pipeline state resident in scratch across all rounds;
        see notes/scratch_map_canonical.md for the rationale):
          - val[256] : plane 0 of the per-lane sector; carried across rounds
            AND the running hash register during each round's hash (entry XOR
            writes `a` into it; the final hash stage leaves `v` there for the
            next round -- zero copy-out ops). Stored as 32 contiguous VLEN=8
            vectors.
          - idx[256] : plane 1; zero-initialized (scratch clears to 0, initial
            idx is 0 -> no init).
          - consts : the 6 hash constants + shift amounts + idx helper
            constants broadcast once at prologue as VLEN=8 vectors and reused
            across all 32 groups x 16 rounds of hashes.

        Each group's 8 per-lane scratch words live in per-lane SoA planes
        (see layout below): val/idx persistent state + t1/t2 per-lane stage
        scratch + a per-lane node_val landing plane. Per-lane t1/t2/nv means
        distinct groups' in-flight stages never alias -- VLIW-packable without
        rename management.

        Gather (non-contiguous tree.values[idx[i]]) is the one operation that
        cannot be vectorized (the ISA has no scatter/gather): for each group of
        8 lanes we compute all 8 gather addresses in one `valu` `+` (idx plus
        the broadcast forest pointer), then issue 8 scalar `load`s landing in
        the per-lane nv plane. This is the "naive loader" stage -- prefetch /
        dual-port packing / speculative both-branch loads are deferred. With
        one slot per bundle the gather is 1 (addr) + 8 (loads) = 9 cycles, then
        entry XOR (1) + 12-slot hash + idx (3, or 1 on the uniform wrap round).

        Branchless idx update: parity = v & 1; base = idx*2 + 1 (valu fma);
        next = base + parity (valu `+`). Equals the reference's
        `2*idx + (1 if even else 2)` bit-exactly (parity=0 -> +1, parity=1 -> +2).

        Wrap is a build-time-known per-round decision (verified uniform wrap
        on round=height for the canonical shape): on that round we skip the
        branchless idx update and write idx := 0 for all lanes (one `valu`
        `& idx,zero`).

        Canonical shape assumed: forest_height=10, n_nodes=2047, batch_size=256.
        """
        assert batch_size == 256, f"v3 supports only batch_size=256 (got {batch_size})"
        assert batch_size % VLEN == 0, "batch_size must be a multiple of VLEN=8"
        assert forest_height == 10, (
            f"v3 hardcodes wrap round tied to height 10 (got {forest_height})")
        assert n_nodes == (2 ** (forest_height + 1) - 1), "n_nodes / height mismatch"

        V = batch_size
        n_groups = V // VLEN
        WRAP_ROUND = forest_height   # verified: all lanes at leaf on round=h -> wrap to root

        # =====================================================================
        # Scratch layout: planes first (8-aligned), shared sector last.
        #
        #   [   0.. 255]  plane 0 : val[256]   (running hash + carried state)
        #   [ 256.. 511]  plane 1 : idx[256]   (tree index, zero-init)
        #   [ 512.. 767]  plane 2 : t1[256]    (per-lane stage scratch)
        #   [ 768..1023]  plane 3 : t2[256]    (per-lane stage scratch)
        #   [1024..1279]  plane 4 : nv[256]    (node_val landing / spare)
        #   [1280..1535]  shared sector (256 words):
        #                   8-word vector regions first  (forest_p_vec, addr_vec,
        #                     12 hash-const vecs, two/one/zero_vec = 17 vecs)
        #                   then 1-word scalars (header 7, zero/eight/addr_a,
        #                     _src scalars for broadcasts)
        #
        # Planes-first guarantees every val_vec = base + 8g is 8-aligned so a
        # vector write covers exactly one region — essential for the DAG builder's
        # region-keyed last_writer/readers_since bookkeeping.
        # =====================================================================

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]

        # ---- Phase 1: planes (5 × V=256 = 1280 words, 8-aligned) ----
        val_base = self.alloc_scratch("val", V)   # plane 0
        idx_base = self.alloc_scratch("idx", V)   # plane 1
        t1_base  = self.alloc_scratch("t1",  V)   # plane 2
        t2_base  = self.alloc_scratch("t2",  V)   # plane 3
        nv_base  = self.alloc_scratch("nv",  V)   # plane 4

        # ---- Phase 2: 8-word vector regions (shared sector, 8-aligned) ----
        # No instructions emitted yet — broadcasts happen in prologue.
        addr_vec     = self.alloc_scratch("addr_vec", VLEN)
        forest_p_vec = self.alloc_scratch("forest_p_vec", VLEN)
        mult4097_vec = self.alloc_scratch("mult4097_vec", VLEN)
        K0_vec       = self.alloc_scratch("K0_vec", VLEN)
        mult33_vec   = self.alloc_scratch("mult33_vec", VLEN)
        K2_vec       = self.alloc_scratch("K2_vec", VLEN)
        mult9_vec    = self.alloc_scratch("mult9_vec", VLEN)
        K4_vec       = self.alloc_scratch("K4_vec", VLEN)
        K1_vec       = self.alloc_scratch("K1_vec", VLEN)
        K3_vec       = self.alloc_scratch("K3_vec", VLEN)
        K5_vec       = self.alloc_scratch("K5_vec", VLEN)
        shift19_vec  = self.alloc_scratch("shift19_vec", VLEN)
        shift9_vec   = self.alloc_scratch("shift9_vec", VLEN)
        shift16_vec  = self.alloc_scratch("shift16_vec", VLEN)
        two_vec      = self.alloc_scratch("two_vec", VLEN)
        one_vec     = self.alloc_scratch("one_vec", VLEN)
        zero_vec     = self.alloc_scratch("zero_vec", VLEN)

        # Collect (vec_addr, value) pairs for deferred broadcasts (prologue).
        vec_bcasts = [
            (mult4097_vec, 4097),        (K0_vec, 0x7ED55D16),
            (mult33_vec,  33),            (K2_vec, 0x165667B1),
            (mult9_vec,   9),             (K4_vec, 0xFD7046C5),
            (K1_vec,      0xC761C23C),   (K3_vec, 0xD3A2646C),
            (K5_vec,      0xB55A4F09),
            (shift19_vec, 19),            (shift9_vec, 9),
            (shift16_vec, 16),
            (two_vec,     2),             (one_vec,  1),
            (zero_vec,    0),
        ]

        # ---- Phase 3: 1-word scalars (after all 8-word vec regions) ----
        # Header vars (loaded from mem, not const — just reserve space).
        for v in init_vars:
            self.alloc_scratch(v, 1)
        # Literal constants (scratch_const emits `load const`).
        zero_const   = self.scratch_const(0, "zero")
        eight_const  = self.scratch_const(8, "eight")
        three_const  = self.scratch_const(3, "three")  # level-2 base offset for select
        addr_a       = self.alloc_scratch("addr_a", 1)   # vload/vstore pointer

        # fma / irr split so literal `9` (mult in stage 4 vs shift in stage 3)
        # doesn't collide as a dict key.
        fma_vec_consts = {
            4097: mult4097_vec, 0x7ED55D16: K0_vec,
            33:   mult33_vec,   0x165667B1: K2_vec,
            9:    mult9_vec,    0xFD7046C5: K4_vec,
        }
        irr_vec_consts = {
            0xC761C23C: K1_vec, 0xD3A2646C: K3_vec, 0xB55A4F09: K5_vec,
            19: shift19_vec, 9: shift9_vec, 16: shift16_vec,
        }

        assert self.scratch_ptr <= SCRATCH_SIZE, "scratch overflow"

        # ---- Phase 3b: tree preload for levels 0-2 (gather dedup) ----
        # Single vload reads mem[forest_values_p + 0..7] (7 nodes = levels 0-2,
        # plus 1 bonus). Then 7 vbroadcasts create shared vector constants
        # tree0_vec..tree6_vec so any group can select its node_val without
        # gathering from mem for rounds 0-2.
        tree_preload = self.alloc_scratch("tree_preload", VLEN)  # 8 words: tree[0..7]
        tree0_vec = self.alloc_scratch("tree0_vec", VLEN)
        tree1_vec = self.alloc_scratch("tree1_vec", VLEN)
        tree2_vec = self.alloc_scratch("tree2_vec", VLEN)
        tree3_vec = self.alloc_scratch("tree3_vec", VLEN)
        tree4_vec = self.alloc_scratch("tree4_vec", VLEN)
        tree5_vec = self.alloc_scratch("tree5_vec", VLEN)
        tree6_vec = self.alloc_scratch("tree6_vec", VLEN)
        # Temp vectors for vselect intermediates (shared, reused across groups).
        sel_lo_vec = self.alloc_scratch("sel_lo_vec", VLEN)
        sel_hi_vec = self.alloc_scratch("sel_hi_vec", VLEN)
        tree_vecs = [tree0_vec, tree1_vec, tree2_vec,
                     tree3_vec, tree4_vec, tree5_vec, tree6_vec]
        three_vec = self.alloc_scratch("three_vec", VLEN)  # broadcast of 3 for level-2 select

        assert self.scratch_ptr <= SCRATCH_SIZE, "scratch overflow (tree preload)"

        # =====================================================================
        # Prologue: load header; vload val[256]; broadcast consts; pause
        # =====================================================================
        for i, v in enumerate(init_vars):
            self.add("load", ("const", addr_a, i))                  # addr_a := i
            self.add("load", ("load",  self.scratch[v], addr_a))     # scratch[v] := mem[i]

        # vload val[256] as 32 vectors of 8 contiguous words from mem[inp_values_p..].
        self.add("alu", ("+", addr_a, self.scratch["inp_values_p"], zero_const))
        for k in range(n_groups):
            self.add("load", ("vload", val_base + k * VLEN, addr_a))
            if k < n_groups - 1:
                self.add("alu", ("+", addr_a, addr_a, eight_const))

        # Broadcast forest_values_p (from a header var, not a literal).
        self.add("valu", ("vbroadcast", forest_p_vec, self.scratch["forest_values_p"]))

        # Broadcast all hash-const / idx-helper vectors from literal scratch_consts.
        for vec_addr, value in vec_bcasts:
            s = self.scratch_const(value)         # deduped by value; emits load const
            self.add("valu", ("vbroadcast", vec_addr, s))

        # vload tree[0..7] (levels 0-2 = 7 nodes + 1 bonus) into tree_preload.
        self.add("alu", ("+", addr_a, self.scratch["forest_values_p"], zero_const))
        self.add("load", ("vload", tree_preload, addr_a))
        # Broadcast tree[0..6] into shared vector constants.
        for i in range(7):
            self.add("valu", ("vbroadcast", tree_vecs[i], tree_preload + i))
        self.add("valu", ("vbroadcast", three_vec, three_const))

        # Pause 1 -- match reference_kernel2's first yield (initial mem).
        self.add("flow", ("pause",))

        # =====================================================================
        # Body -- unrolled rounds x 32 groups, one slot per bundle.
        # =====================================================================
        body = []
        for r in range(rounds):
            is_wrap = (r == WRAP_ROUND)
            for g in range(n_groups):
                # per-group vectors into the SoA per-lane planes (8 contiguous words each)
                val_vec = val_base + g * VLEN
                idx_vec = idx_base + g * VLEN
                t1_g    = t1_base  + g * VLEN
                t2_g    = t2_base  + g * VLEN
                nv_g    = nv_base  + g * VLEN
                base_i  = g * VLEN
                keyval  = [(r, base_i + j, "val") for j in range(VLEN)]
                keynv   = [(r, base_i + j, "node_val") for j in range(VLEN)]
                keyhv   = [(r, base_i + j, "hashed_val") for j in range(VLEN)]
                keywr   = [(r, base_i + j, "wrapped_idx") for j in range(VLEN)]

                # --- node_val gather or preload-select (rounds 0-2 use preloaded) ---
                if r in (0, 11):
                    # Level 0: all lanes at idx=0. node_val = tree[0].
                    body.append(("valu", ("^", nv_g, tree0_vec, zero_vec)))
                elif r in (1, 12):
                    # Level 1: idx in {1,2}. 1 vselect on idx bit 0.
                    # idx=1 (bit0=1) -> tree[1]; idx=2 (bit0=0) -> tree[2].
                    body.append(("valu", ("&", t1_g, idx_vec, one_vec)))   # cond = idx & 1
                    body.append(("flow", ("vselect", nv_g, t1_g, tree1_vec, tree2_vec)))
                elif r in (2, 13):
                    # Level 2: idx in {3,4,5,6}. Subtract level base (3), then
                    # 2-level select on bits 0-1 of (idx-3).
                    # idx-3=0->tree3, 1->tree4, 2->tree5, 3->tree6
                    body.append(("valu", ("-", t1_g, idx_vec, three_vec)))   # idx - 3
                    body.append(("valu", ("&", t2_g, t1_g, one_vec)))        # bit 0 of (idx-3)
                    body.append(("flow", ("vselect", sel_lo_vec, t2_g, tree4_vec, tree3_vec)))
                    body.append(("flow", ("vselect", sel_hi_vec, t2_g, tree6_vec, tree5_vec)))
                    body.append(("valu", (">>", t2_g, t1_g, one_vec)))       # (idx-3) >> 1
                    body.append(("valu", ("&", t2_g, t2_g, one_vec)))        # bit 1 of (idx-3)
                    body.append(("flow", ("vselect", nv_g, t2_g, sel_hi_vec, sel_lo_vec)))
                else:
                    # Rounds 3+: gather from mem (naive scalar loads).
                    body.append(("valu", ("+", addr_vec, idx_vec, forest_p_vec)))
                    for j in range(VLEN):
                        body.append(("load", ("load", nv_g + j, addr_vec + j)))

                body.append(("debug", ("vcompare", nv_g, keynv)))
                body.append(("debug", ("vcompare", val_vec, keyval)))  # val before xor

                # --- entry XOR: val_vec = val_vec ^ nv_g  (a) ---
                body.append(("valu", ("^", val_vec, val_vec, nv_g)))

                # --- 12-slot hash, fully on valu (8 lanes / slot) ---
                body.extend(self.build_vec_hash(val_vec, t1_g, t2_g, r, base_i,
                                                fma_vec_consts, irr_vec_consts))

                # debug: hashed_val == v == val_vec after hash
                body.append(("debug", ("vcompare", val_vec, keyhv)))

                # --- post-hash: idx update or wrap (branchless, on valu) ---
                if is_wrap:
                    # Verified: on this round, all 256 lanes are at leaf,
                    # so next = 2*idx + 1 + parity >= n_nodes -> wrap to 0.
                    body.append(("valu", ("&", idx_vec, idx_vec, zero_vec)))
                else:
                    body.append(("valu", ("&", t1_g, val_vec, one_vec)))        # parity = v & 1
                    body.append(("valu", ("multiply_add", t2_g,
                                           idx_vec, two_vec, one_vec)))         # base = 2*idx + 1
                    body.append(("valu", ("+", idx_vec, t2_g, t1_g)))          # next = base + parity
                body.append(("debug", ("vcompare", idx_vec, keywr)))

        body_instrs = self.build(body, vliw=True, seed=42)
        self.instrs.extend(body_instrs)

        # =====================================================================
        # Epilogue: vstore val[256] back to mem[inp_values_p .. +256]
        # =====================================================================
        self.add("alu", ("+", addr_a, self.scratch["inp_values_p"], zero_const))
        for k in range(n_groups):
            self.add("store", ("vstore", addr_a, val_base + k * VLEN))
            if k < n_groups - 1:
                self.add("alu", ("+", addr_a, addr_a, eight_const))

        # Pause 2 -- match reference_kernel2's final yield (final mem).
        # Must come AFTER the epilogue so machine.mem holds the final values
        # when the test recommends execution on i=1 (final yield).
        self.add("flow", ("pause",))

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
