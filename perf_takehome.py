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

from ir import (
    Sym, RenameEngine, Alu, VecElem, VecFma, VBroadcast, Load, VLoad, Const,
    VStore, VSelect, Pause, DebugVCompare,
)
from scheduler import Weights
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
        self.scratch = {}            # name -> Sym (scalar header vars)
        self.scratch_ptr = 0
        self.const_map = {}
        # Rename engine: every symbol is pinned to the hand-laid-out address
        # (translation only; dynamic allocation is a future extension).
        self.re = RenameEngine()

    def debug_info(self):
        return DebugInfo(scratch_map=self.re.debug_map())

    def build(self, slots: list, vliw: bool = False,
              seed: int | None = None, picker: str = "fma_first",
              weights=None):
        """Convert an IR instruction list into instruction bundles.

        vliw=False: one slot per bundle (the original sequential packing).
        vliw=True:  DAG-driven VLIW scheduler - builds a dependency DAG from
                    the instructions and packs multiple independent slots per
                    cycle respecting per-engine slot limits and
                    read-before-write. picker selects node ordering
                    ("fma_first", "idx", "random", "weighted").
        """
        if not vliw:
            return [{s.engine: [s.lower()]} for s in slots]
        from scheduler import DAG, schedule, prune_to_stores
        dag = DAG(slots)
        pruned = prune_to_stores(dag)
        if len(pruned) != len(dag):
            print(f"prune_to_stores: {len(dag)} -> {len(pruned)} nodes "
                  f"({len(dag) - len(pruned)} dead)")
        dag = pruned
        cap = len(slots)  # worst case: 1 slot/cycle
        return schedule(dag, seed=seed, cap=cap, picker=picker, weights=weights)

    def add(self, instr):
        """Append a single symbolic IR instruction as a one-slot bundle
        (linear code: prologue/epilogue). Resolved + lowered immediately -
        these never see the DAG."""
        instr = self.re.resolve_instr(instr)
        self.instrs.append({instr.engine: [instr.lower()]})

    def alloc_scratch(self, name=None, length=1, is_vec=False):
        """Declare a symbol pinned at the next scratch address (sequential
        layout - same allocation order as the hand-pinned address map)."""
        addr = self.scratch_ptr
        sym = self.re.pin(Sym(name or f"@{addr}", is_vec), addr)
        if name is not None:
            self.scratch[name] = sym
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return sym

    def alloc_plane(self, name, n):
        """Declare n vector symbols pinned as one contiguous plane
        (n*VLEN words, 8-aligned): sym[g] at base + g*VLEN - group g's
        vector of a per-lane SoA plane. Returns the symbol list."""
        base = self.scratch_ptr
        syms = [self.re.pin(Sym(f"{name}[{g}]", True), base + g * VLEN)
                for g in range(n)]
        self.scratch_ptr += n * VLEN
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return syms

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            sym = self.alloc_scratch(name)
            self.add(Const(sym, val))
            self.const_map[val] = sym
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
                slots.append(VecFma(val_vec, val_vec,
                                    fma_vec_consts[mult],
                                    fma_vec_consts[val1]))
            else:
                # Irreducible xor/add-shift stage: 2 parallel elementwise
                # transforms of val_vec, then a `^` (or `+`) combine.
                slots.append(VecElem(op1, t1_vec, val_vec, irr_vec_consts[val1]))
                slots.append(VecElem(op3, t2_vec, val_vec, irr_vec_consts[val3]))
                slots.append(VecElem(op2, val_vec, t1_vec, t2_vec))
            keys = [(r, base_i + j, "hash_stage", hi) for j in range(VLEN)]
            slots.append(DebugVCompare(val_vec, keys))
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
        # Scratch layout: planes, then vector section (CONST, VAR), then scalar
        # section (CONST, VAR). All 8-word vectors follow the 5x256 plane sector
        # so every vector is 8-aligned (one vector == one region); scalars are
        # packed after.
        #
        #   [   0..1279]  per-lane planes: val, idx, t1, t2, nv (5 x 256)
        #   [1280.. ...]  vector section (8 words each):
        #                   CONST vecs: mult4097/K0/mult33/K2/mult9/K4/K1/K3/
        #                     K5/shift19/shift9/shift16/two/one/zero/three (16)
        #                   VAR vecs: forest_p/tree_preload/tree0..tree6 (9)
        #   [ ... .. ...]  scalar section (1 word each):
        #                   CONST scalars: zero/eight/three + 13 bcast sources (16)
        #                   VAR scalars: 7 header vars + addr_a (8)
        #   [ ... ..1535]  free (32 words)
        #
        # Planes-first guarantees every val_vec = base + 8g is 8-aligned so a
        # vector write covers exactly one region - essential for the DAG
        # builder's region-keyed last_writer/readers_since bookkeeping.
        # No addr_vec/sel_lo_vec/sel_hi_vec: those short-lived temporaries reuse
        # per-group planes (nv_g for gather addr via self-addressing loads; t2_g
        # for the level-2 select intermediate) -> no cross-group WAR chains.
        # =====================================================================

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]

        # ---- Phase 1: planes (5 × V=256 = 1280 words, 8-aligned) ----
        # Per-lane SoA planes as 32 group-vector symbols each: val[g] is
        # lanes 8g..8g+7 of the plane, pinned at plane_base + 8g.
        val  = self.alloc_plane("val",  n_groups)  # running hash + carried state
        addr = self.alloc_plane("addr", n_groups)  # tree ADDRESS = idx + forest_p (stored, not idx)
        t1   = self.alloc_plane("t1",   n_groups)  # per-lane stage scratch
        t2   = self.alloc_plane("t2",   n_groups)  # per-lane stage scratch
        nv   = self.alloc_plane("nv",   n_groups)  # node_val landing / spare

        # ---- Phase 2: vector section (8-word regions, CONST first then VAR) ----
        # No addr_vec/sel_lo_vec/sel_hi_vec: those short-lived temporaries reuse
        # per-group planes (nv_g for the gather address via self-addressing
        # loads; t2_g for the level-2 select intermediate) so no register is
        # written by more than one group -> no cross-group WAR dependency chains.
        # CONST vectors: uniform value*8. Small reusable ones are named
        # const_vec_<value> and sorted by value (not tied to a step - e.g. 9 is
        # both the stage-4 multiplier and the stage-3 shift). K0..K5 are the
        # stage-specific hash addend/xor constants. No separate broadcast-source
        # scalars: each is created by `load const` into its own lane 0, then
        # self-broadcast (vbroadcast vec, vec) - see the prologue. Scalar uses of
        # a value read the matching const_vec's lane 0.
        const_vec_0    = self.alloc_scratch("const_vec_0", VLEN, is_vec=True)
        const_vec_1    = self.alloc_scratch("const_vec_1", VLEN, is_vec=True)
        const_vec_2    = self.alloc_scratch("const_vec_2", VLEN, is_vec=True)
        const_vec_3    = self.alloc_scratch("const_vec_3", VLEN, is_vec=True)
        const_vec_9    = self.alloc_scratch("const_vec_9", VLEN, is_vec=True)     # stage4 mult + stage3 shift
        const_vec_16   = self.alloc_scratch("const_vec_16", VLEN, is_vec=True)   # stage5 shift
        const_vec_19   = self.alloc_scratch("const_vec_19", VLEN, is_vec=True)   # stage1 shift
        const_vec_33   = self.alloc_scratch("const_vec_33", VLEN, is_vec=True)   # stage2 mult
        const_vec_4097 = self.alloc_scratch("const_vec_4097", VLEN, is_vec=True) # stage0 mult
        K0_vec = self.alloc_scratch("K0_vec", VLEN, is_vec=True)   # stage 0 addend
        K1_vec = self.alloc_scratch("K1_vec", VLEN, is_vec=True)   # stage 1 xor const
        K2_vec = self.alloc_scratch("K2_vec", VLEN, is_vec=True)   # stage 2 addend
        K3_vec = self.alloc_scratch("K3_vec", VLEN, is_vec=True)   # stage 3 add const
        K4_vec = self.alloc_scratch("K4_vec", VLEN, is_vec=True)   # stage 4 addend
        K5_vec = self.alloc_scratch("K5_vec", VLEN, is_vec=True)   # stage 5 xor const
        # VAR vectors: runtime values (forest_p = header broadcast; tree_preload
        # = non-uniform vload of tree[0..7]; tree0..6 = its lane broadcasts).
        forest_p_vec = self.alloc_scratch("forest_p_vec", VLEN, is_vec=True)
        neg_fp1_vec  = self.alloc_scratch("neg_fp1_vec", VLEN, is_vec=True)  # 1 - forest_p (next-addr: 2*addr + neg_fp1 + parity)
        pos_fp5_vec  = self.alloc_scratch("pos_fp5_vec", VLEN, is_vec=True)  # 5 + forest_p
        tree_preload = self.alloc_scratch("tree_preload", VLEN, is_vec=True)  # 8 words: tree[0..7]
        tree0_vec = self.alloc_scratch("tree0_vec", VLEN, is_vec=True)
        tree1_vec = self.alloc_scratch("tree1_vec", VLEN, is_vec=True)
        tree2_vec = self.alloc_scratch("tree2_vec", VLEN, is_vec=True)
        tree3_vec = self.alloc_scratch("tree3_vec", VLEN, is_vec=True)
        tree4_vec = self.alloc_scratch("tree4_vec", VLEN, is_vec=True)
        tree5_vec = self.alloc_scratch("tree5_vec", VLEN, is_vec=True)
        tree6_vec = self.alloc_scratch("tree6_vec", VLEN, is_vec=True)
        tree_vecs = [tree0_vec, tree1_vec, tree2_vec,
                     tree3_vec, tree4_vec, tree5_vec, tree6_vec]

        # (vec, literal) pairs: the prologue `load const` lane 0 + self-broadcast.
        vec_bcasts = [
            (const_vec_0, 0),    (const_vec_1, 1),    (const_vec_2, 2),
            (const_vec_3, 3),    (const_vec_9, 9),    (const_vec_16, 16),
            (const_vec_19, 19),  (const_vec_33, 33),  (const_vec_4097, 4097),
            (K0_vec, 0x7ED55D16), (K1_vec, 0xC761C23C), (K2_vec, 0x165667B1),
            (K3_vec, 0xD3A2646C), (K4_vec, 0xFD7046C5), (K5_vec, 0xB55A4F09),
        ]

        # Hash-stage consts by value. The literal `9` is shared (stage-4 mult +
        # stage-3 shift both read const_vec_9); kept in two dicts only because
        # fma stages read (mult, addend) and irr stages read (K, shift).
        fma_vec_consts = {
            4097: const_vec_4097, 0x7ED55D16: K0_vec,
            33:   const_vec_33,   0x165667B1: K2_vec,
            9:    const_vec_9,    0xFD7046C5: K4_vec,
        }
        irr_vec_consts = {
            0xC761C23C: K1_vec, 0xD3A2646C: K3_vec, 0xB55A4F09: K5_vec,
            19: const_vec_19, 9: const_vec_9, 16: const_vec_16,
        }

        # ---- Phase 3: scalar section (1 word each, CONST first then VAR) ----
        # No per-vector broadcast-source scalars (self-broadcast from lane 0).
        # The only CONST scalar is `eight` (vload/vstore stride of 8 - no matching
        # const_vec_8 since 8 is never used as a vector). Scalar uses of 0 read
        # const_vec_0's lane 0 (see prologue/epilogue).
        eight_const = self.scratch_const(8, "eight")     # vload/vstore stride
        # VAR scalars: header vars (loaded from mem) + addr_a (vload/vstore ptr).
        for v in init_vars:
            self.alloc_scratch(v, 1)
        addr_a = self.alloc_scratch("addr_a", 1)
        # Per-group output addresses for the epilogue-overlapping vstores: each
        # is inp_values_p + 8g (runtime base + compile-time offset). The offset
        # is `load const`-ed and inp_values_p added in the body (independent per
        # group - no addr_a chain), so the vstores can issue 2/cyc as soon as
        # each group's round-15 val is ready.
        out_addr = [self.re.pin(Sym(f"out_addr[{g}]"), self.scratch_ptr + g)
                    for g in range(n_groups)]
        self.scratch_ptr += n_groups

        assert self.scratch_ptr <= SCRATCH_SIZE, "scratch overflow"

        # =====================================================================
        # Prologue: load header; vload val[256]; broadcast consts; pause
        # =====================================================================
        for i, v in enumerate(init_vars):
            self.add(Const(addr_a, i))                              # addr_a := i
            self.add(Load(self.scratch[v], addr_a))                 # scratch[v] := mem[i]

        # vload val[256] as 32 vectors of 8 contiguous words from mem[inp_values_p..].
        self.add(Alu("+", addr_a, self.scratch["inp_values_p"],
                     const_vec_0.lane(0)))
        for k in range(n_groups):
            self.add(VLoad(val[k], addr_a))
            if k < n_groups - 1:
                self.add(Alu("+", addr_a, addr_a, eight_const))

        # Broadcast forest_values_p (from a header var, not a literal).
        self.add(VBroadcast(forest_p_vec, self.scratch["forest_values_p"]))

        # Create each const vector by `load const` into its own lane 0 then
        # self-broadcast (vbroadcast vec, vec reads lane 0, writes all 8). No
        # separate broadcast-source scalar needed.
        for vec_sym, value in vec_bcasts:
            self.add(Const(vec_sym.lane(0), value))
            self.add(VBroadcast(vec_sym, vec_sym.lane(0)))

        # neg_fp1 = 1 - forest_values_p (used by the next-addr update). Computed
        self.add(VecElem("-", neg_fp1_vec, const_vec_1, forest_p_vec))
        # pos_fp5 = 5 + forest_values_p (used by the level 2 select). Computed
        self.add(VecElem("+", pos_fp5_vec, const_vec_2, const_vec_3))  # pos_fp5 = 5
        self.add(VecElem("+", pos_fp5_vec, pos_fp5_vec, forest_p_vec))

        # vload tree[0..7] (levels 0-2 = 7 nodes + 1 bonus) into tree_preload.
        self.add(Alu("+", addr_a, self.scratch["forest_values_p"],
                     const_vec_0.lane(0)))
        self.add(VLoad(tree_preload, addr_a))
        # Broadcast tree[0..6] into shared vector constants.
        for i in range(7):
            self.add(VBroadcast(tree_vecs[i], tree_preload.lane(i)))

        # Pause 1 -- match reference_kernel2's first yield (initial mem).
        self.add(Pause())

        # =====================================================================
        # Body -- unrolled rounds x 32 groups, one slot per bundle.
        # =====================================================================
        body = []
        # Output addresses: load each group's compile-time offset (8g) as a
        # const, then add the runtime inp_values_p. Independent per group (no
        # addr_a chain) so the round-15 vstores can fire 2/cyc in any order.
        # Scheduled early; ready well before the vstores need them.
        inp_values_p = self.scratch["inp_values_p"]
        for g in range(n_groups):
            body.append(Const(out_addr[g], g * VLEN))   # offset 8g
            body.append(Alu("+", out_addr[g], out_addr[g], inp_values_p))
        for r in range(rounds):
            for g in range(n_groups):
                is_wrap = (r == WRAP_ROUND)
                # per-group vector symbols of the SoA per-lane planes
                val_vec  = val[g]
                addr_vec = addr[g]
                t1_g     = t1[g]
                t2_g     = t2[g]
                nv_g     = nv[g]
                base_i   = g * VLEN
                keyval  = [(r, base_i + j, "val") for j in range(VLEN)]
                keynv   = [(r, base_i + j, "node_val") for j in range(VLEN)]
                keyhv   = [(r, base_i + j, "hashed_val") for j in range(VLEN)]

                # --- node_val gather or preload-select (rounds 0-2 use preloaded) ---
                if r in (0, 11):
                    # Level 0: all lanes at idx=0. node_val = tree[0].
                    body.append(VecElem("^", nv_g, tree0_vec, const_vec_0))
                elif r in (1, 12):
                    # Level 1: idx in {1,2}. Recover idx = addr - forest_p,
                    # then 1 vselect on idx bit 0 (idx=1 -> tree1, idx=2 -> tree2).
                    # parity from last round was in t1_g
                    body.append(VSelect(nv_g, t1_g, tree2_vec, tree1_vec))
                elif r in (2, 13):
                    # Level 2: idx in {3,4,5,6}. Recover idx, subtract level
                    # base (3), then 2-level select on bits 0-1 of (idx-3).
                    #   idx-3=0->tree3, 1->tree4, 2->tree5, 3->tree6

                    body.append(VSelect(nv_g, t1_g, tree4_vec, tree3_vec))  # bit0?tree4:tree3
                    body.append(VSelect(t2_g, t1_g, tree6_vec, tree5_vec))  # bit0?tree6:tree5
                    body.append(VecElem("<", t1_g, addr_vec, pos_fp5_vec))   # low?
                    body.append(VSelect(nv_g, t1_g, nv_g, t2_g))  # low?nv:t2
                else:
                    # Rounds 3+: gather from mem. addr_vec already holds the
                    # tree address (idx + forest_p), so the loads read it
                    # directly - no per-round address-add valu. Self-addressing:
                    # each load reads addr_g+j as the address and writes nv_g+j
                    # as the value. nv_g is then read by the entry XOR below.
                    for j in range(VLEN):
                        body.append(Load(nv_g.lane(j), addr_vec.lane(j)))

                body.append(DebugVCompare(nv_g, keynv))
                body.append(DebugVCompare(val_vec, keyval))  # val before xor

                # --- entry XOR: val_vec = val_vec ^ nv_g  (a) ---
                body.append(VecElem("^", val_vec, val_vec, nv_g))

                # --- 12-slot hash, fully on valu (8 lanes / slot) ---
                body.extend(self.build_vec_hash(val_vec, t1_g, t2_g, r, base_i,
                                                fma_vec_consts, irr_vec_consts))

                # debug: hashed_val == v == val_vec after hash
                body.append(DebugVCompare(val_vec, keyhv))

                # --- post-hash: idx update or wrap (branchless, on valu) ---
                # --- post-hash: addr update (store addr = idx + forest_p, not
                # idx; gather reads addr directly). next_addr = 2*addr +
                # (1-forest_p) + parity = 2*addr + neg_fp1 + parity. Wrap sets
                # addr = forest_p. Round 0 (idx=0 initial) computes next_addr
                # = forest_p+1+parity = (2-neg_fp1)+parity without reading addr
                # (addr plane is not yet valid). ---
                if is_wrap:
                    # idx -> 0, so addr = forest_p = 1 - neg_fp1.
                    body.append(VecElem("-", addr_vec, const_vec_1, neg_fp1_vec))
                else:
                    body.append(VecElem("&", t1_g, val_vec, const_vec_1))        # parity = v & 1
                    if r == 0:
                        # idx=0: next_addr = forest_p + 1 + parity = (2 - neg_fp1) + parity
                        body.append(VecElem("-", t2_g, const_vec_2, neg_fp1_vec))  # 2 - neg_fp1
                    else:
                        # next_addr base = 2*addr + neg_fp1
                        body.append(VecFma(t2_g, addr_vec, const_vec_2, neg_fp1_vec))
                    body.append(VecElem("+", addr_vec, t2_g, t1_g))        # next_addr = base + parity

                # --- on the final round, vstore val_g to its output address
                # (overlaps the body tail via the idle store engine; the linear
                # epilogue vstore loop is gone) ---
                if r == rounds - 1:
                    body.append(VStore(out_addr[g], val_vec))

        # Rename: symbolic -> resolved IR (all-pinned translation), then schedule.
        body = self.re.resolve_instrs(body)
        body_instrs = self.build(body, vliw=True, seed=42, picker="weighted",
                                   weights=Weights(sink=-3, load=-1.5, raw=-0.25,
                                                   war=6, rigid=0.25, idx=-4))
        self.instrs.extend(body_instrs)

        # =====================================================================
        # Epilogue: the val[256] vstores now overlap the body tail (each group's
        # vstore fires from the body once its round-15 val is ready, using the
        # per-group out_addr). Nothing left here but the final pause.
        # =====================================================================

        # Pause 2 -- match reference_kernel2's final yield (final mem).
        # Must come AFTER the body so machine.mem holds the final values
        # when the test recommends execution on i=1 (final yield).
        self.add(Pause())

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
