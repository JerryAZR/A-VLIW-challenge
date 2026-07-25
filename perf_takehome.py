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
    Sym, Alu, VecElem, VecFma, VBroadcast, Load, VLoad, Const, VStore,
    VSelect, Pause, DebugVCompare, Gather,
)
from scheduler import Weights
from problem import (
    DebugInfo,
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


# Picker weights tuned for the regalloc (RAW-only DAG) path by sweep_regalloc -
# the register-freeing bias is always-on there, and these base weights order
# the remaining work best. (Found by random search + local refinement.)
# group=-4 (sweep_rollout): prioritize earlier DAG sink groups - finish a
# group's chains before opening the next. This serialization meters in-flight
# register pressure AND improves throughput: 1413 -> 1290 cyc.
REGALLOC_WEIGHTS = Weights(sink=-1, load=5, raw=1, war=1, rigid=1, idx=-4,
                           group=-4)

# Body scheduler selection: "greedy" (regalloc.schedule, weighted picker +
# freeing bias) or "rollout" (rollout.schedule_rollout: per-cycle
# trial-and-score search over placement orderings).
SCHEDULER_MODE = "rollout"
ROLLOUT_TRIALS = 1           # K=1: priority fn only (1291 cyc, 5x faster);
                             # raise K after other parts are fully optimized
ROLLOUT_SORT_FUNCS = None    # explicit trial set override; None -> default
ROLLOUT_SEED = 42
ROLLOUT_SCORE_WEIGHTS = None  # None -> rollout.ScoreWeights() defaults

# Which small const vectors are COMPUTED on valu (vs const+vbroadcast).
# Measured incrementally (sweep_consts.py); dependencies (19/33/4097 read
# const_vec_16, etc.) work with either form of their source.
COMPUTED_CONSTS = {1, 2, 3, 9, 16, 19, 33, 4097}

# Level-3 path-bit recompute toggle. The level-3 select needs the three
# descent path bits (d0,d1,d2). They can either be RETAINED from rounds 0-2
# (the path_g[0..2] writes in the addr update) or RECOMPUTED here from addr
# (offset = addr-(forest_p+7); d0=offset>>2, d1=(offset>>1)&1, d2=offset&1).
#
# Retention is cheaper (no extra valu) but keeps 3 path registers live across
# rounds 0-3, raising peak vector pressure ~64 granules - the greedy scheduler
# deadlocks on that at SCRATCH_SIZE=1536. The recompute breaks the liveness
# chain (round 0-2 versions free early) at the cost of ~5 valu ops/group/round.
#
# With more scratch space, or once a pressure-aware scheduler can fit the
# retained version, set this False to drop the recompute and use pure retention
# (validated correct via _check_big.py: 1363 cyc retention vs 1389 recompute).
RECOMPUTE_PATH_BITS = False


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        # Register allocator: owns all scratch space, created by
        # _emit_regalloc during build_kernel. Kept for debug_info().
        self.allocator = None
        # The resolved (address-assigned) body instructions, kept from the
        # last build() so a tracing harness can map rid -> instruction for
        # operand-level read/write capture.
        self.resolved_body = None

    def debug_info(self):
        """The simulator's debug scratch map. Only valid after build_kernel()
        has run (the register allocator that owns scratch is created there)."""
        if self.allocator is None:
            raise RuntimeError(
                "debug_info() called before build_kernel() - the register "
                "allocator (and its scratch map) does not exist yet")
        return DebugInfo(scratch_map=self.allocator.debug_map())

    def add_pause(self):
        """Emit a flow barrier as a one-slot bundle (linear prologue/epilogue).
        Pause has no register operands, so it needs no allocation - just lower
        it directly."""
        self.instrs.append({Pause().engine: [Pause().lower()]})

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
            for the branchless addr update (they're dead post-final-combine).
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
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int,
        prune: bool = True,
    ):
        """8-lane-per-group kernel. Cross-lane vectorization over the
        `valu` unit (VLEN=8 lanes per group, 32 groups), packed by the
        DAG-driven VLIW scheduler (vliw=True, weighted picker).

        Layout (all pipeline state lives in scratch; only const_vec_0 is
        pinned - everything else is a versioned temporary allocated by the
        two-phase register allocator, see notes/scratch_map_canonical.md):
          - val[256] : per-lane carried state AND the running hash register
            during each round's hash (entry XOR writes `a` into it; the final
            hash stage leaves `v` there for the next round - zero copy-out).
            Stored as 32 VLEN=8 vectors.
          - addr[256] : per-lane tree ADDRESS (idx + forest_p); the gather
            reads it directly. Carried across rounds.
          - parity[256] : per-lane (val & 1), persists across the round
            boundary to drive the next round's level-1/2 select.
          - t1/t2/nv : short-lived loop-body temps (hash stage scratch, addr
            base, node_val landing) as SSA tags; the allocator re-homes them
            per write so groups never alias.
          - consts : the 6 hash constants + shift amounts, broadcast once at
            prologue as VLEN=8 vectors and reused across all hashes.

        Gather (non-contiguous tree.values[idx[i]]) is the one operation that
        cannot be vectorized (the ISA has no scatter/gather): for each group
        one Gather op reads the per-lane addr vector and is decomposed by the
        scheduler into 8 scalar loads landing in the nv plane.

        Branchless addr update: parity = val & 1; base = 2*addr + neg_fp1
        (valu fma); next_addr = base + parity (valu `+`). Bit-exact with the
        reference's `2*idx + (1 if even else 2)` in idx terms.

        Wrap is a build-time-known per-round decision (verified uniform wrap
        on round=height for the canonical shape): on that round we skip the
        branchless update and write addr := forest_p (one `valu` `-`).

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
        # Scratch layout: the register allocator owns all scratch space. Only
        # const_vec_0 is pinned (at address 0; scratch starts all-zero, so it
        # needs no write). Every other symbol is a versioned temporary,
        # dynamically allocated by the allocator from its free pools
        # (8-aligned vector granules, then a scalar pool at the top). No
        # hand-laid-out map - the allocator assigns addresses on write.
        # =====================================================================

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]

        # ---- Pinned symbol declarations ----
        # ---- Symbol declarations ----
        # The builder defines symbols; the engine owns all scratch space and
        # assigns addresses on write. Only const_vec_0 is pinned; the rest
        # are versioned temporaries.
        #
        # Per-lane SoA state as 32 group-vector symbols each: val[g] /
        # addr[g] / parity[g] are lanes 8g..8g+7 of a logical plane.
        val  = [Sym(f"val[{g}]",  True) for g in range(n_groups)]  # running hash + carried state
        addr = [Sym(f"addr[{g}]", True) for g in range(n_groups)]  # tree ADDRESS = idx + forest_p (stored, not idx)
        # Local temporaries: ONE shared tag each across all groups (loop-body
        # locals - each is dead within one group's round). The allocator
        # re-homes them per write, so different groups' in-flight values
        # still land in different physical homes (no cross-group WAR chains
        # by construction), and homes recycle through the free pool.
        t1 = Sym("t1", True)   # hash stage scratch + select tree intermediate
        t2 = Sym("t2", True)   # hash stage scratch + addr-update base
        t3 = Sym("t3", True)   # select tree intermediate (level 3)
        nv = Sym("nv", True)   # node_val landing / gather pad
        # Parity (hash & 1) doubles as the descent path bit: path[g][l] holds
        # the bit computed at level l of a descent (rounds 0-9 / 11-15 map to
        # levels 0-9 / 0-4). Levels 0-2 stay live until the level-3 select
        # (rounds 3/14); levels 3+ are read only by their own round's addr
        # update and freed right after.
        path = [[Sym(f"path[{g}][{l}]", True) for l in range(forest_height)]
                for g in range(n_groups)]

        # CONST vectors: uniform value*8. Small reusable ones are named
        # const_vec_<value> and sorted by value (not tied to a step - e.g. 9 is
        # both the stage-4 multiplier and the stage-3 shift). K0..K5 are the
        # stage-specific hash addend/xor constants. No separate broadcast-source
        # scalars: each is created by `load const` into its own lane 0, then
        # self-broadcast (vbroadcast vec, vec) - see the prologue. Scalar uses of
        # a value read the matching const_vec's lane 0.
        const_vec_0    = Sym("const_vec_0", True)
        const_vec_1    = Sym("const_vec_1", True)
        const_vec_2    = Sym("const_vec_2", True)
        const_vec_3    = Sym("const_vec_3", True)
        const_vec_9    = Sym("const_vec_9", True)     # stage4 mult + stage3 shift
        const_vec_16   = Sym("const_vec_16", True)   # stage5 shift
        const_vec_19   = Sym("const_vec_19", True)   # stage1 shift
        const_vec_33   = Sym("const_vec_33", True)   # stage2 mult
        const_vec_4097 = Sym("const_vec_4097", True) # stage0 mult
        K0_vec = Sym("K0_vec", True)   # stage 0 addend
        K1_vec = Sym("K1_vec", True)   # stage 1 xor const
        K2_vec = Sym("K2_vec", True)   # stage 2 addend
        K3_vec = Sym("K3_vec", True)   # stage 3 add const
        K4_vec = Sym("K4_vec", True)   # stage 4 addend
        K5_vec = Sym("K5_vec", True)   # stage 5 xor const
        # VAR vectors: runtime values (forest_p = header broadcast; tree_preload
        # = non-uniform vload of tree[0..7]; tree0..6 = its lane broadcasts).
        forest_p_vec = Sym("forest_p_vec", True)
        neg_fp1_vec  = Sym("neg_fp1_vec", True)  # 1 - forest_p (next-addr: 2*addr + neg_fp1 + parity)
        pos_fp5_vec  = Sym("pos_fp5_vec", True)  # 5 + forest_p (level-2 select)
        pos_fp7_vec  = Sym("pos_fp7_vec", True)  # 7 + forest_p (level-3 path recompute)
        tree_preload = Sym("tree_preload", True)  # 8 words: tree[0..7]
        tree0_vec = Sym("tree0_vec", True)
        tree1_vec = Sym("tree1_vec", True)
        tree2_vec = Sym("tree2_vec", True)
        tree3_vec = Sym("tree3_vec", True)
        tree4_vec = Sym("tree4_vec", True)
        tree5_vec = Sym("tree5_vec", True)
        tree6_vec = Sym("tree6_vec", True)
        tree_vecs = [tree0_vec, tree1_vec, tree2_vec,
                     tree3_vec, tree4_vec, tree5_vec, tree6_vec]
        # Level-3 preload: nodes 7-14. tree7 = tree_preload.lane(7) (the bonus
        # 8th word of the tree[0..7] vload); tree8-14 = tree_preload2.lane(0..6).
        tree_preload2 = Sym("tree_preload2", True)  # 8 words: tree[8..15]
        tree7_vec = Sym("tree7_vec", True)
        tree8_vec = Sym("tree8_vec", True)
        tree9_vec = Sym("tree9_vec", True)
        tree10_vec = Sym("tree10_vec", True)
        tree11_vec = Sym("tree11_vec", True)
        tree12_vec = Sym("tree12_vec", True)
        tree13_vec = Sym("tree13_vec", True)
        tree14_vec = Sym("tree14_vec", True)
        tree8_14 = [tree8_vec, tree9_vec, tree10_vec, tree11_vec,
                    tree12_vec, tree13_vec, tree14_vec]

        # Scalars: `eight` (vload/vstore stride of 8 - the only CONST scalar);
        # header vars (loaded from mem); addr_a (vload/vstore ptr); per-group
        # out_addr (= inp_values_p + 8g: runtime base + compile-time offset,
        # so the round-15 vstores can issue 2/cyc in any order).
        eight_const = Sym("eight")
        header = {v: Sym(v) for v in init_vars}
        addr_a = Sym("addr_a")
        out_addr = [Sym(f"out_addr[{g}]") for g in range(n_groups)]

        # Scratch space is owned by the two-phase register allocator
        # (regalloc.py). The ONLY pin is const_vec_0 (at address 0; scratch
        # starts all-zero, so it needs no write). Every other symbol is a
        # versioned SSA tag, allocated a physical register when its writer is
        # placed and freed when its last read commits.
        # (vec, literal) pairs: the prologue `load const` lane 0 + self-broadcast.
        # const_vec_0 is EXCLUDED: scratch starts all-zero, so it needs no
        # const-load + broadcast (and it is the one reserved home - see above).
        # SMALL consts (1, 2, 3, 9, 16, 19, 33, 4097) are NOT here: they are
        # COMPUTED from each other on valu (see below), trading 8 const loads
        # + 8 broadcasts for 9 valu ops on the less-loaded engines.
        vec_bcasts = [
            (K0_vec, 0x7ED55D16), (K1_vec, 0xC761C23C), (K2_vec, 0x165667B1),
            (K3_vec, 0xD3A2646C), (K4_vec, 0xFD7046C5), (K5_vec, 0xB55A4F09),
        ]
        # Small consts NOT in COMPUTED_CONSTS fall back to const+vbroadcast.
        _small_syms = {1: const_vec_1, 2: const_vec_2, 3: const_vec_3,
                       9: const_vec_9, 16: const_vec_16, 19: const_vec_19,
                       33: const_vec_33, 4097: const_vec_4097}
        vec_bcasts += [(sym, v) for v, sym in _small_syms.items()
                       if v not in COMPUTED_CONSTS]

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

        # =====================================================================
        # Prologue: load header; vload val[256]; broadcast consts.
        # Collected SYMBOLICALLY and tagged together with the
        # body below: with no app pins, prologue-written temps (val, consts,
        # header, out_addr) are read in the body, so liveness must span both.
        # =====================================================================
        prologue = []
        prologue.append(Const(eight_const, 8))     # vload/vstore stride
        for i, v in enumerate(init_vars):
            prologue.append(Const(addr_a, i))                      # addr_a := i
            prologue.append(Load(header[v], addr_a))               # header[v] := mem[i]

        # vload val[256] as 32 vectors of 8 contiguous words from mem[inp_values_p..].
        prologue.append(Alu("+", addr_a, header["inp_values_p"],
                            const_vec_0.lane(0)))
        for k in range(n_groups):
            prologue.append(VLoad(val[k], addr_a))
            if k < n_groups - 1:
                prologue.append(Alu("+", addr_a, addr_a, eight_const))

        # Broadcast forest_values_p (from a header var, not a literal).
        prologue.append(VBroadcast(forest_p_vec, header["forest_values_p"]))

        # Create each const vector: a whole-vector VBroadcast births the temp's
        # home (the current engine does not support LaneRef writes on temps),
        # reading a named scalar-const broadcast source. One scalar const per
        # distinct value. (const_vec_0 excluded: it stays 0 from scratch init.)
        scalar_consts = {}
        for vec_sym, value in vec_bcasts:
            if value not in scalar_consts:
                scalar_consts[value] = Sym(f"const_{value}")
                prologue.append(Const(scalar_consts[value], value))
            prologue.append(VBroadcast(vec_sym, scalar_consts[value]))

        # Small const vectors COMPUTED from each other on valu (instead of
        # const load + vbroadcast): trades const-load slots for ~1 net valu
        # op each and fewer prologue nodes. Set membership: COMPUTED_CONSTS.
        # Fallback forms (const+vbroadcast) are added to vec_bcasts above;
        # dependencies work with either form of their source.
        if 1 in COMPUTED_CONSTS:
            prologue.append(VecElem("==", const_vec_1, const_vec_0, const_vec_0))  # 1 = (0==0)
        if 2 in COMPUTED_CONSTS:
            prologue.append(VecElem("+", const_vec_2, const_vec_1, const_vec_1))  # 2 = 1+1
        if 3 in COMPUTED_CONSTS:
            prologue.append(VecElem("+", const_vec_3, const_vec_1, const_vec_2))  # 3 = 1+2
        if 9 in COMPUTED_CONSTS:
            prologue.append(VecElem("*", const_vec_9, const_vec_3, const_vec_3))  # 9 = 3*3
        if 16 in COMPUTED_CONSTS:
            prologue.append(VecElem("<<", const_vec_16, const_vec_2, const_vec_3))  # 16 = 2<<3
        if 19 in COMPUTED_CONSTS:
            prologue.append(VecElem("+", const_vec_19, const_vec_16, const_vec_3))  # 19 = 16+3
        if 33 in COMPUTED_CONSTS:
            prologue.append(VecFma(const_vec_33, const_vec_16, const_vec_2,
                                   const_vec_1))                               # 33 = 16*2+1
        if 4097 in COMPUTED_CONSTS:
            const_vec_64 = Sym("const_vec_64", True)   # temp for 4097
            prologue.append(VecElem("<<", const_vec_64, const_vec_16, const_vec_2))  # 64 = 16<<2
            prologue.append(VecFma(const_vec_4097, const_vec_64, const_vec_64,
                                   const_vec_1))                               # 4097 = 64*64+1

        # neg_fp1 = 1 - forest_values_p (used by the next-addr update). Computed
        prologue.append(VecElem("-", neg_fp1_vec, const_vec_1, forest_p_vec))
        # pos_fp5 = 5 + forest_values_p (used by the level 2 select). Computed
        prologue.append(VecElem("+", pos_fp5_vec, const_vec_2, const_vec_3))  # pos_fp5 = 5
        prologue.append(VecElem("+", pos_fp5_vec, pos_fp5_vec, forest_p_vec))
        # pos_fp7 = 7 + forest_values_p (level-3 path-bit recompute offset).
        prologue.append(VecElem("+", pos_fp7_vec, const_vec_2, const_vec_2))  # 4
        prologue.append(VecElem("+", pos_fp7_vec, pos_fp7_vec, const_vec_3))  # 7
        prologue.append(VecElem("+", pos_fp7_vec, pos_fp7_vec, forest_p_vec))

        # vload tree[0..7] (levels 0-2 = 7 nodes + node 7 as the bonus 8th
        # word) into tree_preload, then tree[8..15] into tree_preload2.
        prologue.append(Alu("+", addr_a, header["forest_values_p"],
                            const_vec_0.lane(0)))
        prologue.append(VLoad(tree_preload, addr_a))
        prologue.append(Alu("+", addr_a, addr_a, eight_const))  # forest_p + 8
        prologue.append(VLoad(tree_preload2, addr_a))
        # Broadcast tree[0..6] into shared vector constants.
        for i in range(7):
            prologue.append(VBroadcast(tree_vecs[i], tree_preload.lane(i)))
        # Broadcast the level-3 nodes: tree7 from preload.lane(7), tree8-14
        # from preload2.lane(0..6).
        prologue.append(VBroadcast(tree7_vec, tree_preload.lane(7)))
        for i in range(7):
            prologue.append(VBroadcast(tree8_14[i], tree_preload2.lane(i)))

        # =====================================================================
        # Body -- unrolled rounds x 32 groups, one slot per bundle.
        # =====================================================================
        body = []
        # Output addresses: load each group's compile-time offset (8g) as a
        # const, then add the runtime inp_values_p. Independent per group (no
        # addr_a chain) so the round-15 vstores can fire 2/cyc in any order.
        # Scheduled early; ready well before the vstores need them.
        inp_values_p = header["inp_values_p"]
        for g in range(n_groups):
            body.append(Const(out_addr[g], g * VLEN))   # offset 8g
            body.append(Alu("+", out_addr[g], out_addr[g], inp_values_p))
        for r in range(rounds):
            # level of this round within its descent (rounds 0-9 -> 0-9,
            # round 10 wraps, rounds 11-15 -> 0-4).
            level = r if r < WRAP_ROUND else r - WRAP_ROUND - 1
            for g in range(n_groups):
                is_wrap = (r == WRAP_ROUND)
                # per-group vector symbols of the SoA per-lane planes
                val_vec  = val[g]
                addr_vec = addr[g]
                path_g   = path[g]      # per-level path bits; path_g[level] is
                                        # this round's parity destination
                # t1 / t2 / nv : the shared loop-body local tags
                base_i   = g * VLEN
                keyval  = [(r, base_i + j, "val") for j in range(VLEN)]
                keynv   = [(r, base_i + j, "node_val") for j in range(VLEN)]
                keyhv   = [(r, base_i + j, "hashed_val") for j in range(VLEN)]

                # --- node_val gather or preload-select (rounds 0-3 use preloaded) ---
                if r in (0, 11):
                    # Level 0: all lanes at idx=0. node_val = tree[0].
                    body.append(VecElem("^", nv, tree0_vec, const_vec_0))
                elif r in (1, 12):
                    # Level 1: idx in {1,2}. idx = 1 + d0, so the level-0 path
                    # bit IS the select bit (idx=1 -> tree1, idx=2 -> tree2).
                    body.append(VSelect(nv, path_g[0], tree2_vec, tree1_vec))
                elif r in (2, 13):
                    # Level 2: idx in {3,4,5,6} = 3 + 2*d0 + d1. Select among
                    # the 4 preloaded nodes, consuming path bits in order
                    # (d0=path[0] first, then d1=path[1]) so d0 frees early.
                    #   idx-3=0->tree3, 1->tree4, 2->tree5, 3->tree6
                    body.append(VSelect(t1, path_g[0], tree5_vec, tree3_vec))  # d0?t5:t3
                    body.append(VSelect(t2, path_g[0], tree6_vec, tree4_vec))  # d0?t6:t4
                    body.append(VSelect(nv, path_g[1], t2, t1))              # d1?(d0=1):(d0=0)
                elif r in (3, 14):
                    # Level 3: idx in {7..14} = 7 + 4*d0 + 2*d1 + d2. 3-level
                    # vselect tree consuming path bits in strict order
                    # d0 -> d1 -> d2, so each path bit frees as early as
                    # possible. t1/t2/t3/nv are the tree's intermediates.
                    #
                    # Path-bit RECOMPUTE (breaks the round 0-2 retention
                    # chain): the path bits were written at rounds 0-2 but are
                    # only needed here. Instead of keeping them live across 3
                    # rounds, recompute them from addr (which is live anyway):
                    # offset = addr - (forest_p + 7); then d0=offset>>2,
                    # d1=(offset>>1)&1, d2=offset&1, written back into the same
                    # path_g[0..2] storage. The round 0-2 versions free early.
                    #
                    # Gated by RECOMPUTE_PATH_BITS: with enough scratch (or a
                    # pressure-aware scheduler) the retained version alone is
                    # correct and ~26 cyc cheaper - flip the flag off then.
                    if RECOMPUTE_PATH_BITS:
                        body.append(VecElem("-", t2, addr_vec, pos_fp7_vec))      # offset = addr - (forest_p+7)
                        body.append(VecElem(">>", path_g[0], t2, const_vec_2))  # d0 = offset >> 2
                        body.append(VecElem(">>", t1, t2, const_vec_1))         # tmp = offset >> 1
                        body.append(VecElem("&", path_g[1], t1, const_vec_1))   # d1 = tmp & 1
                        body.append(VecElem("&", path_g[2], t2, const_vec_1))   # d2 = offset & 1
                    body.append(VSelect(t1, path_g[0], tree11_vec, tree7_vec))   # d0?t11:t7
                    body.append(VSelect(t2, path_g[0], tree12_vec, tree8_vec))   # d0?t12:t8
                    body.append(VSelect(t3, path_g[0], tree13_vec, tree9_vec))   # d0?t13:t9
                    body.append(VSelect(nv, path_g[0], tree14_vec, tree10_vec))  # d0?t14:t10
                    body.append(VSelect(t1, path_g[1], t3, t1))                  # d1?s10:s00
                    body.append(VSelect(t2, path_g[1], nv, t2))                  # d1?s11:s01
                    body.append(VSelect(nv, path_g[2], t2, t1))                  # d2?u1:u0
                else:
                    # Rounds 4+: gather from mem. addr_vec already holds the
                    # tree address (idx + forest_p), so the loads read it
                    # directly - no per-round address-add valu. One Gather op
                    # (nv = mem[addr[0..7]]); the allocator re-homes the
                    # shared nv tag and decomposes this into 8 scalar loads.
                    # nv is then read by the entry XOR below.
                    body.append(Gather(nv, addr_vec))

                body.append(DebugVCompare(nv, keynv))
                body.append(DebugVCompare(val_vec, keyval))  # val before xor

                # --- entry XOR: val_vec = val_vec ^ nv  (a) ---
                body.append(VecElem("^", val_vec, val_vec, nv))

                # --- 12-slot hash, fully on valu (8 lanes / slot) ---
                body.extend(self.build_vec_hash(val_vec, t1, t2, r, base_i,
                                                fma_vec_consts, irr_vec_consts))

                # debug: hashed_val == v == val_vec after hash
                body.append(DebugVCompare(val_vec, keyhv))

                # --- post-hash: addr update or wrap (branchless, on valu) ---
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
                    # path bit = v & 1. Levels 0-2 persist to the level-3
                    # select (rounds 3/14); levels 3+ feed only this addr
                    # update.
                    pdest = path_g[level]
                    body.append(VecElem("&", pdest, val_vec, const_vec_1))
                    if r == 0:
                        # idx=0: next_addr = forest_p + 1 + parity = (2 - neg_fp1) + parity
                        body.append(VecElem("-", t2, const_vec_2, neg_fp1_vec))  # 2 - neg_fp1
                    else:
                        # next_addr base = 2*addr + neg_fp1
                        body.append(VecFma(t2, addr_vec, const_vec_2, neg_fp1_vec))
                    body.append(VecElem("+", addr_vec, t2, pdest))        # next_addr = base + pathbit

                # --- on the final round, vstore val_g to its output address
                # (overlaps the body tail via the idle store engine; the linear
                # epilogue vstore loop is gone) ---
                if r == rounds - 1:
                    body.append(VStore(out_addr[g], val_vec))

        self._emit_regalloc(prologue, body, prune)

        # =====================================================================
        # Epilogue: the val[256] vstores overlap the body tail (each group's
        # vstore fires from the body once its round-15 val is ready, using
        # the per-group out_addr). The two oracle pauses are injected into
        # existing bundles' flow slots by _insert_pauses (0 cycles at
        # grading, where enable_pause=False).
        # =====================================================================

    def _emit_regalloc(self, prologue, body, prune):
        """Two-phase register-allocation path (regalloc.py): SSA tag chains +
        RAW-only DAG + schedule-time allocation, over ONE merged DAG of
        prologue + body (the scheduler interleaves setup work - consts,
        broadcasts, val/tree vloads - with early body compute instead of a
        serial one-slot-per-bundle prologue). const_vec_0 stays pinned at
        address 0.
        """
        from regalloc import (tag_raw_chains, build_dag,
                              schedule as reg_schedule, RegisterAllocator)
        tagged, read_count = tag_raw_chains(prologue + body,
                                            pinned={"const_vec_0"})
        dag = build_dag(tagged, pinned={"const_vec_0"})
        if prune:
            # Drop debug/dead nodes, then recompute read counts over the
            # survivors so registers free when their last surviving reader
            # commits (the pruned readers no longer exist).
            from scheduler import prune_to_stores
            from regalloc import recompute_read_count
            dag = prune_to_stores(dag)
            read_count = recompute_read_count(
                [n.instr for n in dag.nodes], pinned={"const_vec_0"})
        allocator = RegisterAllocator(read_count)
        self.allocator = allocator
        if SCHEDULER_MODE == "rollout":
            from rollout import schedule_rollout
            body_instrs = schedule_rollout(
                dag, read_count, seed=ROLLOUT_SEED, trials=ROLLOUT_TRIALS,
                sort_funcs=ROLLOUT_SORT_FUNCS, weights=REGALLOC_WEIGHTS,
                score_weights=ROLLOUT_SCORE_WEIGHTS, allocator=allocator)
        else:
            body_instrs = reg_schedule(dag, read_count, seed=42,
                                       picker="weighted",
                                       weights=REGALLOC_WEIGHTS,
                                       allocator=allocator)
        self.instrs.extend(body_instrs)
        self._insert_pauses()
        # Kept for dev tools (_validate.py): the scheduled body's RAW-only
        # DAG and the tagged body instructions (map rid -> instruction).
        self.body_dag = dag
        self.resolved_body = [n.instr for n in dag.nodes]

    def _insert_pauses(self):
        """The two oracle barriers (dev-only; 0 extra cycles at grading where
        enable_pause=False and they ride existing bundles' flow slots):

          pause 1 - initial-mem barrier for _check.py's first run(): injected
            into the first store-free bundle with a free flow slot (the
            merged schedule puts all stores in the tail, so this is bundle 0
            in practice).
          pause 2 - final-mem barrier: appended to the last bundle's flow
            slot, or emitted as its own bundle if that slot is taken.
        """
        for b in self.instrs:
            assert "store" not in b, "store scheduled before pause 1"
            if len(b.get("flow", [])) < 1:    # SLOT_LIMITS["flow"] == 1
                b.setdefault("flow", []).append(Pause().lower())
                break
        last = self.instrs[-1]
        if len(last.get("flow", [])) < 1:
            last.setdefault("flow", []).append(Pause().lower())
        else:
            self.instrs.append({Pause().engine: [Pause().lower()]})

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
