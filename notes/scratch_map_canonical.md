# Per-Lane Register Flow — v3a (implemented)

The implemented kernel (`build_kernel` in `perf_takehome.py`) follows a hard
scratch rule:

> **256 words shared, then 5 words per lane across 256 lanes (= 1280),
> exhausting the 1536-word file.**

The per-lane sector is **5 contiguous planes of 256** (struct-of-arrays).
Lane `i` owns one word per plane at `plane_base + i`; group `g` (lanes
`8g..8g+7`) therefore forms a VLEN=8 contiguous vector at `plane_base + 8g`
— vectorizable by `valu` at zero gather cost. This is why per-lane state is
SoA rather than array-of-structs: `valu` elementwise ops require 8 contiguous
words, which strided (AoS) layouts can't provide.

```
SCRATCH 1536 words

[ 0..158]   Shared sector               159  (97 free under 256 budget)
[159..414]  plane 0 : val[256]          256
[415..670]  plane 1 : idx[256]          256
[671..926]  plane 2 : t1[256]           256
[927..1182] plane 3 : t2[256]           256
[1183..1438] plane 4 : nv[256]          256
[1439..1535] shared-sector free          97
```

# Shared sector  [0..158]  (159 / 256 words)

Everything that is **not** per-lane persistent state lives here.

| addrs | name | use |
|---|---|---|
| 0..6 | rounds, n_nodes, batch_size, forest_height, forest_values_p, inp_indices_p, inp_values_p | header (`load` from mem[0..6]) |
| 7 | zero | scalar const 0 (vload/vstore pointer base) |
| 8 | eight | scalar const 8 (vload/vstore pointer stride) |
| 9 | addr_a | vload/vstore scalar pointer (prologue/epilogue only) |
| 10..17 | addr_vec | gather-address vector (8 lanes' `forest_p + idx`), ≤1-cycle live |
| 18..25 | forest_p_vec | broadcast `forest_values_p`, added to idx_vec per group |
| 26..158 | 15 broadcast vec consts + their 14 src scalars | hash/idx constants (see below) |

The 15 vector constants (each a `vbroadcast` of a `scratch_const` src):

| plane-less vec | lanes | role |
|---|---|---|
| mult4097_vec, K0_vec | 8+8 | stage 0 fma: `a*4097 + K0` |
| mult33_vec, K2_vec | 8+8 | stage 2 fma: `a*33 + K2` |
| mult9_vec, K4_vec | 8+8 | stage 4 fma: `a*9 + K4` |
| K1_vec, shift19_vec | 8+8 | stage 1: `(a^K1) ^ (a>>19)` |
| K3_vec, shift9_vec | 8+8 | stage 3: `(a+K3) ^ (a<<9)` |
| K5_vec, shift16_vec | 8+8 | stage 5: `(a^K5) ^ (a>>16)` |
| two_vec, one_vec | 8+8 | idx: `base = 2*idx + 1` fma |
| zero_vec | 8 | wrap round: `idx &= 0` |

Hash consts are kept in two dicts (`fma_vec_consts`, `irr_vec_consts`) so the
literal `9` — both a multiplier (stage 4) and a shift amount (stage 3) —
doesn't collide as a dict key.

**Shared-sector free: 97 words.** Reserved for future shared pipeline buffers
(deeper in-flight prefetch vectors, addr_b for the 2nd load port, etc.).

# Per-lane sector  [159..1438]  (5 × 256 = 1280 words)

SoA planes; each lane's 5 words are at `plane_base + i` across the 5 planes.
Group `g`'s 8-paper vector is `plane_base + 8g`.

| plane | addrs | name | role |
|---|---|---|---|
| 0 | 159..414 | val[i] | carried across rounds **and** the running hash register during each round's hash. Entry XOR writes `a` into it; the final hash stage leaves `v` there for the next round — zero copy-out. Prologue `vload`, epilogue `vstore`. |
| 1 | 415..670 | idx[i] | tree index; zero-initialized (scratch clears to 0; initial idx=0 ⇒ no init). Set to 0 on the wrap round. |
| 2 | 671..926 | t1[i] | per-lane irreducible-stage scratch (one of the two parallel transforms). Live only within a single xor/add-shift stage, dead between stages. Reused post-hash for the idx parity bit. |
| 3 | 927..1182 | t2[i] | per-lane irreducible-stage scratch (the other transform). Same live range as t1. Reused post-hash for the idx `base = 2*idx+1` result. |
| 4 | 1183..1438 | nv[i] | per-lane `node_val` landing (the 8 scalar `load`s group g issues land here ⇒ the entry XOR's second operand). Spare for future per-lane use. |

# Why per-lane temps (not shared ones)

v2 used a single shared `t1_vec`/`t2_vec` pair, reused across all 32 groups
× 16 rounds. That works under one-slot-per-bundle because only one group is
mid-stage at a time. But the moment we VLIW-pack multiple groups in parallel
(different groups at different hash stages in the same bundle), shared stage
scratch would alias across the in-flight groups and force rename tracking.

The 5-per-lane SoA layout removes that hazard by construction: each lane has
its own `t1`/`t2`/`nv`, so two groups packed into the same VLIW bundle are
guaranteed to touch disjoint scratch words regardless of which stage they're
in. The 768-word cost (vs shared t1/t2/nv = 24 words) is absorbed by scratch
having the headroom (97 free in shared + the per-lane planes fit exactly).
This is the "future-proof" tradeoff: trade scratch for scheduler simplicity.

# Canonical op flow (group g, normal round)

Addresses are compile-time literals under the unrolled loop:
```
val_g  = 159 + 8g     idx_g = 415 + 8g     t1_g = 671 + 8g
t2_g   = 927 + 8g      nv_g = 1183 + 8g
addr_vec = 10..17     forest_p_vec = 18..25    (shared)
```
```python
# gather: 1 valu addr + 8 scalar loads  -> nv plane (naive loader)
valu  addr_vec  = idx_g + forest_p_vec
for j in 8:   load  nv_g[j], addr_vec[j]

# entry XOR (valu, NOT alu -- 8 lanes/slot)
valu  val_g    = val_g ^ nv_g                # a

# 12-slot hash on valu (val_g is the running reg; t1_g/t2_g stage scratch)
valu  val_g    = val_g * mult4097_vec + K0_vec          # stage 0 fma
valu  t1_g     = val_g ^ K1_vec                         # stage 1 (parallel)
valu  t2_g     = val_g >> shift19_vec
valu  val_g    = t1_g ^ t2_g                            # stage 1 combine
valu  val_g    = val_g * mult33_vec  + K2_vec           # stage 2 fma
valu  t1_g     = val_g + K3_vec                         # stage 3 (parallel)
valu  t2_g     = val_g << shift9_vec
valu  val_g    = t1_g ^ t2_g                            # stage 3 combine
valu  val_g    = val_g * mult9_vec   + K4_vec           # stage 4 fma
valu  t1_g     = val_g ^ K5_vec                         # stage 5 (parallel)
valu  t2_g     = val_g >> shift16_vec
valu  val_g    = t1_g ^ t2_g                            # stage 5 -> v  (val_g = v)

# branchless idx (valu); reuses t1_g/t2_g (dead post-hash)
valu  t1_g     = val_g & one_vec                        # parity = v & 1
valu  t2_g     = idx_g * two_vec + one_vec              # base = 2*idx + 1
valu  idx_g    = t2_g + t1_g                            # next = base + parity

# wrap round (r == height == 10): replace the above with
valu  idx_g    = idx_g & zero_vec                       # idx := 0
```

Op count per group per round (one slot/bundle under vliw=False):
- gather      : 1 valu + 8 load  = 9
- entry XOR   : 1 valu             = 1
- hash        : 12 valu            = 12
- idx update  : 3 valu  (1 on wrap)= 3 (or 1)
- ──────
- normal round : 25 bundles/group
- wrap round   : 23 bundles/group
- average      : 24.875 bundles/group
Total body: 24.875 × 32 × 16 ≈ 12 736 cycles + ~175 prologue/epilogue ≈ 12 911.

# History (what earlier maps said, now superseded)

- **v2 scratch_map.md** planned a per-lane `[t0,t1,t2]` *triple* (768 words)
  with `t0` as a separate "current hash" reg. The canonical insight (`val[i]`
  *is* the current-hash reg; no t0 needed) removed 256 of those words, and
  vectorization made the per-lane layout SoA rather than the previous
  `608 + 3*i + (0|1|2)` strided form. That map is retained only for history.
- The earlier "shared single-pair t1_vec/t2_vec" idea (canonical v3 draft)
  was dropped in favor of the 5-per-lane rule above — future-proof against
  VLIW rename hazards.
- The v2 sector 2 runtime temps `addr_b`, `node_val_b`, `cnt_round`,
  `cnt_lane` were never used (no looping in the unrolled kernel; no 2nd load
  port yet) and are **reverted to available**: the 97-word shared free region
  is exactly where those will be drawn from when we add dual-port prefetch.
```