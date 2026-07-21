# Per-Lane Register Flow — Canonical Version (revised)

Two corrections vs. v2 of the scratch map, both important:

1. **`val[i]` IS the current-hash register (no separate t0).**
   `val[i]` is read once at entry (XOR with node_val) to produce `a`; then
   every hash stage writes its output back into `val[i]`; the final stage
   leaves `v` there, which is exactly the lane's new `val` for the next
   round. No move is ever needed at the end. The "t0" register from the
   v2 map was pure wastage — 256 words of scratch burned on a rename that
   the dataflow doesn't require.

   Consequence: **only 2 per-lane temps (`t1`, `t2`) are needed**, not 3.
   Both are stage scratch only — live within a single xor/add-shift stage
   and dead between stages (overwritten by the next stage's transforms).
   Reads as a normal program; no rename hazards; saves 256 words.

2. **`node_val` needs a real scratch destination (not magic).**
   The `("load", dest, addr)` instruction writes the fetched word into a
   scratch destination. `node_val` lives in that scratch word from the
   `load` cycle until consumed by the entry XOR one cycle later. In
   sequential mode, this is a single **shared** scratch word (one lane
   active at a time, so the buffer is reused across lanes). When we later
   vectorize across 8 lanes, this same word becomes an 8-word `node_val_vec`
   buffer (one shared region, sized to VLEN). Either way, it is **not**
   per-lane — it's transient and shared.

   Also note the gather address itself lives in scratch (e.g. addr slot 16):
   `alu  addr = forest_values_p + idx[i]` (T+0), then `load  node_val_buf,
   addr` (T+1). The address scratch word too is shared/transient in
   sequential mode.

# Revised scratch map (v3)

```
SCRATCH 1536 words (88% used, 12% free)

[0..15]     Scalar header + constants            16
[16..31]    Runtime temps (shared, transient)   16   (addr/load bufs, loop ctrs)
[32..95]    Vector constants (broadcast)         64   (8 vecs)
[96..351]   val[256]  (resident +  current-hash register, read-write)  256
[352..607]  idx[256]  (resident, read-write)     256
[608..1119] t1[256], t2[256]  (per-lane stage scratch)  512   (256 × 2)
[1120..1535] Free                                      416
```

Detailed sector addresses:

## Sector 1 — Scalar header + constants  [0..15]  (16 words)
| addr | name | source |
|---|---|---|
| 0..6 | rounds, n_nodes, batch_size, forest_height, forest_values_p, inp_indices_p, inp_values_p | `load` mem[0..6] |
| 7 | zero (const 0) | `load const` |
| 8 | one  (const 1) | `load const` |
| 9 | two  (const 2) | `load const` |
| 10–12 | shift_19, shift_9, shift_16 | `load const` |
| 13–15 | K1, K3, K5 | `load const` |

## Sector 2 — Runtime scratch scalars (shared/transient)  [16..31]  (16 words)
| addr | name | use |
|---|---|---|
| 16 | addr_a | gather address A (forest_values_p + idx[i]) — shared in seq mode |
| 17 | addr_b | gather address B (spare port / future 2-ahead prefetch) |
| 18 | node_val_buf | destination of `load node_val`; consumed by entry XOR next cycle |
| 19.. | reserved | future: pipeline state, prefetch buffers |
| 20 | cnt_round | outer-loop counter (when looping) |
| 21 | cnt_lane | inner-loop counter (when looping) |

## Sector 3 — Vector constants  [32..95]  (64 words = 8 vectors)
| addr | name | use |
|---|---|---|
| 32..39 | mult_4097_vec | stage 0 fma: a*4097 + K0 |
| 40..47 | K0_vec | stage 0 fma addend |
| 48..55 | mult_33_vec | stage 2 fma: a*33 + K2 |
| 56..63 | K2_vec | stage 2 fma addend |
| 64..71 | mult_9_vec | stage 4 fma: a*9 + K4 |
| 72..79 | K4_vec | stage 4 fma addend |
| 80..87 | two_vec | idx fma: idx*2 + 1 |
| 88..95 | one_vec | idx fma addend (`+1`) |

## Sector 4 — Lane state (resident)  [96..607]  (512 words)
| addr | name | init |
|---|---|---|
| 96..351 | val[256] | **dual role**: lane's `val` carried across rounds AND the running hash register during the hash. Prologue: `vload` from `mem[inp_values_p ..]` (16 cyc). Epilogue: `vstore` to same |
| 352..607 | idx[256] | initial = 0 (scratch zero-init → **no init needed**); on round 10 (uniform wrap), all 256 lanes set to 0 |

## Sector 5 — Per-lane stage scratch  [608..1119]  (512 words)
| lane i offset | name | live range |
|---|---|---|
| 608 + 2*i + 0 | t1[i] | within a single xor/add-shift stage only (parallel transform u1) |
| 608 + 2*i + 1 | t2[i] | within a single xor/add-shift stage only (parallel transform u2) |

`t1[i]`, `t2[i]` are written by the two parallel transforms of `val[i]` at the
start of a xor/add-shift stage and read by the `^`-combine at the end of the
same stage; they are dead between stages and reused by the next stage's
transforms. No cross-lane sharing ⇒ no rename hazards ⇒ VLIW-packable
without inter-lane stalls.

## Sector 6 — Free  [1120..1535]  (416 words)

# Canonical per-lane op flow (lane i, normal round)

Addresses (compile-time literals when unrolled):
```
val[i]  = 96    + i
idx[i]  = 352   + i
t1[i]   = 608 + 2*i
t2[i]   = 609 + 2*i
node_val_buf = 18    (shared)
addr_a       = 16    (shared)
```

Flow:
```
# --- gather (prefetched ~1 lane ahead; hidden behind compute) ---
alu  addr_a = forest_values_p + idx[i]              ; T+0
load node_val_buf, addr_a                          ; T+1   (load slot A)

# --- pre-hash XOR (val[i] becomes the running hash register `a`) ---
alu  val[i] = val[i] ^ node_val_buf                ; T+2   (val[i] := a)

# --- hash, 12 slots; val[i] is current; t1[i], t2[i] stage scratch ---
# stage 0 (fma): s0 = a * 4097 + K0
valu multiply_add  val[i] = val[i] * mult_4097_vec + K0_vec   ; 1 slot
# stage 1 (xor-shift): s1 = (s0 ^ K1) ^ (s0 >> 19)
alu  t1[i] = val[i] ^ K1                           ; (parallel with t2)
alu  t2[i] = val[i] >> 19
alu  val[i] = t1[i] ^ t2[i]                        ; combine -> s1
# stage 2 (fma): s2 = s1 * 33 + K2
valu multiply_add  val[i] = val[i] * mult_33_vec + K2_vec
# stage 3 (add-shift ^): s3 = (s2 + K3) ^ (s2 << 9)
alu  t1[i] = val[i] + K3
alu  t2[i] = val[i] << 9
alu  val[i] = t1[i] ^ t2[i]                        ; combine -> s3
# stage 4 (fma): s4 = s3 * 9 + K4
valu multiply_add  val[i] = val[i] * mult_9_vec + K4_vec
# stage 5 (xor-shift): s5 = (s4 ^ K5) ^ (s4 >> 16)
alu  t1[i] = val[i] ^ K5
alu  t2[i] = val[i] >> 16
alu  val[i] = t1[i] ^ t2[i]                        ; combine -> v

# --- post-hash: idx update with fma base; branchless ---
alu  t1[i] = val[i] & 1                            ; parity bit
valu multiply_add  t2[i] = idx[i] * two_vec + one_vec  ; idx*2 + 1  (fma: base)
alu  idx[i] = t2[i] + t1[i]                        ; next = base + parity

# --- round 10 (verified uniform wrap round): replace post-hash with ---
# alu  idx[i] = zero     # (i.e. write 0)
# No parity/fma/add steps; val[i] is already v from the hash.
```

Op count per lane per round (matches budget):
- pre-hash XOR: 1 alu
- hash: 12 (3 fma + 9 alu across 3 xor/add-shift stages)
- post-hash idx: 3 alu/valu  (1 mask + 1 fma + 1 add)  [skipped on round 10]
- wrap: 0 (build-time per-round: round 10 picks `idx := 0`)
- val update: 0 (rename: `val[i]` already holds `v` from the hash's final stage)
- ──────
- normal round: 16 ops/lane
- wrap round (10): 13 ops/lane
- average: 15.75 ops/lane

Total kernel: 15.75 × 256 × 16 = 64 512 lane-ops.

# Why val[i] = t0 is correct (live-range argument)
- `val[i]` is read exactly **once** per round (at entry XOR).
- After that XOR, `val[i]`'s old value is dead; the lane's running hash
  (a, s0, s1, ..., s5) can occupy the same scratch word.
- The final stage writes `s5 = v` into `val[i]`; that is exactly the new
  `val` value for the next round. No transfer op.
- Carried round-to-round state is *the same* scratch word as in-hash current.
- Hence no move and no extra register.

# Why `node_val_buf` is shared (not per-lane)
- `node_val` is live from the `load` until consumed by the entry XOR, one
  cycle later. Then dead.
- In sequential mode, only one lane is mid-gather at a time, so a single
  scratch word (addr 18) holding the current gather suffices.
- When vectorized across VLEN=8 lanes, this same word becomes
  `node_val_vec[8]` (one vector of 8 lane values) — still one shared
  region, sized to VLEN. The map degrades to vectorization without any
  address renumbering for other sectors.