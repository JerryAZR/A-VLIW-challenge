# Scratch Memory Map (1536 words) — v2, per-lane private temps

> **SUPERSEDED** by `scratch_map_canonical.md` (v3a, the implemented layout).
> This file is retained for history only: it describes the v2 sequential
> kernel's `[t0,t1,t2]` triple layout (768 words of per-lane temps), the
> runtime-temp sector 2 slots (`addr_b`, `node_val_b`, `cnt_round`,
> `cnt_lane`) that were never used, and the shared-single-pair stage-scratch
> idea. v3a replaced all of this with the 256-shared + 5-per-lane SoA rule.

---

Key insight vs. v1: with 864 words free and a verified live-range structure
(t0 = current hash value, always live; t1, t2 = stage scratch, live only
within a xor/add-shift stage and dead between stages), we give each lane its
own private temp triple [t0, t1, t2]. This eliminates all rename hazards and
makes the per-lane dependency flow read like a normal program.

Lane state `val[256]`, `idx[256]` and the per-lane temps persist in scratch
across rounds by default — no loop restructuring for state persistence is
needed; scratch words hold values across cycles unless overwritten.

## Sector 1 — Scalar header + constants  [0..15]  (16 words)

| addr | name              | source |
|------|-------------------|--------|
| 0    | rounds            | `load` from mem[0] |
| 1    | n_nodes           | `load` from mem[1] |
| 2    | batch_size        | `load` from mem[2] |
| 3    | forest_height     | `load` from mem[3] |
| 4    | forest_values_p   | `load` from mem[4] |
| 5    | inp_indices_p     | `load` from mem[5] |
| 6    | inp_values_p      | `load` from mem[6] |
| 7    | zero (const 0)    | `load const` |
| 8    | one  (const 1)    | `load const` |
| 9    | two  (const 2)    | `load const` |
| 10   | shift_19          | `load const` |
| 11   | shift_9           | `load const` |
| 12   | shift_16          | `load const` |
| 13   | K1                | `load const` |
| 14   | K3                | `load const` |
| 15   | K5                | `load const` |

Stages 0/2/4 (fma) use broadcast vector constants (Sector 3).
Stages 1/3/5 (xor-shift, elementwise ops) use scalar constants here — when
the kernel is sequential these go in lane 0 of a vector op on valu (which
equals scalar behavior on lane 0); when vectorized, they are broadcast
into vectors too (then the "scalar constant" becomes redundant). For now,
scalar constants are used as lane-0 operands of value ops the same way
`tmp_val_vec` was.

## Sector 2 — Runtime scratch scalars  [16..31]  (16 words)

| addr | name          | use |
|------|---------------|-----|
| 16   | addr_a        | mem pointer for load slot A (gather prefetch) |
| 17   | addr_b        | mem pointer for load slot B (gather prefetch, 2nd port) |
| 18   | node_val_a    | gathered tree value via load A |
| 19   | node_val_b    | gathered tree value via load B |
| 20   | cnt_round     | outer-loop counter (when looping) |
| 21   | cnt_lane      | inner-loop counter (when looping) |
| 22.. | (reserved)    | future pipeline state |

## Sector 3 — Vector constants  [32..95]  (64 words = 8 vectors)

Broadcast once at prologue via `vbroadcast` from a scalar `const`; reused
read-only by all hashes.

| addr     | name          | use |
|----------|---------------|-----|
| 32..39   | mult_4097_vec | stage 0 fma: a*4097 + K0 |
| 40..47   | K0_vec        | stage 0 fma addend |
| 48..55   | mult_33_vec   | stage 2 fma: a*33 + K2 |
| 56..63   | K2_vec        | stage 2 fma addend |
| 64..71   | mult_9_vec    | stage 4 fma: a*9 + K4 |
| 72..79   | K4_vec        | stage 4 fma addend |
| 80..87   | two_vec       | idx fma: idx*2 + 1 (`base`) |
| 88..95   | one_vec       | idx fma addend (the `+1`) |

## Sector 4 — Lane state (resident)  [96..607]  (512 words)

Stored as 32 contiguous vectors × 8 lanes each — serves both sequential
(literal address `val[96 + i]`) and vectorized (`valu` over vector
`k = val[96 + 8k .. +8]`) access.

| addr      | name   | init |
|-----------|--------|------|
| 96..351   | val[256] | prologue `vload` from `mem[inp_values_p ..]` (16 cyc) |
| 352..607  | idx[256] | initial = 0; scratch zero-initialized → **no init needed** |

Never rewritten to mem during the rounds. On round 10 (the verified
uniform wrap round), `idx[k]` is set to 0 for all lanes (build-time-known
special path). The final `val[256]` is `vstore`'d to `mem[inp_values_p]`
once at the end.

## Sector 5 — Per-lane private temp triples  [608..1375]  (768 words)

Three temps per lane, never shared across lanes. Layout: lane `i`'s temps
at base `608 + 3*i`, contiguous (so future vectorization sees them as
vectors of t0/t1/t2 across 8 lanes — useful when we vectorize!). 256
lanes × 3 temps = 768 words.

| lane i offset | name | live range |
|---------------|------|------------|
| 608 + 3*i + 0 | t0[i] | entire hash + post-hash (current hash value); always live |
| 608 + 3*i + 1 | t1[i] | within a xor/add-shift stage only (one of the parallel transforms) |
| 608 + 3*i + 2 | t2[i] | within a xor/add-shift stage only (the other transform) |

`t1[i]`, `t2[i]` are dead immediately after each stage's combine writes
back to `t0[i]`. They're reused as scratch for the next stage. No
cross-lane sharing, so no rename hazards; each lane's temps are addressed
by a compile-time-literal chain `608 + 3*i + (0|1|2)`.

Why this is worth 768 words: compute logic now reads as a normal program
(register nitanalog), with no "current_s reuse via overwrite" tricks.
Eliminates an entire class of read-before-write bugs we'd otherwise have
to debug. And the contiguous layout means that when we vectorize, the
same address logic wraps clean: vector k's `t0` across lanes becomes
`valu` over `[608 + 24*k .. +8]`.

## Sector 6 — Free space  [1376..1535]  (160 words)

Reserved for future:
- Deeper in-flight pipeline during vectorization (each additional in-flight
  vector needs ~32 words). 5 in-flight vectors fit in 160.
- Scratch ad-hoc during debugging.

## Totals

| sector | words | % of 1536 |
|---|---|---|
| 1 header+consts | 16 | 1% |
| 2 runtime temps | 16 | 1% |
| 3 vec consts | 64 | 4% |
| 4 lane state | 512 | 33% |
| 5 per-lane temps | 768 | 50% |
| **used** | **1376** | **90%** |
| 6 free | 160 | 10% |

# Per-lane dependency flow (with the above temps)

For lane `i`, one normal (non-wrap) round:

```
# --- pre-hash ---
node_val = mem[forest_values_p + idx[i]]    ; gather (load slot, prefetched)
t0[i] = val[i] ^ node_val                   ; entry XOR (alu) — a = hash input

# --- hash: 12 slots, using t0[i] as current, t1[i]/t2[i] as stage scratch ---
t0[i] = multiply_add(t0[i], mult4097_vec, K0_vec)     ; stage 0 (fma)
t1[i] = t0[i] ^ K1     ; t2[i] = t0[i] >> 19           ; stage 1 parallel (alu)
t0[i] = t1[i] ^ t2[i]                                 ; stage 1 combine (alu)
t0[i] = multiply_add(t0[i], mult33_vec, K2_vec)       ; stage 2 (fma)
t1[i] = t0[i] + K3     ; t2[i] = t0[i] << 9           ; stage 3 parallel (alu)
t0[i] = t1[i] ^ t2[i]                                 ; stage 3 combine (alu)
t0[i] = multiply_add(t0[i], mult9_vec, K4_vec)        ; stage 4 (fma)
t1[i] = t0[i] ^ K5     ; t2[i] = t0[i] >> 16          ; stage 5 parallel (alu)
t0[i] = t1[i] ^ t2[i]                                 ; stage 5 combine (alu) -> t0 = v

# --- post-hash: idx update, branchless via fma ---
t1[i] = t0[i] & 1                                     ; parity bit  (alu)
                                                       ; (parallel-ish with above)
t2[i] = multiply_add(idx[i], two_vec, one_vec)        ; idx*2 + 1 (fma) -- "base"
idx[i+1] = t2[i] + t1[i]                               ; next idx (= base + parity)
val[i] = t0[i]                                        ; new val = v  (free scratch write)

# --- on round 10 (wrap): skip t2/iidx update; idx[i] = 0 (literal) ---
```

Op count per lane per round (matches our reduced budget):
  pre-hash XOR : 1
  hash         : 12  (3 fma stages + 3 xor/add-shift × 3 slots)
  post-hash idx: 3   (parity mask + base fma + add)  [skipped on wrap round]
  wrap         : 0   (build-time per-round: round 10 path picks 0)
  val update   : 0   (rename: alu dest = val[i])
  ──────────────
  normal round : 16
  wrap round   : 13  (no idx update)
```

## Why per-lane temps are safe (no dependence hazards)

Because each lane has its own t0/t1/t2, there is no within-bundle aliasing
risk when we eventually pack multiple lanes into the same bundle (VLIW).
Lane A writing t0[A] and lane B writing t0[B] are independent scratch
addresses; the read-before-write rule only constrains a single slot to
not read an address written elsewhere in the same bundle. With per-lane
temps, two lanes in the same bundle at the same hash stage never write
the same address, so no stalls. This is what makes the packing easier —
the map is "DAG-ready" by construction.