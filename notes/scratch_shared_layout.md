# Scratch shared/constant sector layout

The 1536-word scratch is split into 5 per-lane planes (`val`/`idx`/`t1`/`t2`/`nv`,
5×256 = 1280 words, regions 0-159) and a **256-word shared/constant sector**
(regions 160-191, addresses 1280-1535). This doc maps the shared sector.

Layout as allocated by `KernelBuilder.build_kernel` (post step 9). Each region
is 8 words. `CONST(v)` = a literal constant; `VAR(name)` = a runtime value.
A uniform vector (vbroadcast / identical 8 lanes) is written `CONST(v)*8` or
`VAR(name)*8` (not repeated 8×). `.` = free.

Allocation order: **planes, then vector section (CONST, VAR), then scalar
section (CONST, VAR)**. Every 8-word vector is 8-aligned (one vector per
region); scalars are packed after.

## Vector section - CONST vectors (regions 160-175, @1280-1407)

| region | addr   | name          | contents |
|--------|--------|---------------|----------|
| 160 | @1280 | mult4097_vec  | CONST(4097)*8         (stage 0 fma multiplier) |
| 161 | @1288 | K0_vec        | CONST(0x7ed55d16)*8   (stage 0 addend) |
| 162 | @1296 | mult33_vec    | CONST(33)*8           (stage 2 fma multiplier) |
| 163 | @1304 | K2_vec        | CONST(0x165667b1)*8   (stage 2 addend) |
| 164 | @1312 | mult9_vec     | CONST(9)*8            (stage 4 fma multiplier) |
| 165 | @1320 | K4_vec        | CONST(0xfd7046c5)*8   (stage 4 addend) |
| 166 | @1328 | K1_vec        | CONST(0xc761c23c)*8   (stage 1 xor const) |
| 167 | @1336 | K3_vec        | CONST(0xd3a2646c)*8   (stage 3 add const) |
| 168 | @1344 | K5_vec        | CONST(0xb55a4f09)*8   (stage 5 xor const) |
| 169 | @1352 | shift19_vec   | CONST(19)*8           (stage 1 shift) |
| 170 | @1360 | shift9_vec    | CONST(9)*8            (stage 3 shift) |
| 171 | @1368 | shift16_vec   | CONST(16)*8           (stage 5 shift) |
| 172 | @1376 | two_vec       | CONST(2)*8            (idx: 2*idx) |
| 173 | @1384 | one_vec       | CONST(1)*8            (idx: +1 / parity mask) |
| 174 | @1392 | zero_vec      | CONST(0)*8            (wrap: idx &= 0) |
| 175 | @1400 | three_vec     | CONST(3)*8            (level-2 select base) |

## Vector section - VAR vectors (regions 176-184, @1408-1479)

| region | addr   | name          | contents |
|--------|--------|---------------|----------|
| 176 | @1408 | forest_p_vec  | VAR(forest_p_vec)*8   (vbroadcast of header `forest_values_p`) |
| 177 | @1416 | tree_preload  | VAR(tree[0..7])       (vload, non-uniform: 8 distinct tree values) |
| 178 | @1424 | tree0_vec     | VAR(tree0_vec)*8      (vbroadcast of tree[0]) |
| 179 | @1432 | tree1_vec     | VAR(tree1_vec)*8      (vbroadcast of tree[1]) |
| 180 | @1440 | tree2_vec     | VAR(tree2_vec)*8      (vbroadcast of tree[2]) |
| 181 | @1448 | tree3_vec     | VAR(tree3_vec)*8      (vbroadcast of tree[3]) |
| 182 | @1456 | tree4_vec     | VAR(tree4_vec)*8      (vbroadcast of tree[4]) |
| 183 | @1464 | tree5_vec     | VAR(tree5_vec)*8      (vbroadcast of tree[5]) |
| 184 | @1472 | tree6_vec     | VAR(tree6_vec)*8      (vbroadcast of tree[6]) |

## Scalar section - CONST scalars (regions 185-186, @1480-1495)

The named literals (`zero`/`eight`/`three`) plus the 13 unnamed
broadcast-source consts (the `scratch_const` sources that the prologue
`vbroadcast`s into the CONST vectors above; each is read once and never again).

| region | addr | word 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|--------|------|--------|---|---|---|---|---|---|---|
| 185 | @1480 | CONST(0) [zero] | CONST(8) [eight] | CONST(3) [three] | CONST(4097) | CONST(0x7ed55d16) | CONST(33) | CONST(0x165667b1) | CONST(9) |
| 186 | @1488 | CONST(0xfd7046c5) | CONST(0xc761c23c) | CONST(0xd3a2646c) | CONST(0xb55a4f09) | CONST(19) | CONST(16) | CONST(2) | CONST(1) |

## Scalar section - VAR scalars (region 187, @1496-1503)

| region | addr | word 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|--------|------|--------|---|---|---|---|---|---|---|
| 187 | @1496 | VAR(rounds) | VAR(n_nodes) | VAR(batch_size) | VAR(forest_height) | VAR(forest_values_p) | VAR(inp_indices_p) | VAR(inp_values_p) | VAR(addr_a) |

## Free (regions 188-191, @1504-1535) - 32 words

| region | addr | contents |
|--------|------|----------|
| 188 | @1504 | . * 8 |
| 189 | @1512 | . * 8 |
| 190 | @1520 | . * 8 |
| 191 | @1528 | . * 8 |

## Summary

- **Used: 224 words** (regions 160-187) = 25 vectors (200w: 16 CONST + 9 VAR)
  + 24 scalars (16 CONST + 8 VAR).
- **Free: 32 words** (regions 188-191, @1504-1535).
- `scratch_ptr = 1504`.

All vectors are 8-aligned (each occupies exactly one region). The 13 unnamed
CONST scalars (region 186 tail + region 187 head) are the broadcast sources;
they exist only because `vbroadcast` reads a scratch addr (no literal-arg form),
so each uniform CONST vector needs a 1-word source scalar.
