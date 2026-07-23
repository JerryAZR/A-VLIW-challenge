# Scratch shared/constant sector - address map

The 1536-word scratch is split into 5 per-lane planes (`val`/`idx`/`t1`/`t2`/`nv`,
regions 0-159) and a **256-word shared/constant sector** (regions 160-191,
@1280-1535). This is the address map for the shared sector only.

Each line is one 8-word region. A const (scalar or vector) is written as its
literal; a var is written as its name. A uniform vector region shows one entry
(= all 8 lanes). `.` = free.

Allocation order: planes, then vector section (CONST, VAR), then scalar section
(CONST, VAR) - so all vectors are 8-aligned (one per region).

```
# CONST vectors (regions 160-175)
160 @1280  4097
161 @1288  0x7ed55d16
162 @1296  33
163 @1304  0x165667b1
164 @1312  9
165 @1320  0xfd7046c5
166 @1328  0xc761c23c
167 @1336  0xd3a2646c
168 @1344  0xb55a4f09
169 @1352  19
170 @1360  9
171 @1368  16
172 @1376  2
173 @1384  1
174 @1392  0
175 @1400  3

# VAR vectors (regions 176-184)
176 @1408  forest_p_vec
177 @1416  tree_preload
178 @1424  tree0_vec
179 @1432  tree1_vec
180 @1440  tree2_vec
181 @1448  tree3_vec
182 @1456  tree4_vec
183 @1464  tree5_vec
184 @1472  tree6_vec

# CONST scalars (regions 185-186, 8 words each)
185 @1480  0  8  3  4097  0x7ed55d16  33  0x165667b1  9
186 @1488  0xfd7046c5  0xc761c23c  0xd3a2646c  0xb55a4f09  19  16  2  1

# VAR scalars (region 187, 8 words)
187 @1496  rounds  n_nodes  batch_size  forest_height  forest_values_p  inp_indices_p  inp_values_p  addr_a

# free (regions 188-191)
188 @1504  . . . . . . . .
189 @1512  . . . . . . . .
190 @1520  . . . . . . . .
191 @1528  . . . . . . . .
```

## Notes

- `scratch_ptr = 1504`. Used: 224 words (25 vectors + 24 scalars). Free: 32.
- `forest_p_vec` is `vbroadcast(forest_values_p)` (a header var). `tree_preload`
  is a non-uniform `vload` of `tree[0..7]`; `tree0_vec..tree6_vec` are
  `vbroadcast` of its lanes 0..6.
- The CONST scalars in regions 185-186 are: the named literals `0`/`8`/`3`
  (used directly by the prologue/idx code) plus 13 unnamed broadcast-source
  consts - each is the `scratch_const` a `vbroadcast` reads to fill the matching
  CONST vector above (e.g. region 185 word 3 `4097` -> region 160). They exist
  only because `vbroadcast` takes a scratch addr (no literal-arg form).
- `mult9_vec` (region 164) and `shift9_vec` (region 170) both broadcast `9`; the
  distinct literal `9` is allocated once (region 185 word 6) and reused.
