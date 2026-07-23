# Scratch shared/constant sector - address map

The 1536-word scratch is split into 5 per-lane planes (`val`/`idx`/`t1`/`t2`/`nv`,
regions 0-159) and a **256-word shared/constant sector** (regions 160-191,
@1280-1535). This is the address map for the shared sector only.

Each line is one 8-word region. A const (scalar or vector) is written as its
literal; a var is written as its name. A uniform vector region shows one entry
(= all 8 lanes). `.` = free.

Allocation order: planes, then vector section (CONST, VAR), then scalar section
(CONST, VAR). All vectors are 8-aligned (one per region). Const vectors are
created by `load const` into their own lane 0 then self-broadcast
(`vbroadcast vec, vec`) - no separate broadcast-source scalar. Scalar uses of a
value read the matching const_vec's lane 0 (e.g. `+ const_vec_0` for +0).

```
# CONST vectors (regions 160-174) - sorted by value; small ones are
# const_vec_<value> (reused across steps, not step-tied), K0..K5 are the
# stage-specific hash addend/xor constants.
160 @1280  0              # const_vec_0    (wrap: idx&=0; scalar +0)
161 @1288  1              # const_vec_1    (idx +1; parity mask; select bit mask)
162 @1296  2              # const_vec_2    (idx 2*idx)
163 @1304  3              # const_vec_3    (level-2 select base)
164 @1312  9              # const_vec_9    (stage4 mult + stage3 shift - shared)
165 @1320  16             # const_vec_16   (stage5 shift)
166 @1328  19             # const_vec_19   (stage1 shift)
167 @1336  33             # const_vec_33   (stage2 mult)
168 @1344  4097           # const_vec_4097 (stage0 mult)
169 @1352  0x7ed55d16     # K0_vec         (stage 0 addend)
170 @1360  0xc761c23c     # K1_vec         (stage 1 xor const)
171 @1368  0x165667b1     # K2_vec         (stage 2 addend)
172 @1376  0xd3a2646c     # K3_vec         (stage 3 add const)
173 @1384  0xfd7046c5     # K4_vec         (stage 4 addend)
174 @1392  0xb55a4f09     # K5_vec         (stage 5 xor const)

# VAR vectors (regions 175-183)
175 @1400  forest_p_vec   # vbroadcast of header forest_values_p
176 @1408  tree_preload   # vload tree[0..7] (non-uniform)
177 @1416  tree0_vec      # vbroadcast of tree[0]
178 @1424  tree1_vec      # vbroadcast of tree[1]
179 @1432  tree2_vec      # vbroadcast of tree[2]
180 @1440  tree3_vec      # vbroadcast of tree[3]
181 @1448  tree4_vec      # vbroadcast of tree[4]
182 @1456  tree5_vec      # vbroadcast of tree[5]
183 @1464  tree6_vec      # vbroadcast of tree[6]

# Scalar section (regions 184-...) - CONST scalar then VAR scalars
184 @1472  8  rounds  n_nodes  batch_size  forest_height  forest_values_p  inp_indices_p  inp_values_p
185 @1480  addr_a  .  .  .  .  .  .  .
186 @1488  .  .  .  .  .  .  .  .
187 @1496  .  .  .  .  .  .  .  .
188 @1504  .  .  .  .  .  .  .  .
189 @1512  .  .  .  .  .  .  .  .
190 @1520  .  .  .  .  .  .  .  .
191 @1528  .  .  .  .  .  .  .  .
```

## Notes

- `scratch_ptr = 1481`. Used: 201 words (24 vectors + 9 scalars). Free: 55.
- The only CONST scalar is `8` (region 184 word 0) - the vload/vstore stride.
  There is no `const_vec_8` because 8 is never used as a vector, so a 1-word
  scalar beats an 8-word vector.
- `const_vec_9` is shared by stage 4 (fma multiplier) and stage 3 (shift
  amount) - the former `mult9_vec`/`shift9_vec` pair merged into one.
- `addr_a` (region 185 word 0) is the lone VAR scalar scratch pointer, reused
  as the vload/vstore address temp in prologue/epilogue.
- 55 words free (regions 185-191, @1481-1535) - ample room for future work
  (e.g. per-group output addresses for epilogue/body overlap).
