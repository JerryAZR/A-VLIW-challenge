// Reference: naive myhash, taken literally from problem.py HASH_STAGES.
// All arithmetic mod 2^32 (32-bit unsigned wraparound).
// K constants are stage-fixed (baked in HASH_STAGES at build time).
// Input `a` is 30-bit (val ^ node_val); output is full 32-bit.
//
// Stage shape: a = op2( op1(a, K_i) , op3(a, shift_i) )
//   - op1: + or ^  with constant K_i
//   - op3: << or >> by a fixed shift_i
//   - op2: + or ^  combining the two
//
// Literal op count: 3 binary ops/stage x 6 stages = 18 ops.
// Plus 1 op (val ^ node_val) before entry => 19 ops/lane/round total hash work.

#include <stdint.h>

uint32_t myhash(uint32_t a) {
    // stage 0:  a = (a + K0) + (a << 12)
    a = (a + 0x7ED55D16u) + (a << 12);
    // stage 1:  a = (a ^ K1) ^ (a >> 19)
    a = (a ^ 0xC761C23Cu) ^ (a >> 19);
    // stage 2:  a = (a + K2) + (a << 5)
    a = (a + 0x165667B1u) + (a << 5);
    // stage 3:  a = (a + K3) ^ (a << 9)
    a = (a + 0xD3A2646Cu) ^ (a << 9);
    // stage 4:  a = (a + K4) + (a << 3)
    a = (a + 0xFD7046C5u) + (a << 3);
    // stage 5:  a = (a ^ K5) ^ (a >> 16)
    a = (a ^ 0xB55A4F09u) ^ (a >> 16);
    return a;
}

// Per-lane per-round work (literal naive form):
//   val ^= node_val;           // 1 op
//   val = myhash(val);         // 18 ops
//   idx = 2*idx + 1 + (val&1);// ~3 ops (shift, mask, add)
//   idx = (idx >= n) ? 0 : idx;// ~2 ops (branchless)
//                              // ----
//                              // ~24 ops/lane/round