// Full kernel reference: what the VLIW program must compute.
// Produces exactly the graded output: mem[inp_values_p .. +batch_size].
//
// Memory image layout (built by build_mem_image in problem.py):
//   mem[0] = rounds
//   mem[1] = n_nodes           (== 2^(height+1) - 1; here 2047)
//   mem[2] = batch_size        (== 256)
//   mem[3] = forest_height     (== 10)
//   mem[4] = forest_values_p   -> tree node values [n_nodes words]
//   mem[5] = inp_indices_p     -> per-lane current idx [batch_size words]
//   mem[6] = inp_values_p      -> per-lane current val [batch_size words]
//
// Graded contract: only mem[inp_values_p .. +batch_size] is checked at end.
// mem[inp_indices_p] is NOT checked. Initial indices are all 0 (known
// statically; no load needed for them).
//
// All arithmetic mod 2^32.

#include <stdint.h>

static inline uint32_t myhash(uint32_t a) {
    a = (a + 0x7ED55D16u) + (a << 12);     // stage 0: a*4097 + K0 (fma)
    a = (a ^ 0xC761C23Cu) ^ (a >> 19);     // stage 1: xor-shift (irreducible)
    a = (a + 0x165667B1u) + (a << 5);      // stage 2: a*33 + K2   (fma)
    a = (a + 0xD3A2646Cu) ^ (a << 9);      // stage 3: add-shift ^ (irreducible)
    a = (a + 0xFD7046C5u) + (a << 3);     // stage 4: a*9 + K4    (fma)
    a = (a ^ 0xB55A4F09u) ^ (a >> 16);    // stage 5: xor-shift (irreducible)
    return a;
}

// The kernel. Operates in-place on the memory image.
void kernel(uint32_t *mem) {
    uint32_t rounds         = mem[0];
    uint32_t n_nodes        = mem[1];
    uint32_t batch_size     = mem[2];
    // uint32_t forest_height = mem[3];   // unused by the walk; needed only for tiling
    uint32_t forest_values_p = mem[4];
    uint32_t inp_indices_p   = mem[5];
    uint32_t inp_values_p    = mem[6];

    // Tree node values and per-lane state. Tree is read-only; kept in `mem`.
    const uint32_t *tree = &mem[forest_values_p];

    // Per-lane working state, kept in registers/cache across all rounds.
    // Initial idx[i] = 0 for all lanes (statically known).
    uint32_t idx[256];   // batch_size
    uint32_t val[256];
    for (uint32_t i = 0; i < batch_size; i++) {
        idx[i] = mem[inp_indices_p + i];   // = 0; load elided if trusted
        val[i] = mem[inp_values_p  + i];
    }

    // Main loop: embarrassingly parallel across lanes.
    for (uint32_t round = 0; round < rounds; round++) {
        for (uint32_t i = 0; i < batch_size; i++) {
            uint32_t node_val = tree[idx[i]];          // gather (per-lane non-contiguous)
            uint32_t v = myhash(val[i] ^ node_val);    // 6-stage hash, the hot loop

            // Branch to left/right child: left if even, right if odd.
            //   idx = 2*idx + 1 + (v & 1)
            uint32_t next = (idx[i] << 1) + 1u + (v & 1u);

            // Wrap to root if past the tree.
            // n_nodes == 2^(height+1) - 1 == 2047 for height 10.
            // idx is in [0, 2*idx+2]; wrap when >= n_nodes.
            idx[i] = (next < n_nodes) ? next : 0u;
            val[i] = v;
        }
    }

    // Commit only the graded output: final values.
    for (uint32_t i = 0; i < batch_size; i++) {
        mem[inp_values_p + i] = val[i];
    }
    // mem[inp_indices_p + i] is NOT written — not graded.
}