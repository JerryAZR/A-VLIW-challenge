# The VLIW Challenge Machine — Architecture & ISA Reference

Grounded in `problem.py` (frozen copy at `tests/frozen_problem.py` is what the
submission harness uses). All arithmetic is **mod 2³²** (32-bit unsigned).

## Machine model

A single **in-order VLIW + SIMD core**. No caches, no branch prediction, no
out-of-order, no latency model — every operation completes in the cycle it
issues. The only notion of time is the simulator's `cycle` counter.

Core state:
- `pc` — index into the program (list of instruction bundles)
- `scratch` — **1536 32-bit words**: the register file, constant cache,
  software-managed cache, and vector "registers" are all this one flat space.
  Vector registers are just 8 contiguous scratch words.
- `state` — RUNNING / STOPPED

External `mem` — flat 32-bit word array holding the problem image. Reachable
only via `load` / `store` slots. This is where tree, batch, and output live.

`N_CORES = 1` (multicore intentionally disabled; `coreid` slot is inert).

## Program format

`program: list[Instruction]` where `Instruction = dict[Engine, list[slot]]`.
Each bundle executes in **one cycle**: every engine fires all its slots in
parallel. PC advances by 1 unless a `flow` slot overrides it.

### The one execution rule: read-before-write

Within a cycle, all slots read from the **pre-cycle** scratch/mem state; all
writes accumulate into buffers and commit **at the end** of the cycle:

```python
scratch_write, mem_write = {}, {}
for name, slots in instr.items():
    for slot in slots: ENGINE_FNS[name](core, *slot)   # writes -> buffers
for a,v in scratch_write: core.scratch[a] = v
for a,v in mem_write:    self.mem[a] = v
```

Consequence: a producer and its consumer may **not** be in the same bundle —
the consumer would read the old value. Minimum 1-cycle gap between dependent
ops. (This is the only "latency" in the machine; every op is latency-1.)

Multiple writes to the same scratch addr in one cycle: dict-last-wins; don't
rely on it.

### Cycle accounting

A bundle costs 1 cycle **iff** it contains any non-`debug` engine.
Debug-only bundles are **0 cycles** (and `debug` is disabled at grading
anyway). So debug checkpoints are free; use them liberally while developing.

## Per-engine ISA

Slot limits per cycle: **alu 12 · valu 6 · load 2 · store 2 · flow 1 · debug 64**.

Every operand is a scratch address **except**: `const`'s immediate literal,
`jump`/`cond_jump` targets (program indices), `add_imm`'s immediate. For most
slots the first operand is the destination.

### `alu` — scalar 32-bit, 12 slots/cycle

`("op", dest, a1, a2)` → `dest = (a1 OP a2) mod 2³²`. Binary only.

| op | |
|---|---|
| `+` `-` `*` `//` `cdiv` `^` `&` `\|` `<<` `>>` `%` `<` `==` | arithmetic/logic |
| `cdiv` | `(a+b-1)//b` (ceil div) |
| `<` `==` | 0/1 results |

**No fused ops on `alu`.** No `multiply_add`, no `add_imm` (that's on `flow`),
no rotate, no select, no three-operand add. Just 13 binary integer ops.

### `valu` — SIMD over VLEN=8, 6 slots/cycle, three forms

| form | semantics |
|---|---|
| `("vbroadcast", dest, src)` | `dest[i] = scratch[src]` for i ∈ [0,8) — broadcast a scalar |
| `("multiply_add", dest, a, b, c)` | `dest[i] = a[i]*b[i] + c[i]` mod 2³² — **the only fused op in the ISA** |
| `("op", dest, a1, a2)` | elementwise any of the 13 alu ops over 8 lanes |

`multiply_add` is **`valu`-only** — no equivalent on `alu`. On `alu`,
`a*b + c` costs 2 slots (`*` then `+`).

#### Using `valu` as a scalar fma unit (verified empirically)

`multiply_add` always operates on all 8 lanes. To use it as a 1-lane scalar
op: place the real value in **lane 0** of an 8-word vector region, broadcast
the constant multiplier and addend into full 8-lane vector regions, fire
`multiply_add`. Lane 0 = `val·mult + K` (correct); lanes 1–7 = junk (ignore).
Cost: 1 valu slot + 8-word scratch regions (constants amortized over all
hashes). Verified against the simulator: lane 0 correct, junk lanes ignored.

### `load` — mem → scratch, 2 slots/cycle

| form | semantics |
|---|---|
| `("load", dest, addr)` | `dest = mem[scratch[addr]]` — one word, addr from scratch |
| `("load_offset", dest, addr, off)` | `dest+off = mem[scratch[addr+off]]` — `off` is a literal int |
| `("vload", dest, addr)` | `dest+i = mem[scratch[addr]+i]` for i ∈ [0,8) — **contiguous 8-word fetch** |
| `("const", dest, val)` | `dest = val` — literal baked at build time |

**No scatter/gather.** `vload` is contiguous-only. Non-contiguous per-lane
gathers require scalar `load` slots (≤2/cycle) — the central gather tension.

### `store` — scratch → mem, 2 slots/cycle

| form | semantics |
|---|---|
| `("store", addr, src)` | `mem[scratch[addr]] = scratch[src]` |
| `("vstore", addr, src)` | 8 contiguous words from `src` to `mem[scratch[addr]+0..7]` |

The graded output is `mem[inp_values_p : inp_values_p + batch_size]` — checked
at program end. `inp_indices` is **not** checked.

### `flow` — 1 slot/cycle (the binding constraint for branchy code)

| form | semantics |
|---|---|
| `("select", dest, cond, a, b)` | `dest = cond!=0 ? a : b` (scalar predicated move) |
| `("vselect", dest, cond, a, b)` | same, 8 lanes — predicate-driven vector move |
| `("add_imm", dest, a, imm)` | `dest = a + imm` — saves a const + alu, but competes for the 1 flow slot |
| `("cond_jump", cond, addr)` | if `scratch[cond]!=0` → `pc = addr` (absolute) |
| `("cond_jump_rel", cond, off)` | relative variant (PC already incremented before flow runs) |
| `("jump", addr)` | unconditional `pc = addr` |
| `("jump_indirect", addr)` | `pc = scratch[addr]` — computed branch |
| `("halt",)` | stop the core |
| `("pause",)` | **disabled at grading** (`enable_pause=False`) — do not rely on it |
| `("coreid", dest)` | `dest = core.id` (always 0) |
| `("trace_write", val)` | append `scratch[val]` to trace_buf (dev only) |

Only **1 flow slot/cycle total.** A loop body with a branch *and* a predicate
select costs ≥2 cycles. ⇒ Branchless math (compute parity/wrap arithmetically
via alu/valu) is a first-class optimization lever.

### `debug` — 64 slots/cycle, 0 cycles, disabled at grading

| form | semantics |
|---|---|
| `("compare", loc, key)` | assert `scratch[loc] == value_trace[key]` (dev oracle) |
| `("vcompare", loc, [keys])` | same, 8 lanes |

Free in both cycle accounting and at grading. Sprinkle to bracket suspect
optimizations; the assert pinpoints divergence. Leaving them in committed code
costs nothing.

## The problem (what `build_kernel` must compute)

For each of `batch_size=256` independent lanes, run `rounds=16` iterations:
```
node_val = tree.values[idx]          # gather, per-lane non-contiguous
val      = myhash(val ^ node_val)    # 6-stage 32-bit hash (the hot loop)
idx      = 2*idx + (2 if val&1 else 1)   # left if even, right if odd
idx      = 0 if idx >= n_nodes else idx  # wrap to root
```
`tree.values` has `2^(height+1)-1 = 2047` nodes. Initial `idx[i] = 0` for all
lanes (known statically — no load needed). Initial `val[i]` random 30-bit.

### Graded contract (the only thing checked)

```python
machine.mem[inp_values_p : inp_values_p + batch_size] == ref_mem[...]
```
Final `values` region only. Idx not checked. Read whatever, write whatever
scratch; only the final output vector must be correct.

## `myhash` — algebraic reference

```c
uint32_t myhash(uint32_t a) {                  // all mod 2^32
    a = (a + 0x7ED55D16u) + (a << 12);          // stage 0
    a = (a ^ 0xC761C23Cu) ^ (a >> 19);          // stage 1
    a = (a + 0x165667B1u) + (a << 5);           // stage 2
    a = (a + 0xD3A2646Cu) ^ (a << 9);           // stage 3
    a = (a + 0xFD7046C5u) + (a << 3);           // stage 4
    a = (a ^ 0xB55A4F09u) ^ (a >> 16);          // stage 5
    return a;
}
```
Stage shape: `a = op2( op1(a, K_i) , op3(a, shift_i) )`. Literal = **3 ops/stage
× 6 = 18 ops**.

### Verified reductions (numerically confirmed vs `myhash`)

- **Stages 0, 2, 4** (`+K`/`<<K` combine with `+`): `(a+K) + (a<<s) = a·(1+2^s) + K`
  → **1 `multiply_add` slot each** (3 ops → 1). Multipliers:
  4097, 33, 9.
- **Stages 1, 3, 5** (xor-shift or add-shift with `^` combine): irreducible at
  **3 slots each** (2 parallel transforms of `a`, then `^` combine). Stage 3
  looks like it could be fma (`+K` then `<<`) but the combine is `^`, not `+`,
  so `multiply_add` does not apply.
- **Result: 12 slots/lane/round for `myhash`** (3×1 fma + 3×3). Verified
  bit-exact against `myhash` on 300k random inputs.

### Verified non-reductions (ruled out, with evidence)

- `(a^b)+c == a^(b+c)`? **No** (carries; ~99.98% mismatch on 200k trials).
  ⇒ No fusing a `+K` across a `^`.
- Fusing `+K2` across stage-1's XOR/shift in any arm: **No** (all variants fail
  ~98%+). Carries don't commute across XOR.
- Whole `myhash` as a single GF(2)-affine map? **No** — stage 0's `+K0` makes it
  non-affine (100% violations in random trials; carries are GF(2)-nonlinear).
  The pure xor-shift stages (1, 5) are affine in isolation but are not
  adjacent, so they can't compose across the non-affine stages between them.
  Additionally, the ISA has no bit-matrix-multiply instruction, so a 32×32
  GF(2) matrix-vector product would cost ~512 ops (not fewer). ⇒ GF(2) is a
  dead end on this hardware.

## Roofline (current best estimate)

Per-lane-per-round budget (verified minimums):
| item | slots/lane/round |
|---|---|
| `val ^ node_val` (entry) | 1 |
| `myhash` (reduced, 12-slot) | 12 |
| branchless idx update + wrap | ~5 (loose; floor ~2–3 with care) |
| **total** | **~18** |

Over 256 lanes × 16 rounds = 4096 lane-rounds:

**Memory floor (bytes):** tree 2047 + init vals 256 = 2301 words in (9204 B)
+ 256 words out (1024 B). At `vload`/`vstore` = 32 B/slot × 2 slots/cyc =
64 B/cyc ⇒ **~144 cyc in + 16 cyc out ≈ 160 cyc total memory.** (Scratch is
1536 < 2047, so the tree doesn't fit whole — forces streaming/tiling, not a
bandwidth wall but a capacity constraint.)

**Compute floor (slots):** 18 lane-slots × 4096 = 73 728 lane-slots.
- `valu` only (6 slots×8 lanes = 48 lane-slots/cyc): **1536 cyc**
- `alu`+`valu` idealized (also donate alu's 12 lane-slots/cyc → 60/cyc):
  **1228 cyc** — but optimistic: `alu` has no fma (linear stages cost 2 alu
  slots each there, vs 1 valu slot), and `alu` is 8× less slot-efficient for
  per-lane work; realistically ~1280–1400 cyc.

**Flow floor:** 1 slot/cyc ⇒ branchless math mandatory; loop control ~16 cyc.

**Combined realistic floor ≈ ~1280–1600 cyc** for the independent-hashes
model. The Opus-4.5 score of 1487 sits in this band. Sub-1k requires either
≤11 slots/lane/round (below the verified 12-slot hash minimum + overhead) or
fewer than 4096 hashes (a structural dedup lever not yet identified).

## Baseline empirical profile (PMU, `pmu.py`)

Baseline cycle count: **147 734**.

| engine | slot fires | cap/cyc | util% |
|---|---|---|---|
| alu | 118 784 | 12 | 6.7% |
| valu | 0 | 6 | 0.0% |
| load | 12 564 | 2 | 4.3% |
| store | 8 192 | 2 | 2.8% |
| flow | 8 194 | 1 | 5.5% |

Baseline wastes (one-slot-per-bundle, `k=1` everywhere in histograms):
- Stores 8192 but only 256 needed (stores idx+val every round; idx not graded).
- Loads 12 564 but ~2303 necessary (re-reads idx/val from mem every round).
- Flow 8194 but ~16 necessary (two `select`s per lane-round — replaceable by
  branchless math).
- `alu` is 6.7% utilized; `valu` idle.

## Tooling

- `pmu.py` — `InstrumentedMachine` subclasses `problem.Machine`, counts slot
  fires / op-name breakdown / per-cycle histograms without touching the frozen
  simulator. Run: `python pmu.py`.
- `python perf_takehome.py Tests.test_kernel_cycles` — print cycle count.
- `python perf_takehome.py Tests.test_kernel_trace` then `python watch_trace.py`
  — Perfetto visualization of slot packing / dataflow.
- `python tests/submission_tests.py` — the real gate (correctness + cycle
  ladder). **Do not modify `tests/`.**
- `.pi/sanity.toml` — denies all writes under `tests/` (anti-cheat guard).