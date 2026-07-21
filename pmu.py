"""
A tiny PMU (performance monitoring unit) for the VLIW challenge machine.

Subclasses problem.Machine and overrides each engine method to count slot fires
and op-name breakdowns. Does NOT edit problem.py or tests/ — the submission
harness uses its own Machine, untouched. This is a profiling tool only.

Usage:
    from pmu import InstrumentedMachine, profile_baseline
    m = profile_baseline()      # build + run baseline through PMU
    m.report()                  # print counters
    m.histogram()               # print per-cycle slot-usage histograms
"""

from collections import defaultdict
import random

from problem import (
    Machine,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    Tree,
    Input,
    build_mem_image,
)
from perf_takehome import KernelBuilder


class InstrumentedMachine(Machine):
    """Machine subclass that counts every slot fire, per engine and per op name."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-engine total slot fires (one count per slot executed).
        self.slot_fires = defaultdict(int)          # engine -> int
        # Per-(engine, op_name) breakdown.
        self.op_fires = defaultdict(lambda: defaultdict(int))  # engine -> op -> int
        # Lane-op equivalent: valu slots count as VLEN lane-ops.
        self.lane_ops = defaultdict(int)            # engine -> lane-op count
        # Per-cycle slot usage: hist[engine][k] = number of cycles that engine
        # fired exactly k slots.
        self.cycle_hist = defaultdict(lambda: defaultdict(int))
        # Scratch for the current bundle's counts (reset each step).
        self._bundle_fires = defaultdict(int)

    # --- engine overrides: count, then delegate ---
    def alu(self, core, op, dest, a1, a2):
        self._bundle_fires["alu"] += 1
        self.op_fires["alu"][op] += 1
        return super().alu(core, op, dest, a1, a2)

    def valu(self, core, *slot):
        self._bundle_fires["valu"] += 1
        self.op_fires["valu"][slot[0]] += 1
        return super().valu(core, *slot)

    def load(self, core, *slot):
        self._bundle_fires["load"] += 1
        self.op_fires["load"][slot[0]] += 1
        return super().load(core, *slot)

    def store(self, core, *slot):
        self._bundle_fires["store"] += 1
        self.op_fires["store"][slot[0]] += 1
        return super().store(core, *slot)

    def flow(self, core, *slot):
        self._bundle_fires["flow"] += 1
        self.op_fires["flow"][slot[0]] += 1
        return super().flow(core, *slot)

    # --- step override: record per-cycle usage, then delegate ---
    def step(self, instr, core):
        self._bundle_fires = defaultdict(int)
        super().step(instr, core)
        # After step, fold this bundle's per-engine counts into histograms.
        # (debug-only bundles count as 0 usage of compute engines and don't
        #  increment cycle, so they correctly register as 0-fire cycles here
        #  only if they actually had non-debug work — match Machine.run rule.)
        for name in SLOT_LIMITS:
            if name == "debug":
                continue
            self.cycle_hist[name][self._bundle_fires[name]] += 1
        # Accumulate totals (do this here so debug-disabled runs still count).
        for name, k in self._bundle_fires.items():
            self.slot_fires[name] += k
            lane = k * VLEN if name == "valu" else k
            self.lane_ops[name] += lane

    # --- reporting ---
    def report(self):
        print("=" * 64)
        print("PMU report")
        print("=" * 64)
        print(f"cycles                    : {self.cycle}")
        print(f"VLEN                      : {VLEN}")
        total_lane_ops = sum(self.lane_ops.values())
        print(f"total lane-ops (incl alu) : {total_lane_ops}")
        print()
        print(f"{'engine':<8} {'fires':>10} {'lane-ops':>12} "
              f"{'cap/cyc':>9} {'util%':>7}")
        print("-" * 50)
        for name in ["alu", "valu", "load", "store", "flow", "debug"]:
            cap = SLOT_LIMITS[name]
            fires = self.slot_fires[name]
            lane = self.lane_ops[name]
            util = 100.0 * fires / (self.cycle * cap) if self.cycle and cap else 0
            print(f"{name:<8} {fires:>10} {lane:>12} {cap:>9} {util:>6.1f}%")
        print()
        print("op-name breakdown:")
        for name in ["alu", "valu", "load", "store", "flow"]:
            ops = self.op_fires[name]
            if not ops:
                continue
            items = sorted(ops.items(), key=lambda kv: -kv[1])
            print(f"  {name}:")
            for op, c in items:
                lane = c * VLEN if name == "valu" else c
                print(f"    {op:<14} {c:>8}  (lane-ops: {lane})")

    def histogram(self, top=8):
        print()
        print("per-cycle slot-usage histogram (count of cycles with k slots fired):")
        for name in ["alu", "valu", "load", "store", "flow"]:
            h = self.cycle_hist[name]
            if not h:
                continue
            cap = SLOT_LIMITS[name]
            print(f"  {name} (cap {cap}):")
            for k in range(0, min(cap, top) + 1):
                c = h.get(k, 0)
                bar = "#" * min(c // max(1, self.cycle // 50), 40)
                print(f"    k={k:<2} {c:>7}  {bar}")
            above = sum(v for kk, v in h.items() if kk > top)
            if above:
                print(f"    (>{top}: {above})")


def profile_baseline(forest_height=10, rounds=16, batch_size=256, seed=42,
                     trace=False):
    """Build and run the baseline KernelBuilder kernel through the PMU."""
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    m = InstrumentedMachine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        trace=trace,
    )
    # Match grading settings so counts reflect the graded program path.
    m.enable_pause = False
    m.enable_debug = False
    m.run()
    return m


if __name__ == "__main__":
    m = profile_baseline()
    m.report()
    m.histogram()