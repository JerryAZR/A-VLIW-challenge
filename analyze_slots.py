#!/usr/bin/env python
"""Slot-utilization analyzer for the VLIW scheduler output.

Builds the kernel, extracts the scheduled body bundles, and produces a
matplotlib chart showing per-cycle slot usage by engine type as separate
subplots (one per engine) so nothing overlaps.

Usage:
    python analyze_slots.py                 # default: saves slots.png
    python analyze_slots.py --show          # interactive window
    python analyze_slots.py --picker idx    # try different picker
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from perf_takehome import KernelBuilder
from problem import SLOT_LIMITS


def extract_body_bundles(kb: KernelBuilder):
    """Return the scheduled body bundles (between pause 1 and the vstore
    epilogue), plus the prologue/epilogue cycle counts."""
    pause_idxs = [i for i, instr in enumerate(kb.instrs)
                  if "flow" in instr and instr["flow"][0][0] == "pause"]
    body_start = pause_idxs[0] + 1

    vstore_start = pause_idxs[1]
    for i in range(body_start, pause_idxs[1]):
        instr = kb.instrs[i]
        if "store" in instr and any(s[0] == "vstore" for s in instr["store"]):
            vstore_start = i
            break

    body = kb.instrs[body_start:vstore_start]
    prologue_cyc = body_start
    epilogue_cyc = len(kb.instrs) - vstore_start
    return body, prologue_cyc, epilogue_cyc


def slot_usage_per_cycle(body_bundles):
    """Return a dict[engine -> np.array] of per-cycle slot counts."""
    engines = ["valu", "load", "alu", "flow", "store"]
    n = len(body_bundles)
    data = {eng: np.zeros(n, dtype=int) for eng in engines}
    for i, bundle in enumerate(body_bundles):
        for eng, slots in bundle.items():
            if eng in data:
                data[eng][i] = len(slots)
    return data


def plot(data, caps, title, outpath, show=False):
    n = len(next(iter(data.values())))
    x = np.arange(n)

    engines = ["valu", "load", "alu", "flow"]
    colors = {"valu": "2196F3", "load": "#FF5722",
              "alu": "#4CAF50", "flow": "#FF9800"}
    colors = {"valu": "#2196F3", "load": "#FF5722",
              "alu": "#4CAF50", "flow": "#FF9800"}

    # Two figures: single-cycle bars + 10-cycle rolling-average lines.
    fig, axes = plt.subplots(len(engines), 2, figsize=(22, 10),
                             sharex="col",
                             gridspec_kw={"hspace": 0.35, "wspace": 0.12})

    for row, eng in enumerate(engines):
        vals = data[eng]
        cap = caps.get(eng, 0)

        # Left: single-cycle bars.
        ax = axes[row][0]
        ax.bar(x, vals, width=1.0, color=colors[eng], edgecolor="none")
        ax.axhline(y=cap, color="red", linestyle="--", linewidth=0.8,
                   label=f"cap = {cap}")
        ax.set_ylabel(eng, fontsize=11, fontweight="bold")
        ax.set_ylim(0, cap + 1)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_xlim(0, n)
        if row == 0:
            ax.set_title("Per-cycle", fontsize=12)

        # Right: 10-cycle rolling average line.
        ax2 = axes[row][1]
        if n >= 10:
            window = 10
            rolled = np.convolve(vals, np.ones(window) / window, mode="valid")
            xr = np.arange(window - 1, n)
            ax2.plot(xr, rolled, color=colors[eng], linewidth=1.0)
        else:
            ax2.plot(x, vals, color=colors[eng], linewidth=1.0)
        ax2.axhline(y=cap, color="red", linestyle="--", linewidth=0.8,
                    label=f"cap = {cap}")
        ax2.set_ylim(0, cap + 1)
        ax2.legend(loc="upper right", fontsize=7)
        ax2.set_xlim(0, n)
        if row == 0:
            ax2.set_title("10-cycle rolling average", fontsize=12)

    axes[-1][0].set_xlabel("body cycle", fontsize=11)
    axes[-1][1].set_xlabel("body cycle", fontsize=11)
    fig.suptitle(title, fontsize=13, y=0.98)

    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--picker", default="fma_first",
                        choices=["fma_first", "idx", "random"])
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--out", default="slots.png")
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg")

    kb = KernelBuilder()
    kb.build_kernel(10, 2047, 256, 16)

    body, pro_cyc, epi_cyc = extract_body_bundles(kb)
    print(f"Prologue: {pro_cyc} cyc | Body: {len(body)} cyc | Epilogue: {epi_cyc} cyc")

    data = slot_usage_per_cycle(body)
    caps = {eng: SLOT_LIMITS[eng] for eng in ["valu", "load", "alu", "flow", "store"]}

    total_cyc = len(body)
    for eng in ["valu", "load", "alu", "flow"]:
        used = int(data[eng].sum())
        cap = caps[eng] * total_cyc
        util = 100.0 * used / cap if cap else 0
        idle = int((data[eng] == 0).sum())
        print(f"  {eng:5s}: {used:6d} slots / {cap:6d} cap = {util:5.1f}%  "
              f"({idle} idle cycles)")

    total_per_cyc = sum(data[eng] for eng in data)
    bubbles = int((total_per_cyc < 3).sum())
    no_load = int((data["load"] == 0).sum())
    print(f"\n  Bubble cycles (<3 total slots): {bubbles}")
    print(f"  Cycles with 0 load slots:       {no_load}")

    title = (f"Slot utilization (picker={args.picker}) - "
             f"body {total_cyc} cyc, {pro_cyc}+{epi_cyc} pro/epi")
    plot(data, caps, title, args.out, show=args.show)


if __name__ == "__main__":
    main()
