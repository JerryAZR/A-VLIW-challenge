"""Tracing simulator: logs every completed instruction with its stable rid,
encoding, the addresses it read (with the values read) and the addresses it
wrote (with the values written).

Usage:
    from trace_sim import TracingMachine
    m = TracingMachine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES,
                       value_trace=vt)
    m.run(); m.run()
    # log written to trace_sim.log (one line per executed slot)

Each log line:
    cycle C  rid R  engine  slot=(op, ...)  read  addr=v ...  write  addr=v ...

The rid is recovered by matching the executed (lowered) slot back to the IR
instruction that produced it, via the stable instruction ids carried on
TaggedSlots. Slots that don't map to an IR instruction (e.g. raw prologue
tuples) get rid=-1.

Design notes:
- Values are captured at the moment the slot executes: reads from
  core.scratch BEFORE the slot's writes commit; writes from the slot's
  scratch_write contribution (which commits at end of the bundle's step).
- Because all slots in a bundle write to scratch_write and commit together,
  a read in the same bundle sees the PRE-bundle scratch (read-before-write
  within a cycle is the scheduler's guarantee, not the sim's).
"""

from problem import Machine, VLEN


def _fmt_pairs(pairs):
    return " ".join(f"{a}={v}" for a, v in pairs)


class TracingMachine(Machine):
    """A Machine that logs every executed slot with rid, encoding, reads, writes."""

    def __init__(self, *args, resolved_body=None, log_path="trace_sim.log", **kwargs):
        super().__init__(*args, **kwargs)
        self._log = open(log_path, "w")
        # rid -> resolved IR instruction, for operand-level read/write capture
        # that reuses ir.py's single source of ISA operand truth (reads()/
        # writes()) instead of re-deriving layouts from slot shapes.
        self._by_rid = {i.rid: i for i in resolved_body} if resolved_body else {}

    def close(self) -> None:
        """Flush and close the trace log."""
        self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- per-engine wrappers that log around the real op -------------------

    def _log_slot(self, core, engine, slot, read_pairs, write_pairs):
        # The slot is a TaggedSlot carrying the stable instruction id
        # end-to-end (assigned at construction; the scheduler's tagged_lower()
        # attaches it). Plain prologue tuples have no tag -> rid=-1.
        rid = getattr(slot, "rid", -1)
        self._log.write(
            f"cycle {self.cycle:6d}  rid {rid:6d}  {engine:5s}  {slot}\n"
            f"          read  {_fmt_pairs(read_pairs)}\n"
            f"          write {_fmt_pairs(write_pairs)}\n"
        )

    # -- per-slot logging -----------------------------------------------------
    # step() below iterates the bundle itself (rather than calling the base
    # Machine.step) for one reason: the base unpacks each slot as ``fn(core,
    # *slot)``, which re-packs a TaggedSlot into a plain tuple and loses its
    # rid. Iterating here keeps the original TaggedSlot so its rid reaches the
    # log. Execution still goes through the base engine fns (alu/valu/load/
    # store/flow) - those are NOT duplicated - and the commit mirrors the base.

    def step(self, instr, core):
        # Snapshot scratch so each slot's reads reflect pre-bundle state
        # (read-before-write within a cycle is the scheduler's guarantee).
        pre = list(core.scratch)
        engine_fns = {"alu": self.alu, "valu": self.valu, "load": self.load,
                      "store": self.store, "flow": self.flow}
        self.scratch_write = {}
        self.mem_write = {}
        for name, slots in instr.items():
            if name == "debug":
                # Delegate the debug-oracle handling to the base class logic.
                self._run_debug(slots, core)
                continue
            fn = engine_fns[name]
            for slot in slots:
                scratch_before = dict(self.scratch_write)
                mem_before = dict(self.mem_write)
                reads = self._capture_reads(pre, slot)
                fn(core, *slot)
                writes = ([(a, v) for a, v in self.scratch_write.items()
                           if a not in scratch_before]
                          + [(f"mem[{a}]", v) for a, v in self.mem_write.items()
                             if a not in mem_before])
                self._log_slot(core, name, slot, reads, writes)
        for addr, val in self.scratch_write.items():
            core.scratch[addr] = val
        for addr, val in self.mem_write.items():
            self.mem[addr] = val
        del self.scratch_write
        del self.mem_write

    def _run_debug(self, slots, core):
        """The debug-oracle (compare/vcompare) handling, kept in lockstep with
        the base Machine.step's debug block."""
        if not self.enable_debug:
            return
        for slot in slots:
            if slot[0] == "compare":
                loc, key = slot[1], slot[2]
                ref = self.value_trace[key]
                res = core.scratch[loc]
                assert res == ref, f"{res} != {ref} for {key} at pc={core.pc}"
            elif slot[0] == "vcompare":
                loc, keys = slot[1], slot[2]
                ref = [self.value_trace[key] for key in keys]
                res = core.scratch[loc : loc + VLEN]
                assert res == ref, (
                    f"{res} != {ref} for {keys} at pc={core.pc} loc={loc}")

    def _expand(self, pre, reg_ids):
        """Expand (addr, is_vec) RegIds to [(addr, value)] using pre-bundle
        scratch. A vector RegId covers addr..addr+VLEN-1; a scalar covers addr.
        This is the one place that interprets the RegId shape - the operand
        *identity* comes from ir.py's reads()/writes(), not re-derived here."""
        out = []
        for addr, is_vec in reg_ids:
            n = VLEN if is_vec else 1
            for j in range(n):
                a = addr + j
                out.append((a, pre[a] if 0 <= a < len(pre) else None))
        return out

    def _capture_reads(self, pre, slot):
        """Return [(addr, value)] the slot reads, using pre-bundle scratch.

        Driven by the IR instruction the slot's rid points to (ir.py owns the
        operand semantics); falls back to nothing for untagged prologue slots."""
        rid = getattr(slot, "rid", -1)
        instr = self._by_rid.get(rid)
        if instr is None:
            return []
        return self._expand(pre, instr.reads())
