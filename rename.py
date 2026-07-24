"""Rename engine: owns all scratch space; translates symbolic IR into
resolved IR.

    re = RenameEngine(pinned_syms)   # allocate & pin, seed free lists
    ... emit symbolic IR ...
    resolved = re.rename(instrs)     # symbolic -> resolved, one pass

Two symbol classes:

  - **pinned** - declared up front in the constructor's symbol list.
    Always resolves to its pinned address; writes land in place. For
    carried state (val/addr planes), constants, header vars, out_addr.
  - **temp** - never declared; born implicitly on its first whole-symbol
    write (or a ``Refresh`` directive). Every whole-symbol write re-homes
    the symbol: the old home is freed (provably dead - post-rewrite reads
    resolve to the new home) and a fresh home is dequeued from the free
    pool (rename-on-write). False dependencies between temp versions are
    eliminated by construction; recycled addresses surface as ordinary
    WAR edges in the DAG.

Free lists are FIFO queues seeded at construction with all space after
the pinned region: 8-aligned vector granules, then a scalar pool in the
topmost words. Enqueue-freed-first then dequeue gives the steady-state
fallback by construction: a re-write on an empty pool dequeues its own
just-freed home (in-place write); only a *birth* on an empty pool
overflows (loud error).

Lane access (``LaneRef`` views) resolves against the symbol's *current*
home; a lane access to a symbol with no home is an error (emit a
``Refresh`` or whole-vector write first - the discipline that keeps
lane-written vectors from pinning homes forever). ``Refresh`` on a
pinned symbol is a no-op, so emission can Refresh unconditionally.
"""

from collections import deque

from ir import Sym, Reg, LaneRef, Refresh
from problem import VLEN, SCRATCH_SIZE

# Scalar temps live in the topmost words of scratch; everything between
# the pinned region and the scalar pool is 8-aligned vector granules.
SCALAR_POOL_WORDS = 8


class RenameEngine:
    def __init__(self, pinned: list[Sym]):
        # Allocate & pin: vectors first (stable), then scalars, sequential.
        self._pins: dict[Sym, int] = {}
        ptr = 0
        for sym in sorted(pinned, key=lambda s: not s.is_vec):
            self._pins[sym] = ptr
            ptr += VLEN if sym.is_vec else 1
        # Free lists: all remaining space.
        vec_start = (ptr + VLEN - 1) // VLEN * VLEN
        scalar_start = max(ptr, SCRATCH_SIZE - SCALAR_POOL_WORDS)
        self._free_vec = deque(range(vec_start, scalar_start - VLEN + 1, VLEN))
        self._free_scalar = deque(range(scalar_start, SCRATCH_SIZE))
        self._table: dict[Sym, int] = {}     # temp sym -> current home

    def rename(self, instrs: list) -> list:
        """Symbolic -> resolved, single forward pass. Refresh directives
        are consumed (re-homing their symbol) and dropped from the output.
        For every other instruction: map its read operands, then its write
        operands (reads first - a self-read-write instruction must see the
        OLD home on its reads), and rebuild it resolved."""
        out = []
        for instr in instrs:
            if isinstance(instr, Refresh):
                if instr.vec not in self._pins:   # no-op on pinned symbols
                    self._write(instr.vec)
                continue
            rd = [self.read_op(o) for o in instr.read_operands()]
            wr = [self.write_op(o) for o in instr.write_operands()]
            out.append(instr.resolve(rd, wr))
        return out

    def read_op(self, o):
        """Resolve a read operand against current homes (rename contract)."""
        if isinstance(o, LaneRef):
            return LaneRef(self._home(o.vec), o.j)
        return self._home(o)

    def write_op(self, o):
        """Resolve a write operand (rename contract). Whole-symbol writes
        to temps re-home; lane writes resolve against the current home."""
        if isinstance(o, LaneRef):
            return LaneRef(self._home(o.vec), o.j)
        return self._write(o)

    def debug_map(self) -> dict[int, tuple[str, int]]:
        """addr -> (name, length) for the simulator's debug scratch map."""
        m = {addr: (sym.name, VLEN if sym.is_vec else 1)
             for sym, addr in self._pins.items()}
        m.update({addr: (sym.name, VLEN if sym.is_vec else 1)
                  for sym, addr in self._table.items()})
        return m

    # -- internals --

    def _home(self, sym: Sym) -> Reg:
        """The current home of a symbol. Error on read / lane access of a
        symbol that was never written (use-before-def / missing Refresh)."""
        addr = self._pins.get(sym)
        if addr is None:
            addr = self._table.get(sym)
        if addr is None:
            raise KeyError(
                f"symbol {sym.name!r} has no home - read or lane write "
                f"before any whole-symbol write (use-before-def, or a "
                f"missing Refresh)")
        return Reg(addr, sym.is_vec)

    def _write(self, sym: Sym) -> Reg:
        """Whole-symbol write: pinned writes in place; temps re-home via
        the FIFO free pool (birth when the symbol has no old home)."""
        pinned = self._pins.get(sym)
        if pinned is not None:
            return Reg(pinned, sym.is_vec)
        free = self._free_vec if sym.is_vec else self._free_scalar
        old = self._table.pop(sym, None)
        if old is not None:
            free.append(old)          # enqueue freed home first...
        try:
            new = free.popleft()      # ...then dequeue (fallback: own home)
        except IndexError:
            raise RuntimeError(
                f"rename: out of {'vector' if sym.is_vec else 'scalar'} "
                f"temp space at birth of {sym.name!r} "
                f"({len(self._table)} live temps)") from None
        self._table[sym] = new
        return Reg(new, sym.is_vec)
