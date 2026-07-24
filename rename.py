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
home; a lane access to a symbol with no home is an error. Lane writes do
not occur - the gather is emitted as a whole-vector ``Gather`` op, which
rename-on-write re-homes per gather round (and is decomposed into scalar
loads after renaming). Remaining LaneRefs are prologue lane reads on
pinned symbols (broadcast sources), where a home always exists.
"""

from collections import deque

from ir import Sym, Reg, LaneRef, Load, Gather, Free
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
        """Symbolic -> resolved. First a backward liveness pass inserts
        Free directives after each symbol's last use; then a single
        forward pass renames: reads mapped against current homes, then
        writes (a self-read-write instruction sees the OLD home on its
        reads), Free directives return homes to the pool, Gathers are
        decomposed into VLEN scalar loads on the resolved homes (lane
        addresses are plain arithmetic)."""
        out = []
        for instr in _auto_free(instrs):
            if isinstance(instr, Free):
                self._free(instr.sym)
                continue
            if isinstance(instr, Gather):
                addr = self.read_op(instr.addr)
                dest = self.write_op(instr.dest)
                out.extend(Load(Reg(dest.addr + j), Reg(addr.addr + j))
                           for j in range(VLEN))
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
        symbol that was never written (use-before-def)."""
        addr = self._pins.get(sym)
        if addr is None:
            addr = self._table.get(sym)
        if addr is None:
            raise KeyError(
                f"symbol {sym.name!r} has no home - read or lane access "
                f"before any whole-symbol write (use-before-def)")
        return Reg(addr, sym.is_vec)

    def _free(self, sym: Sym) -> None:
        """Honor a Free directive: return the symbol's home to the pool.
        No-op on pinned symbols and on symbols with no current home."""
        home = self._table.pop(sym, None)
        if home is not None:
            (self._free_vec if sym.is_vec else self._free_scalar).append(home)

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


def _auto_free(instrs: list) -> list:
    """Backward liveness pass: insert Free directives so every dead symbol
    version's home returns to the pool.

    Going backward, ``live`` is the set of symbols whose value is needed
    after the current instruction. A Free(s) goes after instruction I iff
    s is touched by I (read or written) and s is NOT in live_after(I):

      - read, live_after: still live - no free.
      - read, not live_after: last use - free the current home.
      - write, live_after: birth of a needed version - no free.
      - write, not live_after: dead version - free it (also covers
        dead-on-arrival writes).
      - read+write (self-read-write): the OLD version's home is recycled
        by rename-on-write at the write itself - never Free it here. If
        the NEW version is dead (not live_after), Free kills the new home,
        which is sound.

    Lane reads keep the vector alive just like vector reads (liveness
    tracks base symbols). Built as a new list (no insert-while-iterating);
    Frees sorted by symbol name for deterministic output (Sym hashes are
    process-randomized).
    """
    def bases(ops):
        return {o.vec if isinstance(o, LaneRef) else o for o in ops}

    live: set[Sym] = set()
    out = []
    for instr in reversed(instrs):
        rd = bases(instr.read_operands())
        wr = bases(instr.write_operands())
        frees = [Free(s) for s in sorted(rd | wr, key=lambda s: s.name)
                 if s not in live]
        live -= wr
        live |= rd
        out.extend(frees)
        out.append(instr)
    out.reverse()
    return out
