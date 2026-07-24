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
# The pool must hold every simultaneously-live scalar temp: with no app pins,
# the 32 per-group out_addr (written at body start, read by the round-15
# vstores) are all live across the body, plus header vars read in the body.
SCALAR_POOL_WORDS = 48

# Debug logging. When DEBUG_DIR is set, the rename pass writes three
# separate log files into it (so they can be diffed side by side):
#   autofree.txt  - the post-auto-free instruction stream: input
#                   instructions plus the auto-added Free directives,
#                   with sequence number and content.
#   alloc.txt     - every home allocation / free, tied to the rename id
#                   of the instruction that triggered it.
#   gather.txt    - every Gather decomposition into VLEN scalar loads.
# ``None`` disables all logging.
DEBUG_DIR: str | None = None

# Monotonic source of stable instruction ids. Each instruction that enters
# the rename pass is stamped with a unique ``rid`` (rename id) that never
# changes, so the auto-free dump, the alloc/free log, the gather
# decomposition log, and the DAG builder's warnings all refer to the same
# numbering. The id is attached via ``object.__setattr__`` (bypassing frozen-
# dataclass immutability); ``dataclasses.replace`` copies the instance dict,
# so the id survives the ``resolve()`` rebuild onto the resolved copy.
_next_rid = [0]


def _stamp(instr):
    """Assign a stable rid to an instruction (in place) if it lacks one."""
    if getattr(instr, "rid", None) is None:
        object.__setattr__(instr, "rid", _next_rid[0])
        _next_rid[0] += 1
    return instr


def rid_of(instr) -> int:
    """Public accessor: the stable rename id stamped on an instruction during
    the rename pass. -1 if the instruction never passed through rename."""
    return getattr(instr, "rid", -1)


class _Log:
    """One log file per stream, shared across all rename() calls in a run so
    the many small prologue calls and the one big body call append into the
    same file (opened once, truncated on first open). No-ops when DEBUG_DIR
    is unset."""

    _files: dict = {}        # name -> open file handle (class-level, shared)

    def __init__(self, name: str):
        self._name = name

    def write(self, line: str) -> None:
        if DEBUG_DIR is None:
            return
        fh = _Log._files.get(self._name)
        if fh is None:
            import os
            os.makedirs(DEBUG_DIR, exist_ok=True)
            fh = open(os.path.join(DEBUG_DIR, self._name), "w")
            _Log._files[self._name] = fh
        fh.write(line + "\n")

    def close(self) -> None:
        # Files stay open for the whole run (flushed on process exit); closing
        # per-call would truncate the shared file on the next open.
        pass


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
        orig_log = _Log("original.txt")
        af_log = _Log("autofree.txt")
        alloc_log = _Log("alloc.txt")
        gather_log = _Log("gather.txt")
        self._alloc_log = alloc_log          # consumed by _write / _free
        # Stamp every input instruction with a stable rid BEFORE liveness, so
        # the symbolic instructions and their resolved copies share the id
        # (resolve() carries the instance dict over via dataclasses.replace).
        instrs = [_stamp(i) for i in instrs]
        out = []
        for seq, instr in enumerate(instrs):
            rid = rid_of(instr)
            orig_log.write(f"{seq:6d} rid={rid:6d}  INSTR   {instr}")
        for seq, instr in enumerate(_auto_free(instrs)):
            # A Free carries the rid of the instruction that triggered it.
            rid = rid_of(instr)
            if isinstance(instr, Free):
                af_log.write(f"{seq:6d} rid={rid:6d}  FREE    {instr.sym.name}")
                self._free(instr.sym, rid)
                continue
            af_log.write(f"{seq:6d} rid={rid:6d}  INSTR   {instr}")
            if isinstance(instr, Gather):
                addr = self.read_op(instr.addr)
                dest = self.write_op(instr.dest, rid)
                gather_log.write(
                    f"rid={rid:6d}  Gather dest={instr.dest.name}@{dest.addr} "
                    f"addr={instr.addr.name}@{addr.addr} ->")
                for j in range(VLEN):
                    gather_log.write(
                        f"          lane{j}: Load dest={dest.addr + j} "
                        f"addr={addr.addr + j}")
                    out.append(_stamp(Load(Reg(dest.addr + j), Reg(addr.addr + j))))
                continue
            rd = [self.read_op(o) for o in instr.read_operands()]
            wr = [self.write_op(o, rid) for o in instr.write_operands()]
            out.append(instr.resolve(rd, wr))
        af_log.close(); alloc_log.close(); gather_log.close()
        self._alloc_log = None
        return out

    def read_op(self, o):
        """Resolve a read operand against current homes (rename contract)."""
        if isinstance(o, LaneRef):
            return LaneRef(self._home(o.vec), o.j)
        return self._home(o)

    def write_op(self, o, rid: int = -1):
        """Resolve a write operand (rename contract). Whole-symbol writes
        to temps re-home; lane writes resolve against the current home."""
        if isinstance(o, LaneRef):
            return LaneRef(self._home(o.vec), o.j)
        return self._write(o, rid)

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

    def _free(self, sym: Sym, rid: int = -1) -> None:
        """Honor a Free directive: return the symbol's home to the pool.
        No-op on pinned symbols and on symbols with no current home."""
        home = self._table.pop(sym, None)
        if home is not None:
            (self._free_vec if sym.is_vec else self._free_scalar).append(home)
            if self._alloc_log is not None:
                self._alloc_log.write(
                    f"rid={rid:6d}  FREE  {sym.name:14s} home={home}")

    def _write(self, sym: Sym, rid: int = -1) -> Reg:
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
        if self._alloc_log is not None:
            act = "BIRTH" if old is None else "REHOME"
            old_s = "-" if old is None else str(old)
            self._alloc_log.write(
                f"rid={rid:6d}  {act:6s} {sym.name:14s} old={old_s:5s} new={new}")
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

    **Dead-write elision.** A write whose version is never read downstream
    (no written symbol is in ``live_after``) is dead-on-arrival and is
    dropped, not emitted. This matters beyond cleanup: the DAG tracks RAW
    (weight-1) and WAR (weight-0) but deliberately no WAW, relying on a
    reader to bridge two writes of the same region (write1 -> reader ->
    write2). A dead write has no such reader, so it would silently break the
    ordering chain between two live writes that share a recycled home -
    letting the scheduler reorder the second write ahead of a consumer that
    still needed the first write's value. Dropping dead writes removes the
    broken link. Stores/vstores write no scratch symbol (their side effect
    is on mem), and debug nodes write nothing, so neither is ever dropped.
    When an instruction is dropped its reads are also dropped, so we do NOT
    update ``live`` for it at all - this lets deadness cascade backward to
    now-unreferenced producers."""
    def bases(ops):
        return {o.vec if isinstance(o, LaneRef) else o for o in ops}

    live: set[Sym] = set()
    out = []
    for instr in reversed(instrs):
        rd = bases(instr.read_operands())
        wr = bases(instr.write_operands())
        # Dead-write drop: writes at least one symbol, none read downstream.
        if wr and not (wr & live):
            continue            # drop the instruction entirely (and its reads)
        # Each Free is stamped with the rid of the instruction whose last-use
        # triggered it (the instruction the Free is inserted after).
        frees = []
        for s in sorted(rd | wr, key=lambda s: s.name):
            if s not in live:
                f = Free(s)
                object.__setattr__(f, "rid", rid_of(instr))
                frees.append(f)
        live -= wr
        live |= rd
        out.extend(frees)
        out.append(instr)
    out.reverse()
    return out
