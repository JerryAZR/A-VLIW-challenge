"""Intermediate representation for kernel programs.

A proper instruction layer between kernel emission and the simulator's
(engine, slot-tuple) format. Every instruction is a small frozen dataclass
whose operands are typed register references; each class knows:

  - ``engine``       - the functional unit it issues on (ClassVar)
  - ``reads()``      - operand register ids read, as (addr, is_vec) pairs
  - ``writes()``     - destination register ids written, same shape
  - ``lower(res)``   - the simulator slot tuple, resolving register refs
                       through ``res`` (identity for pinned addresses)

Programs exist in two phases:

  - **symbolic** - operands are ``Sym`` (declared variables) or ``LaneRef``
    views of them. This is what the kernel builder emits.
  - **resolved** - a ``RenameEngine`` has translated every ``Sym`` to a
    ``Reg`` (a physical scratch address) via its pin table. ``reads()`` /
    ``writes()`` / the DAG operate on resolved instructions only.

Register references:

  - ``Sym(name, is_vec)``   - a declared variable (symbolic phase)
  - ``Reg(addr, is_vec)``   - a physical scalar word or 8-lane vector region
                              (resolved phase)
  - ``LaneRef(vec, j)``     - one scalar lane of a vector, in either phase -
                              like indexing a declared vector in a programming
                              language: the lane is a view, not its own symbol

``reads()``/``writes()`` return the same (addr, is_vec) tuple pairs the
DAG's ReadWriteTable has always consumed, so the dependency machinery is
unchanged. Immediates (const values, vcompare keys) are plain ints, not
registers.

Only the ops the kernel actually emits are modeled. PC-modifying flow ops
(jumps/halt/pause) are not schedulable: ``Pause`` exists for the linear
prologue/epilogue and the DAG builder rejects it.
"""

from dataclasses import dataclass, replace
from typing import ClassVar, Callable, Union

from problem import VLEN

# A resolved register id, as consumed by the scheduler's ReadWriteTable:
# (base_addr, is_vector). is_vector=True covers [addr..addr+VLEN-1].
RegId = tuple[int, bool]

# resolve hook: maps a Reg to its base address (identity when pinned).
Resolver = Callable[["Reg"], int]


def _ident(r: "Reg") -> int:
    return r.addr


@dataclass(frozen=True)
class Sym:
    """A declared variable - the symbolic-phase operand. Shape (scalar or
    8-lane vector) is fixed at declaration. Becomes a ``Reg`` (physical
    address) when the RenameEngine resolves it."""
    name: str
    is_vec: bool = False

    def lane(self, j: int) -> "LaneRef":
        """Explicit per-lane scalar view: vec[j], like indexing a declared
        vector. The lane is a view of this symbol, not a symbol itself."""
        assert self.is_vec, f"lane() requires a vector Sym (got {self})"
        assert 0 <= j < VLEN
        return LaneRef(self, j)


@dataclass(frozen=True)
class Reg:
    """A physical scalar word (is_vec=False) or 8-lane vector region -
    the resolved-phase operand."""
    addr: int
    is_vec: bool = False

    def lane(self, j: int) -> "LaneRef":
        """Explicit per-lane scalar view (e.g. gather landing slots)."""
        assert self.is_vec, "lane() requires a vector Reg"
        assert 0 <= j < VLEN
        return LaneRef(self, j)

    def resolve(self, res: Resolver = _ident) -> int:
        return res(self)

    def reg_id(self) -> RegId:
        return (self.addr, self.is_vec)


@dataclass(frozen=True)
class LaneRef:
    """One scalar lane of a vector (``vec[j]``), in either phase."""
    vec: Union[Sym, Reg]
    j: int

    def resolve(self, res: Resolver = _ident) -> int:
        assert isinstance(self.vec, Reg), "resolve() requires resolved phase"
        return res(self.vec) + self.j

    def reg_id(self) -> RegId:
        assert isinstance(self.vec, Reg), "reg_id() requires resolved phase"
        return (self.vec.addr + self.j, False)


Operand = Union[Sym, Reg, LaneRef]


def _ids(ops) -> list[RegId]:
    return [o.reg_id() for o in ops]


class Instr:
    """Base class for IR instructions. ``engine`` is a ClassVar."""
    engine: ClassVar[str]
    # Operand fields read / written by this instruction (field names) -
    # they define the position order of the read_operands() /
    # write_operands() / resolve() rename contract. Reads are always
    # resolved before writes (a self-read-write instruction must see the
    # OLD home on its reads). Non-operand fields (op strings, immediates,
    # keys) are never listed and pass through untouched.
    _RD: ClassVar[tuple] = ()
    _WR: ClassVar[tuple] = ()

    def reads(self) -> list[RegId]:
        raise NotImplementedError

    def writes(self) -> list[RegId]:
        raise NotImplementedError

    def lower(self, res: Resolver = _ident) -> tuple:
        """The simulator slot tuple (without the engine tag)."""
        raise NotImplementedError

    # -- rename contract -------------------------------------------------
    # The instruction exposes its read/write operands positionally; the
    # rename engine maps each operand and hands back position-indexed
    # lists; resolve() rebuilds. Neither side pokes into the other's
    # internals.

    def read_operands(self) -> tuple:
        """Read operands, in field order."""
        return tuple(getattr(self, f) for f in self._RD)

    def write_operands(self) -> tuple:
        """Write operands, in field order."""
        return tuple(getattr(self, f) for f in self._WR)

    def resolve(self, rd: list, wr: list) -> "Instr":
        """Rebuild with resolved operands: position-indexed lists of the
        same length/order as read_operands() / write_operands()."""
        assert len(rd) == len(self._RD) and len(wr) == len(self._WR)
        return replace(self, **dict(zip(self._RD, rd)),
                       **dict(zip(self._WR, wr)))


# ---------------------------------------------------------------------------
# alu
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Alu(Instr):
    """Scalar binary op: dest = a1 OP a2 (mod 2^32)."""
    engine: ClassVar[str] = "alu"
    _RD: ClassVar[tuple] = ("a1", "a2")
    _WR: ClassVar[tuple] = ("dest",)
    op: str
    dest: Operand
    a1: Operand
    a2: Operand

    def reads(self):
        return _ids([self.a1, self.a2])

    def writes(self):
        return _ids([self.dest])

    def lower(self, res=_ident):
        return (self.op, self.dest.resolve(res),
                self.a1.resolve(res), self.a2.resolve(res))


# ---------------------------------------------------------------------------
# valu
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VecElem(Instr):
    """Elementwise vector op over VLEN lanes: dest[i] = a1[i] OP a2[i].

    Spillable to per-lane alu slots by the scheduler."""
    engine: ClassVar[str] = "valu"
    _RD: ClassVar[tuple] = ("a1", "a2")
    _WR: ClassVar[tuple] = ("dest",)
    op: str
    dest: Operand
    a1: Operand
    a2: Operand

    def reads(self):
        return _ids([self.a1, self.a2])

    def writes(self):
        return _ids([self.dest])

    def lower(self, res=_ident):
        return (self.op, self.dest.resolve(res),
                self.a1.resolve(res), self.a2.resolve(res))


@dataclass(frozen=True)
class VecFma(Instr):
    """Fused multiply_add: dest[i] = a[i]*b[i] + c[i]. valu-only (rigid:
    no scalar fma exists, so the scheduler cannot spill it)."""
    engine: ClassVar[str] = "valu"
    _RD: ClassVar[tuple] = ("a", "b", "c")
    _WR: ClassVar[tuple] = ("dest",)
    dest: Operand
    a: Operand
    b: Operand
    c: Operand

    def reads(self):
        return _ids([self.a, self.b, self.c])

    def writes(self):
        return _ids([self.dest])

    def lower(self, res=_ident):
        return ("multiply_add", self.dest.resolve(res), self.a.resolve(res),
                self.b.resolve(res), self.c.resolve(res))


@dataclass(frozen=True)
class VBroadcast(Instr):
    """dest[i] = scratch[src] for all lanes."""
    engine: ClassVar[str] = "valu"
    _RD: ClassVar[tuple] = ("src",)
    _WR: ClassVar[tuple] = ("dest",)
    dest: Operand
    src: Operand

    def reads(self):
        return _ids([self.src])

    def writes(self):
        return _ids([self.dest])

    def lower(self, res=_ident):
        return ("vbroadcast", self.dest.resolve(res), self.src.resolve(res))


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Load(Instr):
    """Scalar gather: dest = mem[scratch[addr]]."""
    engine: ClassVar[str] = "load"
    _RD: ClassVar[tuple] = ("addr",)
    _WR: ClassVar[tuple] = ("dest",)
    dest: Operand
    addr: Operand

    def reads(self):
        return _ids([self.addr])

    def writes(self):
        return _ids([self.dest])

    def lower(self, res=_ident):
        return ("load", self.dest.resolve(res), self.addr.resolve(res))


@dataclass(frozen=True)
class VLoad(Instr):
    engine: ClassVar[str] = "load"
    _RD: ClassVar[tuple] = ("addr",)
    _WR: ClassVar[tuple] = ("dest",)
    dest: Operand
    addr: Operand

    def reads(self):
        return _ids([self.addr])

    def writes(self):
        return _ids([self.dest])

    def lower(self, res=_ident):
        return ("vload", self.dest.resolve(res), self.addr.resolve(res))


@dataclass(frozen=True)
class Gather(Instr):
    """Vector gather: dest[j] = mem[scratch[addr+j]] for j in [0, VLEN).

    The ISA has no gather; the rename engine decomposes this into VLEN
    scalar loads AFTER renaming (lane addresses are plain arithmetic on
    resolved homes). At rename time it is a whole-vector read of addr and
    a whole-vector write of dest - rename-on-write re-homes dest per
    gather, so no lane-access discipline (and no Refresh directive) is
    needed. Never reaches the DAG or the simulator."""
    engine: ClassVar[str] = "load"
    _RD: ClassVar[tuple] = ("addr",)
    _WR: ClassVar[tuple] = ("dest",)
    dest: Operand
    addr: Operand

    def reads(self):
        return _ids([self.addr])

    def writes(self):
        return _ids([self.dest])

    def lower(self, res=_ident):
        raise NotImplementedError(
            "Gather must be decomposed into scalar loads by "
            "RenameEngine.rename() before lowering")


@dataclass(frozen=True)
class Const(Instr):
    """dest = <literal>. A load-engine slot with an immediate operand."""
    engine: ClassVar[str] = "load"
    _WR: ClassVar[tuple] = ("dest",)
    dest: Operand
    val: int

    def reads(self):
        return []

    def writes(self):
        return _ids([self.dest])

    def lower(self, res=_ident):
        return ("const", self.dest.resolve(res), self.val)


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VStore(Instr):
    """mem[scratch[addr]+i] = src[i] for i in [0, VLEN)."""
    engine: ClassVar[str] = "store"
    _RD: ClassVar[tuple] = ("addr", "src")
    addr: Operand
    src: Operand

    def reads(self):
        return _ids([self.addr, self.src])

    def writes(self):
        return []

    def lower(self, res=_ident):
        return ("vstore", self.addr.resolve(res), self.src.resolve(res))


# ---------------------------------------------------------------------------
# flow
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VSelect(Instr):
    """dest[i] = a[i] if cond[i] != 0 else b[i]."""
    engine: ClassVar[str] = "flow"
    _RD: ClassVar[tuple] = ("cond", "a", "b")
    _WR: ClassVar[tuple] = ("dest",)
    dest: Operand
    cond: Operand
    a: Operand
    b: Operand

    def reads(self):
        return _ids([self.cond, self.a, self.b])

    def writes(self):
        return _ids([self.dest])

    def lower(self, res=_ident):
        return ("vselect", self.dest.resolve(res), self.cond.resolve(res),
                self.a.resolve(res), self.b.resolve(res))


@dataclass(frozen=True)
class Pause(Instr):
    """Prologue/epilogue barrier. Modifies PC - NOT schedulable; the DAG
    builder rejects it. Disabled at grading."""
    engine: ClassVar[str] = "flow"

    def reads(self):
        return []

    def writes(self):
        return []

    def lower(self, res=_ident):
        return ("pause",)


# ---------------------------------------------------------------------------
# debug (0-cycle, disabled at grading)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DebugVCompare(Instr):
    """Dev oracle: assert scratch[loc..loc+7] == value_trace[keys]."""
    engine: ClassVar[str] = "debug"
    _RD: ClassVar[tuple] = ("loc",)
    loc: Operand
    keys: list

    def reads(self):
        return _ids([self.loc])

    def writes(self):
        return []

    def lower(self, res=_ident):
        return ("vcompare", self.loc.resolve(res), self.keys)
