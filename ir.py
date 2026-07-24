"""Intermediate representation for kernel programs.

A proper instruction layer between kernel emission and the simulator's
(engine, slot-tuple) format. Every instruction is a small frozen dataclass
whose operands are typed register references; each class knows:

  - ``engine``       - the functional unit it issues on (ClassVar)
  - ``reads()``      - operand register ids read, as (addr, is_vec) pairs
  - ``writes()``     - destination register ids written, same shape
  - ``lower(res)``   - the simulator slot tuple, resolving register refs
                       through ``res`` (identity for pinned addresses; the
                       future rename engine supplies a symbol->addr map)

Register references come in two shapes:

  - ``Reg(addr, is_vec)``  - a whole scalar word or 8-lane vector region
  - ``LaneRef(vec, lane)`` - one scalar lane of a vector register (the
    gather's per-lane loads; explicit so renaming never needs raw address
    arithmetic on vector bases)

``reads()``/``writes()`` return the same (addr, is_vec) tuple pairs the
DAG's ReadWriteTable has always consumed, so the dependency machinery is
unchanged. Immediates (const values, vcompare keys, vstore/vload stride)
are plain ints, not registers.

Only the ops the kernel actually emits are modeled. PC-modifying flow ops
(jumps/halt/pause) are not schedulable: ``Pause`` exists for the linear
prologue/epilogue and the DAG builder rejects it.
"""

from dataclasses import dataclass
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
class Reg:
    """A whole scalar word (is_vec=False) or 8-lane vector region."""
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
    """One scalar lane of a vector register."""
    vec: Reg
    j: int

    def resolve(self, res: Resolver = _ident) -> int:
        return res(self.vec) + self.j

    def reg_id(self) -> RegId:
        return (self.vec.addr + self.j, False)


Operand = Union[Reg, LaneRef]


def _ids(ops) -> list[RegId]:
    return [o.reg_id() for o in ops]


class Instr:
    """Base class for IR instructions. ``engine`` is a ClassVar."""
    engine: ClassVar[str]

    def reads(self) -> list[RegId]:
        raise NotImplementedError

    def writes(self) -> list[RegId]:
        raise NotImplementedError

    def lower(self, res: Resolver = _ident) -> tuple:
        """The simulator slot tuple (without the engine tag)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# alu
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Alu(Instr):
    """Scalar binary op: dest = a1 OP a2 (mod 2^32)."""
    engine: ClassVar[str] = "alu"
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
    op: str
    dest: Reg
    a1: Reg
    a2: Reg

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
    dest: Reg
    a: Reg
    b: Reg
    c: Reg

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
    dest: Reg
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
    """Contiguous 8-word fetch: dest[i] = mem[scratch[addr]+i]."""
    engine: ClassVar[str] = "load"
    dest: Reg
    addr: Operand

    def reads(self):
        return _ids([self.addr])

    def writes(self):
        return _ids([self.dest])

    def lower(self, res=_ident):
        return ("vload", self.dest.resolve(res), self.addr.resolve(res))


@dataclass(frozen=True)
class Const(Instr):
    """dest = <literal>. A load-engine slot with an immediate operand."""
    engine: ClassVar[str] = "load"
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
    addr: Operand
    src: Reg

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
    dest: Reg
    cond: Reg
    a: Reg
    b: Reg

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
    loc: Reg
    keys: list

    def reads(self):
        return _ids([self.loc])

    def writes(self):
        return []

    def lower(self, res=_ident):
        return ("vcompare", self.loc.resolve(res), self.keys)
