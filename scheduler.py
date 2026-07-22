"""DAG builder for the VLIW scheduler.

See notes/scheduler_design.md for the design. This module implements:
- slot_io: dispatch over every ISA form -> (reads, writes) in scratch addrs
- build_dag: program-order walk building dependency edges (RAW weight-1,
  WAR weight-0) with edge dedup per (src, dst, weight) and invariant
  asserts. Dead writes are warned, not eliminated.

The scheduler proper (frontier + partial schedules + cycle placement) will
be added in a follow-up. This module just builds and validates the DAG.
"""

from dataclasses import dataclass, field

from problem import VLEN

# Flow ops that modify the PC — our DAG cannot represent control flow.
# Hitting one means a jump leaked into the body.
FLOW_PANIC = {"cond_jump", "cond_jump_rel", "jump", "jump_indirect", "halt", "pause"}


@dataclass
class DNode:
    idx: int                                     # program index
    engine: str                                  # alu / valu / load / store / flow / debug
    slot: tuple                                  # raw slot tuple
    reads: list[int] = field(default_factory=list)
    writes: list[tuple[int, bool]] = field(default_factory=list)  # (addr, is_vector)
    in_edges: list[tuple[int, int]] = field(default_factory=list)  # (src_idx, weight) deduped
    out_edges: list[tuple[int, int]] = field(default_factory=list) # (dst_idx, weight) deduped
    unresolved: int = 0                          # == len(in_edges) at construction
    ready_cycle: int = 0
    commit_cycle: int = -1
    # partial-schedule fields (used by the scheduler, not by build_dag):
    lanes_done: int = 0
    lanes_total: int = 0
    engine_choice: str | None = None


def slot_io(engine: str, slot: tuple) -> tuple[list[tuple[int, bool]], list[tuple[int, bool]]]:
    """Return (reads, writes) for a slot.

    Each entry is (addr, is_vector).
    is_vector=True on reads means: reads the full 8-lane vector [addr..addr+7].
    is_vector=True on writes means: writes the full 8-lane vector [addr..addr+7].
    is_vector=False means: scalar, just the single word at addr.

    Memory accesses are NOT dependency edges (the body only reads the
    read-only tree; scratch has no indirect reads).
    """
    reads: list[tuple[int, bool]] = []
    writes: list[tuple[int, bool]] = []

    if engine == "alu":
        # ("op", dest, a1, a2) — binary scalar
        _, dest, a1, a2 = slot
        reads.append((a1, False))
        reads.append((a2, False))
        writes.append((dest, False))

    elif engine == "valu":
        op = slot[0]
        if op == "vbroadcast":
            # ("vbroadcast", dest, src) — dest is vec, src is scalar
            _, dest, src = slot
            reads.append((src, False))
            writes.append((dest, True))
        elif op == "multiply_add":
            # ("multiply_add", dest, a, b, c) — all vec (VLEN=8)
            _, dest, a, b, c = slot
            reads.append((a, True))
            reads.append((b, True))
            reads.append((c, True))
            writes.append((dest, True))
        else:
            # (op, dest, a1, a2) — elementwise over VLEN=8 lanes
            _, dest, a1, a2 = slot
            reads.append((a1, True))
            reads.append((a2, True))
            writes.append((dest, True))

    elif engine == "load":
        op = slot[0]
        if op == "load":
            # ("load", dest, addr) — scalar; addr holds mem ptr
            _, dest, addr = slot
            reads.append((addr, False))
            writes.append((dest, False))
        elif op == "load_offset":
            # ("load_offset", dest, addr, offset) — offset is a literal int
            # writes scratch[dest + offset], reads scratch[addr + offset]
            _, dest, addr, offset = slot
            reads.append((addr + offset, False))
            writes.append((dest + offset, False))
        elif op == "vload":
            # ("vload", dest, addr) — addr is scalar (mem base ptr)
            _, dest, addr = slot
            reads.append((addr, False))
            writes.append((dest, True))
        elif op == "const":
            # ("const", dest, val) — val is literal, no reads
            _, dest, _val = slot
            writes.append((dest, False))
        else:
            raise NotImplementedError(f"Unknown load op: {op}")

    elif engine == "store":
        op = slot[0]
        if op == "store":
            # ("store", addr, src) — reads addr (mem ptr) + src (data); no writes
            _, addr, src = slot
            reads.append((addr, False))
            reads.append((src, False))
        elif op == "vstore":
            # ("vstore", addr, src) — reads addr (scalar) + src..src+7 (vec)
            _, addr, src = slot
            reads.append((addr, False))
            reads.append((src, True))
        else:
            raise NotImplementedError(f"Unknown store op: {op}")

    elif engine == "flow":
        op = slot[0]
        if op in FLOW_PANIC:
            raise NotImplementedError(
                f"Flow op '{op}' at slot {slot} modifies PC — "
                f"cannot be represented in the DAG")
        elif op == "select":
            # ("select", dest, cond, a, b) — scalar
            _, dest, cond, a, b = slot
            reads.extend([(cond, False), (a, False), (b, False)])
            writes.append((dest, False))
        elif op == "vselect":
            # ("vselect", dest, cond, a, b) — vec
            _, dest, cond, a, b = slot
            reads.extend([(cond, True), (a, True), (b, True)])
            writes.append((dest, True))
        elif op == "add_imm":
            # ("add_imm", dest, a, imm) — imm is literal
            _, dest, a, _imm = slot
            reads.append((a, False))
            writes.append((dest, False))
        elif op == "coreid":
            # ("coreid", dest) — writes dest, no reads
            _, dest = slot
            writes.append((dest, False))
        elif op == "trace_write":
            # ("trace_write", val) — reads val
            _, val = slot
            reads.append((val, False))
        else:
            raise NotImplementedError(f"Unknown flow op: {op}")

    elif engine == "debug":
        op = slot[0]
        if op == "compare":
            # ("compare", loc, key) — reads loc (scalar)
            _, loc, _key = slot
            reads.append((loc, False))
        elif op == "vcompare":
            # ("vcompare", loc, keys) — reads loc..loc+7 (vec)
            _, loc, _keys = slot
            reads.append((loc, True))
        elif op == "comment":
            pass  # no deps
        else:
            pass  # unknown debug ops are no-ops for dependency purposes

    else:
        raise NotImplementedError(f"Unknown engine: {engine}")

    return reads, writes


# ---------------------------------------------------------------------------
# Region state for build_dag
# ---------------------------------------------------------------------------

# last_writer[region] is one of:
#   DNode                         — vec form: one node wrote all 8 lanes
#   list[DNode | None] of 8       — list form: per-lane writers
# readers_since[region] is list[int] (node indices that read since last write)


def build_dag(slots: list[tuple[str, tuple]]) -> tuple[list[DNode], set[int]]:
    """Build a DAG from a sequential slot list.

    Returns (nodes, frontier) where frontier is the set of node indices with
    zero in-edges (ready at cycle 0).

    Edge model:
    - RAW (weight 1): producer → consumer, read accesses the prior writer.
      A vector read depends on the per-lane writers of all 8 lanes (deduped).
      A scalar read of lane j depends only on lane j's writer.
    - WAR (weight 0): old reader → new writer of the SAME lane. Scalar writes
      only get WAR edges from readers of that specific lane; vector writes get
      WAR from all 8 lanes' readers. Same cycle is safe (read-before-write).
    - No WAW edges. Dead writes are warned, not eliminated.
    - Self-RW: no self-edge (RAW uses prior writer; WAR skips self).

    Per-lane reader tracking (keyed by (region, lane)) avoids false WAR edges
    and false dead-write warnings for scalar writes to different lanes of the
    same region (e.g. the 8 scalar gather loads landing into nv).
    """
    nodes: list[DNode] = []
    last_writer: dict[int, DNode | list] = {}       # region -> DNode(vec) or list[DNode|None]×8
    readers_since: dict[tuple[int, int], list[int]] = {}  # (region, lane) -> [node indices]
    frontier: set[int] = set()

    for idx, (engine, slot) in enumerate(slots):
        reads, writes = slot_io(engine, slot)
        node = DNode(idx=idx, engine=engine, slot=slot, reads=reads, writes=writes)
        nodes.append(node)

        # ---- Step 1: RAW ----
        read_lanes: set[tuple[int, int]] = set()   # (region, lane) pairs this node reads
        for addr, is_vec in reads:
            r = addr >> 3
            if is_vec:
                lanes = range(VLEN)
            else:
                lanes = (addr & 7,)
            for lane in lanes:
                key = (r, lane)
                if key in read_lanes:
                    continue
                read_lanes.add(key)
                lw = last_writer.get(r)
                if lw is None:
                    continue
                if isinstance(lw, list):
                    src = lw[lane]
                    if src is None:
                        continue
                    src_idx = src.idx
                else:
                    src_idx = lw.idx
                # dedup-add RAW edge (src_idx -> idx, weight=1)
                if (src_idx, 1) not in node.in_edges:
                    node.in_edges.append((src_idx, 1))
                    nodes[src_idx].out_edges.append((idx, 1))

        # ---- Step 2: frontier? ----
        if not node.in_edges:
            frontier.add(idx)

        # ---- Step 3: append self to readers_since for each read (region, lane) ----
        for r, lane in read_lanes:
            readers_since.setdefault((r, lane), []).append(idx)

        # ---- Step 4: WAR ----
        write_lanes: list[tuple[int, int, bool]] = []  # (region, lane, is_vec)
        for addr, is_vec in writes:
            r = addr >> 3
            if is_vec:
                for lane in range(VLEN):
                    write_lanes.append((r, lane, True))
            else:
                write_lanes.append((r, addr & 7, False))

        war_seen: set[int] = set()   # dedup WAR sources across lanes
        for r, lane, is_vec in write_lanes:
            for old_idx in readers_since.get((r, lane), []):
                if old_idx == idx or old_idx in war_seen:
                    continue
                war_seen.add(old_idx)
                node.in_edges.append((old_idx, 0))
                nodes[old_idx].out_edges.append((idx, 0))

        # ---- Step 5: dead-write warning ----
        # Self-reads COUNT: a node that reads+writes the same addr (e.g. fma on
        # val_vec) reads the OLD writer's output before writing its own. So the
        # old writer IS read (by self) — not dead. Do not exclude self here.
        for r, lane, is_vec in write_lanes:
            lw = last_writer.get(r)
            if lw is None:
                continue
            old = lw[lane] if isinstance(lw, list) else lw
            if old is None:
                continue
            rs = readers_since.get((r, lane), [])
            if not rs:
                print(f"WARN: dead write — node {idx} '{engine} {slot[0]}' "
                      f"overwrites unread writer {old.idx} at region {r} lane {lane}")

        # ---- Step 6: clear readers_since, update last_writer ----
        for r, lane, is_vec in write_lanes:
            readers_since[(r, lane)] = []
        # Commit last_writer per region (collapse vector writes to vec form)
        write_regions: dict[int, bool] = {}
        for addr, is_vec in writes:
            r = addr >> 3
            write_regions[r] = write_regions.get(r, False) or is_vec
        for r, is_vec in write_regions.items():
            if is_vec:
                last_writer[r] = node
            else:
                for addr, is_vec in writes:
                    if (addr >> 3) == r and not is_vec:
                        lane = addr & 7
                        lw = last_writer.get(r)
                        if lw is None or isinstance(lw, DNode):
                            old = lw if lw is not None else None
                            lw = ([old] * VLEN) if old is not None else ([None] * VLEN)
                        lw[lane] = node
                        last_writer[r] = lw

        # ---- Final: recompute unresolved ----
        node.unresolved = len(node.in_edges)

    # ---- Bidirectional invariant asserts ----
    for n in nodes:
        assert n.unresolved == len(n.in_edges), (
            f"Node {n.idx}: unresolved={n.unresolved} != len(in_edges)={len(n.in_edges)}")
        for src_idx, w in n.in_edges:
            assert (n.idx, w) in nodes[src_idx].out_edges, (
                f"Node {n.idx}: in-edge ({src_idx},{w}) not in src out_edges")
        for dst_idx, w in n.out_edges:
            assert (n.idx, w) in nodes[dst_idx].in_edges, (
                f"Node {n.idx}: out-edge ({dst_idx},{w}) not in dst in_edges")

    return nodes, frontier