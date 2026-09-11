"""Composite-wide structure from declared symbolic forms.

A process declares its flux as reactions
(:meth:`~hallsim.process.Process.reaction_channels`), its algebraic outputs as
rules (:meth:`~hallsim.process.Process.assignment_rules`) and any derivative
given directly as a rate rule (:meth:`~hallsim.process.Process.rate_rules`).
This module assembles what those declarations imply for the whole composite,
over its store paths:

- the stoichiometry ``N``, one column per channel, with the rows it fully
  describes and the rows something outside it also moves;
- the sparsity pattern of the Jacobian ``∂f/∂y`` from each law's free
  symbols, an undeclared process contributing a dense block over its own
  ports and nothing else;
- that Jacobian in as many forward passes as the pattern has colours, which
  is the cost of ``jacfwd`` only when nothing is declared;
- the composite's vector field as sympy, for structural analyses.

A pattern is checked once against the composite's own derivative along a
random direction (:func:`check_pattern`): a declaration that omits a port its
derivative reads would otherwise corrupt every Jacobian silently.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from fractions import Fraction
from functools import reduce
from math import gcd, lcm

import jax
import jax.numpy as jnp
import numpy as np
import sympy

from hallsim.process import PortRole
from hallsim.sbml_math import TIME
from hallsim.store import as_paths
from hallsim.units import canonical_units, conversion_factor


class StructureError(ValueError):
    """A declared symbolic form does not fit the composite it lives in."""


# ── exact linear algebra ────────────────────────────────────────────────


def rational_null_space(rows, n_cols: int) -> list[dict[int, Fraction]]:
    """Exact basis of ``{x : A·x = 0}`` for ``A`` given row by row, each row
    a sequence of length ``n_cols`` or a ``{col: value}`` dict.

    Gauss-Jordan over the rationals on sparse rows, so the answer is the
    null space itself rather than whatever survived a floating-point
    threshold, and a banded ``A`` costs what its band costs rather than its
    square. Returns one ``{col: Fraction}`` per basis vector, in free-column
    order; floats are rationalised to a denominator of at most ``10**6``.
    """
    a: list[dict[int, Fraction]] = []
    for row in rows:
        items = row.items() if isinstance(row, dict) else enumerate(row)
        a.append(
            {
                int(j): (
                    Fraction(v).limit_denominator(10**6)
                    if isinstance(v, float)
                    else Fraction(v)
                )
                for j, v in items
                if v
            }
        )
    in_col: dict[int, set[int]] = defaultdict(set)
    for i, row in enumerate(a):
        for j in row:
            in_col[j].add(i)
    used: set[int] = set()
    pivot_row: dict[int, int] = {}
    for c in range(n_cols):
        candidates = [i for i in in_col.get(c, ()) if i not in used]
        if not candidates:
            continue
        i = min(candidates, key=lambda k: (len(a[k]), k))
        used.add(i)
        pivot_row[c] = i
        row = a[i]
        inv = 1 / row[c]
        if inv != 1:
            for j in row:
                row[j] *= inv
        for k in list(in_col[c]):
            if k == i:
                continue
            other = a[k]
            f = other[c]
            for j, v in row.items():
                new = other.get(j, 0) - f * v
                if new:
                    if j not in other:
                        in_col[j].add(k)
                    other[j] = new
                elif j in other:
                    del other[j]
                    in_col[j].discard(k)
    basis = []
    for free in range(n_cols):
        if free in pivot_row:
            continue
        vec = {free: Fraction(1)}
        for c, i in pivot_row.items():
            v = a[i].get(free)
            if v:
                vec[c] = -v
        basis.append(vec)
    return basis


def integerize(vec: dict[int, Fraction]) -> dict[int, int]:
    """The smallest integer multiple of a rational vector, sign-normalised so
    its first non-zero entry is positive — ``ATP + ADP``, not
    ``(-0.707, -0.707)``."""
    items = sorted(vec.items())
    denom = reduce(lcm, (v.denominator for _, v in items), 1)
    ints = [(j, int(v * denom)) for j, v in items]
    common = reduce(gcd, (abs(i) for _, i in ints if i), 0) or 1
    ints = [(j, i // common) for j, i in ints]
    first = next((i for _, i in ints if i), 1)
    sign = -1 if first < 0 else 1
    return {j: sign * i for j, i in ints if i}


# ── stoichiometry over store paths ──────────────────────────────────────


def _written_rows(proc, topo, row_of) -> dict[str, list[int]]:
    return {
        port: [row_of[p] for p in as_paths(topo[port])]
        for port, s in proc.ports_schema().items()
        if s.role in (PortRole.EVOLVED, PortRole.EXCLUSIVE)
    }


@dataclasses.dataclass(frozen=True)
class CompositeStoichiometry:
    """``N`` over a composite's store paths.

    ``matrix`` is ``(n_paths, n_channels)`` and ``channels`` names each
    column ``"<process>/<reaction>"``. ``described`` are the rows every
    writer moves through its ``N`` and nothing else; ``undescribed`` are the
    rows some writer moves outside it — an opaque process, a rate rule — so
    a law touching one is a conjecture ``N`` cannot settle on its own. Rows
    in neither are written by no continuous process. ``opaque`` names the
    processes that declare no ``N`` at all.
    """

    matrix: np.ndarray
    channels: tuple
    described: frozenset
    undescribed: frozenset
    opaque: tuple

    @property
    def n_paths(self) -> int:
        return self.matrix.shape[0]

    def null_space(self, rows=None) -> list[dict[int, int]]:
        """Integer basis of ``{L : L·N = 0}`` supported on ``rows`` (all by
        default): every candidate conservation law the declared wiring
        allows, each as ``{row: coefficient}``."""
        rows = list(range(self.n_paths) if rows is None else rows)
        if self.matrix.shape[1] == 0:
            return [{r: 1} for r in rows]
        position = {r: k for k, r in enumerate(rows)}
        # L·N = 0 is Nᵀ·Lᵀ = 0: one equation per channel over the rows.
        equations = []
        for c in range(self.matrix.shape[1]):
            eq = {
                position[r]: float(self.matrix[r, c])
                for r in np.flatnonzero(self.matrix[:, c])
                if r in position
            }
            if eq:
                equations.append(eq)
        return [
            {rows[k]: v for k, v in integerize(vec).items()}
            for vec in rational_null_space(equations, len(rows))
        ]

    def exact_laws(self) -> list[dict[int, int]]:
        """Integer basis of the laws ``N`` settles by itself: those over rows
        nothing moves outside it."""
        keep = [r for r in range(self.n_paths) if r not in self.undescribed]
        return self.null_space(keep)


def composite_stoichiometry(composite, keys=None) -> CompositeStoichiometry:
    """Composite-level ``N`` over store paths.

    Assembled from each continuous process's
    :meth:`~hallsim.process.Process.stoichiometry`, its species mapped
    through the topology so two models sharing a path share a row, with the
    port-to-path unit factor the RHS applies on every write. A row is
    *described* only when every process writing it declares it: a
    hand-written edge adding to an imported species can move it any way it
    likes, so ``N`` alone cannot claim conservation there.
    """
    keys = composite.store_keys() if keys is None else list(keys)
    row_of = {k: i for i, k in enumerate(keys)}
    canon = canonical_units(composite.processes, composite.topology)
    columns, channels, opaque = [], [], []
    status: dict[int, bool] = {}
    for name, proc in composite.continuous_processes().items():
        topo = composite.topology.get(name, {})
        schema = proc.ports_schema()
        written = _written_rows(proc, topo, row_of)
        declared = proc.stoichiometry()
        if declared is None:
            opaque.append(name)
            for rows in written.values():
                for r in rows:
                    status[r] = False
            continue
        factor = {
            port: [
                conversion_factor(schema[port].units, canon.get(p, ""))
                for p in as_paths(topo[port])
            ]
            for port in written
        }
        rows_of = {sp: written.get(sp, []) for sp in declared["species"]}
        covered = {r for rows in rows_of.values() for r in rows}
        for rows in written.values():
            for r in rows:
                if r in covered:
                    status.setdefault(r, True)
                else:
                    status[r] = False
        for c, rid in enumerate(declared["reactions"]):
            column = np.zeros(len(keys))
            for i, sp in enumerate(declared["species"]):
                coeff = declared["matrix"][i][c]
                if coeff:
                    for r, fac in zip(rows_of[sp], factor.get(sp, ())):
                        column[r] += coeff * fac
            columns.append(column)
            channels.append(f"{name}/{rid}")
    matrix = np.stack(columns, axis=1) if columns else np.zeros((len(keys), 0))
    return CompositeStoichiometry(
        matrix=matrix,
        channels=tuple(channels),
        described=frozenset(r for r, ok in status.items() if ok),
        undescribed=frozenset(r for r, ok in status.items() if not ok),
        opaque=tuple(opaque),
    )


def composite_moieties(composite, keys=None) -> list[dict[str, int]]:
    """The conserved moieties the declared stoichiometry settles exactly,
    each as ``{store path: integer coefficient}`` — the chemistry statement
    of a conservation law, the same for every parameter value. Only over
    paths some process integrates: a slot nothing writes (an algebraic
    output, an unwired input) is constant, not conserved."""
    keys = composite.store_keys() if keys is None else list(keys)
    structure = composite_stoichiometry(composite, keys)
    written = structure.described | structure.undescribed
    return [
        {keys[r]: c for r, c in sorted(law.items())}
        for law in structure.exact_laws()
        if set(law) <= written
    ]


# ── Jacobian sparsity and colouring ─────────────────────────────────────


def colour_columns(n: int, rows, cols):
    """Greedy distance-1 colouring of the column intersection graph: two
    columns conflict when some row holds both, and columns of one colour
    can share a forward-mode seed (Curtis, Powell and Reid 1974). Returns
    ``(colour per column, n_colours)``; a column with no entries takes
    colour 0."""
    import scipy.sparse as sp

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if n == 0:
        return np.zeros(0, dtype=np.int32), 0
    if rows.size == 0:
        return np.zeros(n, dtype=np.int32), 1
    pattern = sp.csc_matrix(
        (np.ones(rows.size, dtype=np.int32), (rows, cols)), shape=(n, n)
    )
    conflicts = (pattern.T @ pattern).tocsr()
    indptr, indices = conflicts.indptr, conflicts.indices
    colour = np.full(n, -1, dtype=np.int32)
    for j in range(n):
        used = colour[indices[indptr[j] : indptr[j + 1]]]
        used = used[used >= 0]
        if used.size == 0:
            colour[j] = 0
            continue
        present = np.zeros(int(used.max()) + 2, dtype=bool)
        present[used] = True
        colour[j] = int(np.argmin(present))
    return colour, int(colour.max()) + 1


@dataclasses.dataclass(frozen=True)
class JacobianPattern:
    """Where ``∂f/∂y`` can be non-zero over ``n`` states, with a colouring
    of its columns. ``n_colours`` forward passes form the whole Jacobian;
    a dense pattern needs ``n``, which is exactly ``jacfwd``."""

    n: int
    rows: np.ndarray
    cols: np.ndarray
    colour: np.ndarray
    n_colours: int

    @classmethod
    def from_entries(cls, n: int, rows, cols) -> "JacobianPattern":
        rows = np.asarray(rows, dtype=np.int32).reshape(-1)
        cols = np.asarray(cols, dtype=np.int32).reshape(-1)
        if rows.size:
            pairs = np.unique(np.stack([cols, rows], axis=1), axis=0)
            cols, rows = pairs[:, 0], pairs[:, 1]
        colour, n_colours = colour_columns(n, rows, cols)
        return cls(n, rows, cols, colour, n_colours)

    @classmethod
    def dense(cls, n: int) -> "JacobianPattern":
        grid = np.arange(n)
        return cls.from_entries(n, np.tile(grid, n), np.repeat(grid, n))

    def with_entries(self, rows, cols) -> "JacobianPattern":
        return self.from_entries(
            self.n,
            np.concatenate([self.rows, np.asarray(rows, dtype=np.int32)]),
            np.concatenate([self.cols, np.asarray(cols, dtype=np.int32)]),
        )

    def with_laws(self, laws) -> "JacobianPattern":
        """The pattern of ``f + LᵀL·y``: a full block on each law's
        support."""
        rows, cols = [], []
        for law in np.asarray(laws, dtype=float).reshape(-1, self.n):
            support = np.flatnonzero(np.abs(law) > 1e-12)
            rows.append(np.repeat(support, support.size))
            cols.append(np.tile(support, support.size))
        if not rows:
            return self
        return self.with_entries(np.concatenate(rows), np.concatenate(cols))

    @property
    def nnz(self) -> int:
        return int(self.rows.size)


def jacobian_pattern(composite, keys=None) -> JacobianPattern:
    """Sparsity of the composite's flat RHS ``∂f/∂y`` over ``keys``, from
    each process's :meth:`~hallsim.process.Process.port_dependencies`.

    An ASSIGNED path is overwritten by its rule before the derivative pass,
    so a dependence on it is a dependence on whatever the rule reads,
    followed through chained rules; the column of the assigned slot itself
    is empty.
    """
    keys = composite.store_keys() if keys is None else list(keys)
    idx = {k: i for i, k in enumerate(keys)}
    evolved: dict[int, set[int]] = defaultdict(set)
    assigned: dict[int, set[int]] = defaultdict(set)
    for name, proc in composite.continuous_processes().items():
        topo = composite.topology.get(name, {})
        schema = proc.ports_schema()
        for port, deps in proc.port_dependencies().items():
            cols = {idx[p] for d in deps for p in as_paths(topo[d])}
            target = (
                assigned if schema[port].role is PortRole.ASSIGNED else evolved
            )
            for path in as_paths(topo[port]):
                target[idx[path]] |= cols

    resolved: dict[int, set[int]] = {}

    def resolve(a, trail=()):
        if a in resolved:
            return resolved[a]
        if a in trail:
            raise StructureError(
                f"algebraic cycle through {keys[a]!r}: an ASSIGNED path "
                "depends on itself"
            )
        out: set[int] = set()
        for c in assigned[a]:
            out |= resolve(c, trail + (a,)) if c in assigned else {c}
        resolved[a] = out
        return out

    rows, cols = [], []
    for r, deps in evolved.items():
        full: set[int] = set()
        for c in deps:
            full |= resolve(c) if c in assigned else {c}
        rows.extend([r] * len(full))
        cols.extend(sorted(full))
    return JacobianPattern.from_entries(len(keys), rows, cols)


def compressed_jacobian(fn, y, pattern: JacobianPattern):
    """``∂fn/∂y`` at ``y`` as a dense ``(n, n)`` array, formed in
    ``pattern.n_colours`` forward passes rather than ``n``. Entries outside
    the pattern are exactly zero. Traces and differentiates like
    ``jax.jacfwd``."""
    y = jnp.asarray(y)
    n = pattern.n
    seeds = np.zeros((pattern.n_colours, n))
    seeds[pattern.colour, np.arange(n)] = 1.0
    tangents = jax.vmap(lambda s: jax.jvp(fn, (y,), (s,))[1])(
        jnp.asarray(seeds, dtype=y.dtype)
    )
    if pattern.rows.size == 0:
        return jnp.zeros((n, n), dtype=tangents.dtype)
    values = tangents[pattern.colour[pattern.cols], pattern.rows]
    return (
        jnp.zeros((n, n), dtype=values.dtype)
        .at[pattern.rows, pattern.cols]
        .set(values)
    )


def check_pattern(fn, y, pattern: JacobianPattern, keys=None, *, rtol=1e-6):
    """Raise unless the Jacobian ``pattern`` yields agrees with ``fn``'s own
    linearisation along one random direction at ``y``; returns that
    Jacobian, so the caller need not form it twice.

    A pattern missing an entry puts two entries on one seed, and the row
    disagrees with probability one. Concrete values only; it is the guard
    between a wrong declaration and a silently wrong Jacobian, run once per
    composite by :func:`hallsim.steady_state.conservation_laws`.
    """
    y = jnp.asarray(y)
    direction = jnp.asarray(
        np.random.default_rng(0).standard_normal(y.shape), dtype=y.dtype
    )
    jac = compressed_jacobian(fn, y, pattern)
    exact = jax.jvp(fn, (y,), (direction,))[1]
    err = np.asarray(jnp.abs(jac @ direction - exact))
    scale = np.asarray(jnp.abs(jac) @ jnp.abs(direction) + jnp.abs(exact))
    floor = float(np.max(scale)) * 1e-12 if scale.size else 0.0
    bad = np.flatnonzero(err > rtol * (scale + floor))
    if bad.size:
        names = [keys[i] if keys else str(i) for i in bad[:8]]
        more = f" and {bad.size - 8} more" if bad.size > 8 else ""
        raise StructureError(
            "the Jacobian sparsity read off reaction_channels(), rate_rules() "
            "and assignment_rules() disagrees with the composite's own "
            f"derivative on {names}{more}: a process writing one of these "
            "reads a port its symbolic form does not name. Fix that "
            "declaration; port_dependencies() shows what each process claims."
        )
    return jac


# ── the field as sympy ──────────────────────────────────────────────────


def parameter_field(proc, name: str) -> str:
    """The dotted field naming parameter symbol ``name`` on ``proc``, in the
    convention of :class:`~hallsim.calibration.ParameterRef`:
    ``parameters.<name>`` for an imported model's constant, else the field
    itself."""
    params = getattr(proc, "parameters", None)
    if isinstance(params, dict) and name in params:
        return f"parameters.{name}"
    return name


@dataclasses.dataclass(frozen=True)
class SymbolicField:
    """The composite's vector field as sympy over its store paths.

    ``derivatives`` maps each integrated path to ``dy/dt`` with every
    assigned path substituted by its rule, ``assigned`` maps each ASSIGNED
    path to that resolved rule. State enters as ``Symbol(path)``, a
    parameter as ``Symbol("<process>.<field>")`` in the dotted form
    :class:`~hallsim.calibration.ParameterRef` uses, time as
    :data:`~hallsim.sbml_math.TIME` on the composite clock. A process with
    no symbolic form contributes an undefined function ``<process>__<port>``
    of the paths it reads. ``parameters`` holds each parameter's value.
    """

    derivatives: dict
    assigned: dict
    parameters: dict
    opaque: tuple


def exact(value):
    """A float coefficient as the rational it stands for, so ``1.0`` does
    not turn every ratio into ``1.0*``."""
    return sympy.nsimplify(value, rational=True)


def symbolic_field(composite, keys=None) -> SymbolicField:
    """Assemble :class:`SymbolicField` for ``composite``. Concrete values
    only: parameter values are read, and a reconciled clock's scale is
    applied as the RHS applies it."""
    keys = composite.store_keys() if keys is None else list(keys)
    canon = canonical_units(composite.processes, composite.topology)
    derivatives: dict[str, object] = {}
    assigned_raw: dict[str, object] = {}
    parameters: dict[str, object] = {}
    opaque: list[str] = []

    for name, proc in composite.continuous_processes().items():
        topo = composite.topology.get(name, {})
        schema = proc.ports_schema()
        values = proc.symbol_values()
        scale = float(getattr(proc, "time_scale", 1.0))

        def read_symbol(port):
            paths = as_paths(topo[port])
            if len(paths) != 1:
                raise StructureError(
                    f"{name!r} reads port {port!r}, which binds {len(paths)} "
                    "store paths; a symbolic form reads one"
                )
            fac = conversion_factor(
                canon.get(paths[0], ""), schema[port].units
            )
            sym = sympy.Symbol(paths[0])
            return sym * fac if fac != 1.0 else sym

        def bind(expr):
            expr = sympy.sympify(expr)
            subs = {TIME: TIME * scale} if scale != 1.0 else {}
            for sym in expr.free_symbols:
                if sym is TIME:
                    continue
                if sym.name in schema:
                    subs[sym] = read_symbol(sym.name)
                elif sym.name in values:
                    full = sympy.Symbol(
                        f"{name}.{parameter_field(proc, sym.name)}"
                    )
                    parameters[full.name] = values[sym.name]
                    subs[sym] = full
                else:
                    raise StructureError(
                        f"{name!r}: {sym.name!r} in its symbolic form is "
                        "neither a port nor a parameter it lists in "
                        "symbol_values()"
                    )
            return expr.xreplace(subs) if subs else expr

        def write_factor(port, path):
            return conversion_factor(schema[port].units, canon.get(path, ""))

        reads = sorted(
            path
            for port, s in schema.items()
            if not (s.role is PortRole.EVOLVED and not s.reads_value)
            for path in as_paths(topo[port])
        )
        read_args = [sympy.Symbol(p) for p in reads]
        written = [
            port
            for port, s in schema.items()
            if s.role in (PortRole.EVOLVED, PortRole.EXCLUSIVE)
        ]
        channels = proc.reaction_channels()
        for port in written:
            for path in as_paths(topo[port]):
                derivatives.setdefault(path, sympy.Integer(0))
        if channels is None and written:
            opaque.append(name)
            for port in written:
                for path in as_paths(topo[port]):
                    derivatives[path] += sympy.Function(f"{name}__{port}")(
                        *read_args
                    )
        # N is authoritative where it describes a port (a frozen row is
        # zero there); the channel pairs stand for the rest.
        coefficient = {
            (port, j): coeff
            for j, channel in enumerate(channels or ())
            for port, coeff in channel.stoichiometry
        }
        declared = proc.stoichiometry() if channels is not None else None
        for i, sp in enumerate(declared["species"] if declared else ()):
            for j, coeff in enumerate(declared["matrix"][i]):
                coefficient[(sp, j)] = coeff
        for j, channel in enumerate(channels or ()):
            law = bind(channel.rate_law) * exact(scale)
            for port in written:
                coeff = coefficient.get((port, j), 0.0)
                if not coeff:
                    continue
                for path in as_paths(topo[port]):
                    derivatives[path] += (
                        exact(coeff) * exact(write_factor(port, path)) * law
                    )
        for port, expr in proc.rate_rules():
            if port not in written:
                continue
            rule = bind(expr) * exact(scale)
            for path in as_paths(topo[port]):
                derivatives[path] += exact(write_factor(port, path)) * rule
        rules = dict(proc.assignment_rules())
        for port, s in schema.items():
            if s.role is not PortRole.ASSIGNED:
                continue
            for path in as_paths(topo[port]):
                if port in rules:
                    assigned_raw[path] = exact(
                        write_factor(port, path)
                    ) * bind(rules[port])
                else:
                    if name not in opaque:
                        opaque.append(name)
                    assigned_raw[path] = sympy.Function(f"{name}__{port}")(
                        *read_args
                    )

    # Rules are acyclic (the composite rejects a cycle), so substituting
    # until nothing changes resolves every chain.
    resolved = dict(assigned_raw)
    for _ in range(len(resolved) + 1):
        subs = {sympy.Symbol(p): e for p, e in resolved.items()}
        new = {p: sympy.sympify(e).xreplace(subs) for p, e in resolved.items()}
        if new == resolved:
            break
        resolved = new
    subs = {sympy.Symbol(p): e for p, e in resolved.items()}
    derivatives = {
        p: sympy.sympify(e).xreplace(subs) for p, e in derivatives.items()
    }
    return SymbolicField(derivatives, resolved, parameters, tuple(opaque))
