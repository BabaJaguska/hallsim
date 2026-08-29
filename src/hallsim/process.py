"""Process — the fundamental building block of composable simulations.

An Equinox module owning the derivatives of a subset of state variables. It
declares typed *ports* — named connection points with roles — and implements
``derivative`` (CONTINUOUS), ``update`` (DISCRETE), or ``condition`` +
``handler`` (EVENT), receiving only the ports it declared. Parameters are JAX
arrays, so a Process differentiates, JIT-compiles, and vmaps.

>>> class Decay(Process):
...     rate: float = 0.1
...
...     def ports_schema(self) -> dict[str, Port]:
...         return {"x": Port(role=PortRole.EVOLVED, default=1.0)}
...
...     def derivative(self, t, state):
...         return {"x": -self.rate * state["x"]}
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

import equinox as eqx
import jax.numpy as jnp


def _as_traced(value):
    """``value`` with its floats as JAX arrays, or ``None`` if nothing to do.
    Reaches one level into a dict/tuple/list — enough for a parameter dict or a
    per-source tuple of rate constants."""
    if type(value) is float:
        return jnp.asarray(value)
    if type(value) is dict and any(type(v) is float for v in value.values()):
        return {
            k: jnp.asarray(v) if type(v) is float else v
            for k, v in value.items()
        }
    if type(value) in (tuple, list) and any(type(v) is float for v in value):
        items = [jnp.asarray(v) if type(v) is float else v for v in value]
        return type(value)(items)
    return None


def calibratable(
    default,
    *,
    clamp: "tuple[float, float] | None" = None,
    description: str = "",
):
    """Declare a Process field as a fittable mechanism parameter::

        k_act: float = calibratable(0.02, description="edge strength")
        K_mtor: float = 4.0  # measurement-grounded — stays fixed

    Marked fields surface through :meth:`Composite.calibration_targets`; plain
    defaults stay out of the calibration surface. ``clamp`` defaults to two
    orders of magnitude around the current value.
    """
    return eqx.field(
        default=default,
        metadata={
            "calibratable": True,
            "clamp": clamp,
            "description": description,
        },
    )


def read_param(proc, field: str):
    """Current value at ``field`` on ``proc``, in the dotted convention shared
    by :attr:`hallsim.hallmarks.ParameterMapping.param_name` and
    :class:`hallsim.calibration.ParameterRef`: ``"alpha"`` reads ``proc.alpha``,
    ``"parameters.<key>"`` one entry of a parameters dict."""
    if "." in field:
        field_name, key = field.split(".", 1)
        return getattr(proc, field_name)[key]
    return getattr(proc, field)


def write_param(proc, field: str, value):
    """A copy of ``proc`` with ``field`` set to ``value``, in the same dotted
    convention as :func:`read_param`.

    The one implementation of "change a parameter". Reach it as
    :meth:`Process.with_param`, :meth:`hallsim.composite.Composite.with_params`
    or :meth:`hallsim.calibration.CalibrationProblem.with_overrides` —
    whichever object is in hand. Hand-rolling ``eqx.tree_at`` instead is what
    the calibration guard rejects: an edit to a *fitted* field is overwritten by
    the next substitution, and an ablation that silently does nothing looks
    exactly like an edge with no influence.
    """
    # tree_at rebuilds via tree_unflatten, which skips __check_init__.
    coerced = _as_traced(value)
    if coerced is not None:
        value = coerced
    if "." in field:
        field_name, key = field.split(".", 1)
        current = getattr(proc, field_name)
        if not isinstance(current, dict):
            raise TypeError(
                f"Dotted field {field!r} requires {field_name!r} to be a dict "
                f"on {type(proc).__name__}; got {type(current).__name__}"
            )
        if key not in current:
            raise KeyError(
                f"Key {key!r} not in {field_name}; "
                f"available: {sorted(current.keys())}"
            )
        return eqx.tree_at(
            lambda p, fn=field_name, k=key: getattr(p, fn)[k], proc, value
        )
    if not hasattr(proc, field):
        raise AttributeError(
            f"{type(proc).__name__} has no field {field!r}. Fittable and "
            f"settable fields: {sorted(_settable_fields(proc))}"
        )
    return eqx.tree_at(lambda p, pn=field: getattr(p, pn), proc, value)


def _settable_fields(proc) -> list[str]:
    """Field names ``write_param`` accepts, for an error message that answers
    the question rather than only refusing it."""
    names = [
        f.name
        for f in dataclasses.fields(proc)
        if not f.metadata.get("static", False)
    ]
    params = getattr(proc, "parameters", None)
    if isinstance(params, dict):
        names += [f"parameters.{k}" for k in sorted(params)]
    return names


# ---------------------------------------------------------------------------
# Port role enum
# ---------------------------------------------------------------------------


class ProcessKind(enum.Enum):
    """What kind of update rule a process uses: CONTINUOUS (the default)
    implements ``derivative(t, state) -> dy/dt``, solved by Diffrax; DISCRETE
    implements ``update(t, state) -> delta`` every ``dt_step``; EVENT pairs
    ``condition`` with a ``handler`` that fires once on a False→True crossing.
    """

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    EVENT = "event"


class PortRole(enum.Enum):
    """How a port participates. Write semantics are validated at composition
    time.

    INPUT read-only, no derivative. EVOLVED contributes a derivative, and
    several processes writing one path sum. EXCLUSIVE is sole owner of that
    derivative; a second writer raises. LATCHED is written only by
    DISCRETE/EVENT processes and read as constant within a macro step.

    ASSIGNED is an algebraic output: the process *computes* the path's value
    each step via ``assign`` instead of integrating it — a cross-process
    assignment rule, evaluated before the derivative pass so integrated
    processes read the fresh value. Sole owner, like EXCLUSIVE.
    """

    INPUT = "input"
    EVOLVED = "evolved"
    EXCLUSIVE = "exclusive"
    LATCHED = "latched"
    ASSIGNED = "assigned"


# ---------------------------------------------------------------------------
# Port descriptor
# ---------------------------------------------------------------------------


class Port:
    """A single named connection point on a Process.

    ``role`` is a :class:`PortRole`; ``default`` seeds the initial store, and
    ``None`` abstains — the port writes to the path but claims nothing about
    where it starts, which is what a coupling edge wiring into another model's
    species wants. A path every claimant abstains on has no initial value and
    raises. ``units`` (``"uM"``) and ``ontology``
    (``{"GO": "GO:0006915"}``) feed the validator and LLM-assisted composition.

    ``reads_value`` applies to EVOLVED ports only: set it False for a **pure
    source**, one whose contribution depends on the process's other inputs
    rather than the path it writes (a Hill-gated cross-model edge, a running
    integral), so the graph analyzer doesn't infer a spurious cycle.
    """

    __slots__ = (
        "role",
        "default",
        "units",
        "description",
        "ontology",
        "reads_value",
    )

    def __init__(
        self,
        role: PortRole = PortRole.EVOLVED,
        default: float | jnp.ndarray | None = 0.0,
        units: str = "",
        description: str = "",
        ontology: dict[str, str] | None = None,
        reads_value: bool = True,
    ) -> None:
        self.role = role
        self.default = default
        self.units = units
        self.description = description
        self.ontology = ontology or {}
        self.reads_value = reads_value

    def __repr__(self) -> str:
        return (
            f"Port(role={self.role.value!r}, default={self.default}, "
            f"units={self.units!r})"
        )


# ---------------------------------------------------------------------------
# Process base class
# ---------------------------------------------------------------------------


class Process(eqx.Module):
    """Abstract base for composable biological processes.

    Subclasses implement ``ports_schema()`` plus whatever their ``kind``
    requires (see :class:`ProcessKind`), and may override ``metadata()`` for
    LLM-assisted composition. ``timescale`` (seconds) is what the Scheduler
    groups on — processes within ~100x share a solve; ``dt_step`` spaces a
    DISCRETE process's updates.
    """

    kind: ProcessKind = ProcessKind.CONTINUOUS
    timescale: float | None = None
    dt_step: float | None = None

    # Folded into metadata() when set. Plain class attributes, not fields, so
    # they add nothing to the traced pytree.
    hallmark = None
    reference = None
    description = None

    # Read by the Scheduler in plain Python (grouping, update spacing), never
    # inside the traced computation — structure, not dynamics.
    _PYTHON_FIELDS = frozenset({"timescale", "dt_step"})

    def __check_init__(self):
        """Coerce float parameters to JAX arrays, dict and tuple fields
        included (an imported model's ``parameters`` dict is the reason).

        A Python float is a *static* leaf to ``eqx.filter_jit``, so its value
        bakes into the compiled solve and every distinct value recompiles; as
        arrays, one executable serves them all. Ints stay put — in a container
        they are indices. ``__check_init__``, not ``__post_init__``: equinox
        always runs it and skips it on ``tree_unflatten``.
        """
        for f in dataclasses.fields(self):
            if f.metadata.get("static") or f.name in self._PYTHON_FIELDS:
                continue
            value = getattr(self, f.name, None)
            coerced = _as_traced(value)
            if coerced is not None:
                object.__setattr__(self, f.name, coerced)

    # --- Interface: CONTINUOUS -----------------------------------------------

    def ports_schema(self) -> dict[str, Port]:
        """``{port_name: Port(...)}`` — every port this process reads or
        writes must be declared here."""
        raise NotImplementedError

    def derivative(
        self, t: float, state: dict[str, jnp.ndarray]
    ) -> dict[str, jnp.ndarray]:
        """Time derivatives (CONTINUOUS processes).

        ``state`` maps port name → current value, restricted to the ports
        declared in ``ports_schema``. Returns port name → dy/dt; only EVOLVED
        and EXCLUSIVE ports may appear.
        """
        raise NotImplementedError

    # --- Interface: ASSIGNED (algebraic) -------------------------------------

    def assign(
        self, t: float, state: dict[str, jnp.ndarray]
    ) -> dict[str, jnp.ndarray]:
        """``{port: value}`` for this process's ASSIGNED ports — a value, not a
        derivative. Evaluated in dependency order before each derivative pass,
        so integrated processes read the fresh result."""
        return {}

    def discontinuity_times(self) -> tuple[float, ...]:
        """Composite-clock times where this process's contribution jumps — a
        forcing pulse's edges, a timed step. The Scheduler passes them as
        ``jump_ts`` so the solver lands on each exactly, instead of resolving
        it by step rejection, which also keeps the gradient clean across it."""
        return ()

    # --- Interface: DISCRETE -------------------------------------------------

    def update(
        self, t: float, state: dict[str, jnp.ndarray]
    ) -> dict[str, jnp.ndarray]:
        """State delta (DISCRETE processes), called every ``dt_step`` seconds.
        Additive: ``new_state = old_state + delta``."""
        raise NotImplementedError

    # --- Interface: EVENT ----------------------------------------------------

    def condition(self, t: float, state: dict[str, jnp.ndarray]) -> bool:
        """Event trigger (EVENT processes). The Scheduler tracks the previous
        value and fires :meth:`handler` only on a False→True transition."""
        raise NotImplementedError

    def handler(
        self, t: float, state: dict[str, jnp.ndarray]
    ) -> dict[str, jnp.ndarray]:
        """Additive delta applied once when ``condition`` becomes True."""
        raise NotImplementedError

    # --- Metadata ------------------------------------------------------------

    def calibratable_params(self) -> list:
        """Mechanism parameters this Process exposes as fittable — every field
        declared with :func:`calibratable`, at its current value.

        Subclasses with a non-field parameter surface (``SBMLProcess``'s
        constants dict) extend this. Safe to expose a hallmark target too:
        :meth:`Composite.calibration_targets` subtracts those.
        """
        from hallsim.calibration import CalibratableParam, default_clamp

        out: list = []
        for f in dataclasses.fields(self):
            if not f.metadata.get("calibratable"):
                continue
            value = float(getattr(self, f.name))
            out.append(
                CalibratableParam(
                    process_name="",
                    field=f.name,
                    default=value,
                    clamp=f.metadata.get("clamp") or default_clamp(value),
                    description=f.metadata.get("description", ""),
                )
            )
        return out

    def metadata(self) -> dict[str, Any]:
        """Structured metadata for discovery and LLM-assisted composition.
        Override to add pathway IDs, GO terms, SBML annotations, etc."""
        meta = {
            "name": type(self).__name__,
            "kind": self.kind.value,
            "ports": {
                name: {
                    "role": port.role.value,
                    "units": port.units,
                    "description": port.description,
                    "ontology": port.ontology,
                }
                for name, port in self.ports_schema().items()
            },
        }
        for key in ("hallmark", "reference", "description"):
            value = getattr(self, key, None)
            if value:
                meta[key] = value
        if self.timescale is not None:
            meta["timescale"] = self.timescale
        if self.dt_step is not None:
            meta["dt_step"] = self.dt_step
        return meta

    def coupling_structure(self) -> dict | None:
        """Equation structure for :mod:`hallsim.coupling_wiring`:
        ``param_constant``, ``param_sbo``, ``variables`` (dynamic quantity
        ids), ``rules`` (the algebraic dependency graph), ``boundary``.

        ``None`` for an *opaque* process — a hand-coded or neural one with no
        declared rule graph — which skips the structural check. Format
        importers override it, so the checker stays format-agnostic."""
        return None

    def stoichiometry(self) -> dict | None:
        """Species × reaction stoichiometry ``N``, as ``{"species": (port
        name, ...), "reactions": (id, ...), "matrix": ((coeff, ...), ...)}``
        with one row per species.

        ``N`` is the process's wiring, independent of every rate constant, so
        it settles the conserved moieties exactly — where the null space of a
        Jacobian only ever says "nothing much is moving *here*, at *these*
        parameters". Declare it whenever the dynamics really are
        ``dy/dt = N·v(y)``.

        ``None`` means undeclared, not "no conservation": callers fall back to
        inferring the moieties numerically (see
        :func:`hallsim.steady_state.conservation_laws`)."""
        return None

    # --- Helpers -------------------------------------------------------------

    def with_param(self, field: str, value) -> "Process":
        """A copy of this process with one parameter changed.

        ``field`` is a field name (``"k_act"``) or a dotted entry of a
        parameters dict (``"parameters.kdeg"``)::

            edge = edge.with_param("k_act", 0.0)      # ablate an edge
            proc = proc.with_param("parameters.k", 2.0)

        For a whole composite use
        :meth:`hallsim.composite.Composite.with_params`; inside a calibration
        use :meth:`hallsim.calibration.CalibrationProblem.with_overrides`, which
        also wins over the fitted iterate. All three are this one call.
        """
        return write_param(self, field, value)

    def ports_with_role(self, role: PortRole) -> dict[str, Port]:
        """Subset of ``ports_schema()`` filtered by port role."""
        return {k: v for k, v in self.ports_schema().items() if v.role == role}

    def output_port_names(self) -> set[str]:
        """Names of ports that write a store path (EVOLVED, EXCLUSIVE,
        LATCHED, or ASSIGNED)."""
        writes = (
            PortRole.EVOLVED,
            PortRole.EXCLUSIVE,
            PortRole.LATCHED,
            PortRole.ASSIGNED,
        )
        return {k for k, v in self.ports_schema().items() if v.role in writes}
