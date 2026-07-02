"""Process — the fundamental building block of composable simulations.

A Process is an Equinox module that computes time derivatives for a subset
of state variables.  It declares typed *ports* — named connection points
with roles (input, evolved, exclusive, latched) — and implements a
``derivative`` method that receives only the port values it declared.

Three process kinds are supported:

- **CONTINUOUS** (default): computes ``derivative(t, state) -> dy/dt``.
- **DISCRETE**: computes ``update(t, state) -> delta_state``, called at
  fixed intervals specified by ``dt_step``.
- **EVENT**: declares a ``condition(t, state) -> bool`` and a
  ``handler(t, state) -> delta_state``, fired when condition crosses
  from False to True.

Because Process is an Equinox module, its parameters are JAX arrays by
default and can be differentiated, JIT-compiled, and vmapped.

Example
-------
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


def calibratable(
    default,
    *,
    clamp: "tuple[float, float] | None" = None,
    description: str = "",
):
    """Declare a Process field as a fittable mechanism parameter.

    Use in place of a plain default at the field-declaration site::

        class MyEdge(Process):
            k_act: float = calibratable(
                0.02, description="edge strength; fit vs the NFKBIA reporter"
            )
            K_mtor: float = 4.0  # measurement-grounded — stays fixed

    Every field marked this way is discovered automatically by
    :meth:`Process.calibratable_params` and surfaced through
    :meth:`hallsim.composite.Composite.calibration_targets`. Fields left
    as plain defaults stay out of the calibration surface. ``clamp``
    defaults to a two-order-of-magnitude box around the current value
    (see :func:`hallsim.calibration.default_clamp`).
    """
    return eqx.field(
        default=default,
        metadata={
            "calibratable": True,
            "clamp": clamp,
            "description": description,
        },
    )


# ---------------------------------------------------------------------------
# Port role enum
# ---------------------------------------------------------------------------


class ProcessKind(enum.Enum):
    """What kind of update rule a process uses.

    CONTINUOUS
        Computes ``derivative(t, state) -> dy/dt``.  Solved by an ODE
        integrator (Diffrax).  This is the default.

    DISCRETE
        Computes ``update(t, state) -> delta_state``.  Called at fixed
        intervals specified by ``dt_step``.  Returns an additive delta.

    EVENT
        Declares ``condition(t, state) -> bool`` and
        ``handler(t, state) -> delta_state``.  The handler fires once
        when the condition crosses from False to True.  Returns a delta.
    """

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    EVENT = "event"


class PortRole(enum.Enum):
    """How a port participates in the simulation.

    INPUT
        Read-only.  The process reads this value but does not contribute
        a derivative.  Must be provided by another process's evolved/exclusive
        port or by the initial state.

    EVOLVED
        This process contributes a derivative for this variable.  Multiple
        processes may write derivatives to the same store path — their
        contributions are summed (additive composition).

    EXCLUSIVE
        This process is the sole owner of this variable's derivative.
        No other process may contribute.  Validated at composition time.

    LATCHED
        Written by discrete or event processes.  Continuous processes may
        read a LATCHED value but treat it as constant within a macro step.
        Only discrete/event processes may write to a LATCHED port.
    """

    INPUT = "input"
    EVOLVED = "evolved"
    EXCLUSIVE = "exclusive"
    LATCHED = "latched"


# ---------------------------------------------------------------------------
# Port descriptor
# ---------------------------------------------------------------------------


class Port:
    """Describes a single named connection point on a Process.

    Parameters
    ----------
    role:
        How this port participates (see :class:`PortRole`).
    default:
        Default initial value (scalar or array).  Used when building
        the initial store if no other process provides a value.
    units:
        Physical units string, e.g. ``"uM"`` or ``"dimensionless"``.
    description:
        Human-readable description for metadata / LLM consumption.
    ontology:
        Optional ontology annotation, e.g. ``{"GO": "GO:0006915"}``.
    reads_value:
        For an EVOLVED port only: whether the additive contribution
        depends on the path's *current* value. Default True (the general
        case — the graph analyzer then treats the port as both writer and
        reader). Set False for a **pure source** — a contribution that
        depends only on the process's other inputs, not on the path it
        writes (e.g. a Hill-gated cross-model edge, or a running integral)
        — so the analyzer doesn't infer a spurious feedback cycle.
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
        default: float | jnp.ndarray = 0.0,
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

    Subclasses must implement:
    - ``ports_schema()`` — declare named ports.

    Depending on ``kind``:
    - CONTINUOUS: implement ``derivative(t, state)`` — compute dy/dt.
    - DISCRETE: implement ``update(t, state)`` — compute delta at intervals.
    - EVENT: implement ``condition(t, state)`` and ``handler(t, state)``.

    Optionally override:
    - ``metadata()`` — structured info for LLM-assisted composition.

    Attributes
    ----------
    kind:
        Process kind (class-level). Default: ``ProcessKind.CONTINUOUS``.
    timescale:
        Characteristic timescale in seconds (class-level, optional).
        Used by the Scheduler to auto-group continuous processes.
        Processes within ~100x of each other share a group.
    dt_step:
        For DISCRETE processes: interval between update calls (seconds).
    """

    kind: ProcessKind = ProcessKind.CONTINUOUS
    timescale: float | None = None
    dt_step: float | None = None

    # --- Interface: CONTINUOUS -----------------------------------------------

    def ports_schema(self) -> dict[str, Port]:
        """Return a dict of ``{port_name: Port(...)}``.

        Every port this process reads or writes must be declared here.
        """
        raise NotImplementedError

    def derivative(
        self, t: float, state: dict[str, jnp.ndarray]
    ) -> dict[str, jnp.ndarray]:
        """Compute time derivatives (CONTINUOUS processes).

        Parameters
        ----------
        t:
            Current simulation time.
        state:
            Dict mapping port names -> current values (only ports declared
            in ``ports_schema`` are provided).

        Returns
        -------
        Dict mapping port names -> dy/dt values.  Only evolved and exclusive
        ports should appear in the output.
        """
        raise NotImplementedError

    # --- Interface: DISCRETE -------------------------------------------------

    def update(
        self, t: float, state: dict[str, jnp.ndarray]
    ) -> dict[str, jnp.ndarray]:
        """Compute a state delta (DISCRETE processes).

        Called every ``dt_step`` seconds by the Scheduler.  Returns an
        additive delta: ``new_state = old_state + delta``.

        Parameters
        ----------
        t:
            Current simulation time.
        state:
            Dict mapping port names -> current values.

        Returns
        -------
        Dict mapping port names -> delta values to add.
        """
        raise NotImplementedError

    # --- Interface: EVENT ----------------------------------------------------

    def condition(self, t: float, state: dict[str, jnp.ndarray]) -> bool:
        """Event trigger condition (EVENT processes).

        Returns ``True`` when the event should fire.  The Scheduler
        tracks the previous value and only fires the handler on a
        False -> True transition.
        """
        raise NotImplementedError

    def handler(
        self, t: float, state: dict[str, jnp.ndarray]
    ) -> dict[str, jnp.ndarray]:
        """Event handler (EVENT processes).

        Called once when ``condition`` crosses from False to True.
        Returns an additive delta.
        """
        raise NotImplementedError

    # --- Metadata ------------------------------------------------------------

    def calibratable_params(self) -> list:
        """Mechanism parameters this Process exposes as fittable.

        Returns a list of :class:`hallsim.calibration.CalibratableParam`
        descriptors with ``process_name=""`` (the Composite-level
        aggregator fills in the namespace) and ``field`` either a plain
        attribute name or ``"parameters.<key>"``.

        Discovery is generic: every field declared with
        :func:`calibratable` is surfaced automatically, with its current
        value as the default and a two-order-of-magnitude clamp unless
        the field supplied its own. A Process exposes a rate constant by
        declaring it ``k: float = calibratable(...)`` instead of a plain
        default — no per-Process enumeration body. Subclasses with a
        non-field parameter surface (e.g. ``SBMLProcess``'s constants
        dict) override and extend ``super().calibratable_params()``.

        :class:`Composite.calibration_targets` subtracts hallmark-
        controlled parameters from the listing before returning to
        callers, so it's safe for a Process to expose a parameter
        that's *also* a hallmark target — the discovery API hides it
        from default Calibrator wiring unless explicitly requested.
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

        Override to provide pathway IDs, GO terms, species descriptions,
        SBML annotations, coupling documentation, etc.
        """
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
        if self.timescale is not None:
            meta["timescale"] = self.timescale
        if self.dt_step is not None:
            meta["dt_step"] = self.dt_step
        return meta

    # --- Helpers -------------------------------------------------------------

    def ports_with_role(self, role: PortRole) -> dict[str, Port]:
        """Subset of ``ports_schema()`` filtered by port role."""
        return {k: v for k, v in self.ports_schema().items() if v.role == role}

    def output_port_names(self) -> set[str]:
        """Names of ports that produce derivatives or deltas (EVOLVED,
        EXCLUSIVE, or LATCHED)."""
        writes = (PortRole.EVOLVED, PortRole.EXCLUSIVE, PortRole.LATCHED)
        return {k for k, v in self.ports_schema().items() if v.role in writes}
