"""Process — the fundamental building block of composable simulations.

A Process is an Equinox module that computes time derivatives for a subset
of state variables.  It declares typed *ports* — named connection points
with roles (input, evolved, exclusive) — and implements a ``derivative``
method that receives only the port values it declared.

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

import enum
from typing import Any

import equinox as eqx
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Port role enum
# ---------------------------------------------------------------------------

class PortRole(enum.Enum):
    """How a port participates in the ODE system.

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
    """

    INPUT = "input"
    EVOLVED = "evolved"
    EXCLUSIVE = "exclusive"


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
    """

    __slots__ = ("role", "default", "units", "description", "ontology")

    def __init__(
        self,
        role: PortRole = PortRole.EVOLVED,
        default: float | jnp.ndarray = 0.0,
        units: str = "",
        description: str = "",
        ontology: dict[str, str] | None = None,
    ) -> None:
        self.role = role
        self.default = default
        self.units = units
        self.description = description
        self.ontology = ontology or {}

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
    - ``derivative(t, state)`` — compute dy/dt for evolved/exclusive ports.

    Optionally override:
    - ``metadata()`` — structured info for LLM-assisted composition.
    """

    # --- Interface -----------------------------------------------------------

    def ports_schema(self) -> dict[str, Port]:
        """Return a dict of ``{port_name: Port(...)}``.

        Every port this process reads or writes must be declared here.
        """
        raise NotImplementedError

    def derivative(self, t: float, state: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        """Compute time derivatives.

        Parameters
        ----------
        t:
            Current simulation time.
        state:
            Dict mapping port names → current values (only ports declared
            in ``ports_schema`` are provided).

        Returns
        -------
        Dict mapping port names → dy/dt values.  Only evolved and exclusive
        ports should appear in the output.
        """
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        """Structured metadata for discovery and LLM-assisted composition.

        Override to provide pathway IDs, GO terms, species descriptions,
        SBML annotations, coupling documentation, etc.
        """
        return {
            "name": type(self).__name__,
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

    # --- Helpers -------------------------------------------------------------

    def evolved_ports(self) -> dict[str, Port]:
        """Ports with role EVOLVED."""
        return {k: v for k, v in self.ports_schema().items() if v.role == PortRole.EVOLVED}

    def exclusive_ports(self) -> dict[str, Port]:
        """Ports with role EXCLUSIVE."""
        return {k: v for k, v in self.ports_schema().items() if v.role == PortRole.EXCLUSIVE}

    def input_ports(self) -> dict[str, Port]:
        """Ports with role INPUT."""
        return {k: v for k, v in self.ports_schema().items() if v.role == PortRole.INPUT}

    def output_port_names(self) -> set[str]:
        """Names of ports that produce derivatives (evolved + exclusive)."""
        return {
            k for k, v in self.ports_schema().items()
            if v.role in (PortRole.EVOLVED, PortRole.EXCLUSIVE)
        }
