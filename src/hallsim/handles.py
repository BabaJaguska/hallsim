"""Perturbation handles: a named severity that moves parameters across
processes.

A :class:`Handle` carries a list of :class:`ParameterMapping`, each naming a
process, a parameter and how that parameter moves with severity. Applying a
handle builds *new* processes (they are immutable); the transform is
**multiplicative of the current base**, ``base * f(severity)``, so a
calibrated value is scaled, never overwritten. Severity is JAX-traceable and
``jax.grad`` flows through it, for sensitivity analysis and sweeps.

**Severity is an experimental-design knob, not a fittable parameter.** Set it
per condition and fit mechanism parameters with Calibrator; the calibration
layer refuses to fit a handle's own targets unless told otherwise.

The hallmarks of aging ship as one registry of handles,
:data:`hallsim.hallmarks.HALLMARK_REGISTRY`; a gene dosage, a drug or any
other perturbation is another :class:`Handle` in a registry of its own.

>>> treated = with_handles(composite, {"Mitochondrial Dysfunction": 0.7})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import equinox as eqx

from hallsim.process import Process


@dataclass(frozen=True)
class FittableCoeff:
    """A mapping coefficient the Calibrator may fit.

    Stands in for a plain float in a mapping's ``floor``. The Calibrator
    discovers it via a :class:`hallsim.calibration.HandleCoeffRef` and
    substitutes a fitted value per loss evaluation (clamp / prior handled
    like any :class:`hallsim.calibration.ParameterRef`). Outside calibration
    the mapping evaluates at ``init``.
    """

    init: float
    clamp: tuple[float, float] | None = None
    prior: float | None = None
    prior_sigma: float = 0.5
    description: str = ""


@dataclass
class ParameterMapping:
    """Maps a handle's severity to a process parameter value, two forms:

    - **Affine** (``floor`` set): ``base * (floor + slope * severity)``. Use
      ``floor=1`` for a modifier that leaves ``base`` untouched at neutral,
      ``floor=0`` for an input that is off there. ``slope`` is the signed gain
      per unit severity and is required — the neutral point is fixed at
      severity=0, not at either end. Either coefficient may be a
      :class:`FittableCoeff`.
    - **Custom** (``transform`` set): ``transform(severity, base)``, for a dial
      that sets the value directly and ignores ``base`` (``lambda h, _: h``).

    ``process_name`` keys into the composite; ``param_name`` is an attribute
    (``"alpha"``) or dotted path (``"parameters.<key>"``). ``base`` is read
    fresh on each application, so an earlier calibration flows through.
    """

    process_name: str
    param_name: str
    floor: "float | FittableCoeff | None" = None
    slope: float | None = None
    transform: Callable[[Any, Any], Any] | None = None
    description: str = ""

    @property
    def floor_value(self):
        f = self.floor
        return f.init if isinstance(f, FittableCoeff) else f

    @property
    def slope_value(self):
        s = self.slope
        return s.init if isinstance(s, FittableCoeff) else s

    def value(self, severity, base):
        """Resolve the parameter value at ``severity`` given current ``base``."""
        if self.transform is not None:
            return self.transform(severity, base)
        if self.floor is None:
            raise ValueError(
                f"ParameterMapping {self.process_name}.{self.param_name} "
                "needs either an affine `floor` or a `transform`."
            )
        slope = self.slope_value
        if slope is None:
            raise ValueError(
                f"ParameterMapping {self.process_name}.{self.param_name} "
                "is affine but has no `slope`; the signed severity gain is "
                "required (neutral is fixed at severity=0)."
            )
        return base * (self.floor_value + slope * severity)


@dataclass
class Handle:
    """One perturbation handle: a named severity and where it lands.

    Attributes
    ----------
    name:
        Human-readable name (e.g., "Mitochondrial Dysfunction", "Trisomy 21").
    description:
        Brief description of the biology.
    mappings:
        List of ParameterMapping defining how severity affects processes.
    category:
        A family label, e.g. the Lopez-Otin taxonomy for a hallmark of aging.
    references:
        Literature references supporting the parameter mappings.
    """

    name: str
    description: str = ""
    mappings: list[ParameterMapping] = field(default_factory=list)
    category: str = ""
    references: list[str] = field(default_factory=list)

    def apply(
        self,
        processes: dict[str, Process],
        severity: float,
    ) -> dict[str, Process]:
        """New ``{name: Process}`` with this handle applied at ``severity``
        (in [-1, 1], 0 = homeostasis). Untargeted processes pass through
        unchanged; targeted ones are rebuilt via ``eqx.tree_at``. A dotted
        ``param_name`` reaches inside a dict-valued field, which is how a
        handle drives one SBML rate constant."""
        result = dict(processes)
        if self.mappings and not any(
            m.process_name in result for m in self.mappings
        ):
            raise KeyError(
                f"Handle {self.name!r} has no target in this composite: "
                f"none of {sorted({m.process_name for m in self.mappings})} "
                f"is a process here (it has "
                f"{sorted(result)}). Setting its severity would change "
                "nothing and every arm would run identically. Build the "
                "composite with the process the dial drives, or drop the "
                "handle from the condition."
            )
        for mapping in self.mappings:
            pname = mapping.process_name
            if pname not in result:
                continue
            proc = result[pname]
            if "." in mapping.param_name:
                # Dotted form: target a key inside a dict-valued field.
                field_name, key = mapping.param_name.split(".", 1)
                current = getattr(proc, field_name)
                if not isinstance(current, dict):
                    raise TypeError(
                        f"Dotted param_name {mapping.param_name!r} "
                        f"requires {field_name!r} to be a dict on "
                        f"{type(proc).__name__}; got "
                        f"{type(current).__name__}"
                    )
                if key not in current:
                    raise KeyError(
                        f"Key {key!r} not in {pname}.{field_name}; "
                        f"available: {sorted(current.keys())}"
                    )
                base = current[key]
                new_val = mapping.value(severity, base)
                result[pname] = eqx.tree_at(
                    lambda p, fn=field_name, k=key: getattr(p, fn)[k],
                    proc,
                    new_val,
                )
            else:
                base = getattr(proc, mapping.param_name)
                new_val = mapping.value(severity, base)
                result[pname] = eqx.tree_at(
                    lambda p, pn=mapping.param_name: getattr(p, pn),
                    proc,
                    new_val,
                )
        return result

    def summary(
        self,
        severity: float,
        processes: dict[str, Process] | None = None,
    ) -> dict[str, Any]:
        """What each mapping resolves to at ``severity``. With ``processes``,
        reads each target's real base; without, uses ``base=1.0``, which shows
        the transform's shape but not the absolute value."""
        out: dict[str, Any] = {}
        for m in self.mappings:
            base: Any = 1.0
            if processes is not None and m.process_name in processes:
                proc = processes[m.process_name]
                if "." in m.param_name:
                    field_name, key = m.param_name.split(".", 1)
                    base = getattr(proc, field_name)[key]
                else:
                    base = getattr(proc, m.param_name)
            out[f"{m.process_name}.{m.param_name}"] = m.value(severity, base)
        return out


def apply_handles(
    processes: dict[str, Process],
    handles: dict[str, float],
    registry: dict[str, Handle] | None = None,
) -> dict[str, Process]:
    """Apply several handles' severities to a process dict.

    Parameters
    ----------
    processes:
        ``{name: Process}`` from a Composite.
    handles:
        ``{handle_name: severity}`` — which handles to apply.
    registry:
        ``{name: Handle}`` to look the names up in. Defaults to the
        hallmarks of aging, :data:`hallsim.hallmarks.HALLMARK_REGISTRY`.

    Returns
    -------
    New process dict with every handle applied.
    """
    if registry is None:
        from hallsim.hallmarks import HALLMARK_REGISTRY

        registry = HALLMARK_REGISTRY
    result = dict(processes)
    for hname, severity in handles.items():
        handle = registry[hname]
        result = handle.apply(result, severity)
    return result


def with_handles(composite, handles: dict[str, float], *, registry=None):
    """Return a new Composite with ``handles`` severities applied.

    Applies :func:`apply_handles` to ``composite.processes`` and rewires
    them on the same topology, with topology + semantic checks off (the
    wiring is unchanged from the validated base — only parameter values
    move). The one call for "give me the treated/severity variant of this
    composite", e.g. ``Scheduler().run(with_handles(base, {...}), ...)``.
    """
    from hallsim.composite import Composite

    return Composite(
        processes=apply_handles(
            composite.processes, handles, registry=registry
        ),
        topology=composite.topology,
        validate=False,
        semantic_validation={"check_semantics": False},
    )
