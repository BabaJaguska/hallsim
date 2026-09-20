"""Perturbation handles: a named severity that moves parameters across
processes.

A :class:`Handle` carries a list of :class:`ParameterMapping`, each naming a
process, a parameter and how that parameter moves with severity. Applying a
handle builds *new* processes (they are immutable). A rate is **scaled
relative to its current value**, so a calibrated value is moved, never
overwritten; an input level (``calibratable(..., level=True)``) rests at 0
and is **set** by severity. Severity 0 is neutral for both, and a composite
with no handle applied is at neutral. Severity is JAX-traceable and
``jax.grad`` flows through it, for sensitivity analysis and sweeps.

**Severity is an experimental-design knob, not a fittable parameter.** Set it
per condition and fit mechanism parameters with Calibrator; the calibration
layer refuses to fit a handle's own targets unless told otherwise.

A registry is a plain ``{name: Handle}`` dict and every call names the one
it uses; the demos ship the hallmarks of aging as one
(``demos.models.hallmarks.HALLMARK_REGISTRY``), and a gene dosage, a drug or
any other perturbation is another :class:`Handle` in a registry of its own.

>>> treated = with_handles(composite, {"Mitochondrial Dysfunction": 0.7},
...                        registry=HALLMARK_REGISTRY)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from hallsim.process import Process
from hallsim.tracing import is_traced


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


def is_level(proc, param_name: str) -> bool:
    """Whether ``param_name`` on ``proc`` is declared an input level
    (``calibratable(..., level=True)``) rather than a rate. A dotted
    parameters entry is always a rate."""
    if "." in param_name:
        return False
    for f in dataclasses.fields(proc):
        if f.name == param_name:
            return bool(f.metadata.get("level", False))
    return False


@dataclass
class ParameterMapping:
    """Maps a handle's severity to one parameter as ``floor + slope *
    severity``, read against the parameter's kind:

    - a **rate** (the default) has a published or calibrated value, so the
      move is relative: ``base * (floor + slope * severity)``. ``floor=1``
      leaves the base untouched at severity 0.
    - a **level** (``calibratable(..., level=True)``, as the forcing sources
      declare their amplitudes) has no reference value and rests at 0, so
      severity sets it: ``floor + slope * severity``; ``floor=0, slope=1``
      is the exposure fraction.

    Neutral is severity 0 in both. ``slope`` is the signed gain per unit
    severity; either coefficient may be a :class:`FittableCoeff`.
    ``process_name`` keys into the composite; ``param_name`` is an attribute
    (``"alpha"``) or dotted path (``"parameters.<key>"``). ``base`` is read
    fresh on each application, so an earlier calibration flows through.
    """

    process_name: str
    param_name: str
    floor: "float | FittableCoeff | None" = None
    slope: "float | FittableCoeff | None" = None
    description: str = ""

    @property
    def floor_value(self):
        f = self.floor
        return f.init if isinstance(f, FittableCoeff) else f

    @property
    def slope_value(self):
        s = self.slope
        return s.init if isinstance(s, FittableCoeff) else s

    def value(self, severity, base, *, level: bool = False):
        """The parameter at ``severity`` given its current ``base``."""
        floor, slope = self.floor_value, self.slope_value
        if floor is None or slope is None:
            raise ValueError(
                f"ParameterMapping {self.process_name}.{self.param_name} "
                "needs both `floor` and `slope` (neutral is fixed at "
                "severity=0)."
            )
        moved = floor + slope * severity
        if level:
            return moved
        if not is_traced(base) and bool(jnp.all(jnp.asarray(base) == 0.0)):
            raise ValueError(
                f"{self.process_name}.{self.param_name} is 0, and a handle "
                "moves a rate relative to its value, so this mapping can "
                "never move it. If it is an input level, declare it "
                "calibratable(..., level=True) so severity sets it directly; "
                "otherwise map a parameter that has a value."
            )
        return base * moved


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
                new_val = mapping.value(
                    severity, base, level=is_level(proc, mapping.param_name)
                )
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
        the mapping's shape but not the absolute value."""
        out: dict[str, Any] = {}
        for m in self.mappings:
            base: Any = 1.0
            level = False
            if processes is not None and m.process_name in processes:
                proc = processes[m.process_name]
                if "." in m.param_name:
                    field_name, key = m.param_name.split(".", 1)
                    base = getattr(proc, field_name)[key]
                else:
                    base = getattr(proc, m.param_name)
                    level = is_level(proc, m.param_name)
            out[f"{m.process_name}.{m.param_name}"] = m.value(
                severity, base, level=level
            )
        return out


def apply_handles(
    processes: dict[str, Process],
    handles: dict[str, float],
    registry: dict[str, Handle],
) -> dict[str, Process]:
    """Apply several handles' severities to a process dict.

    Parameters
    ----------
    processes:
        ``{name: Process}`` from a Composite.
    handles:
        ``{handle_name: severity}`` — which handles to apply.
    registry:
        ``{name: Handle}`` to look the names up in.

    Returns
    -------
    New process dict with every handle applied.
    """
    if registry is None:
        raise ValueError(
            "apply_handles needs a registry ({name: Handle}); the demos' "
            "hallmarks of aging are demos.models.hallmarks.HALLMARK_REGISTRY"
        )
    result = dict(processes)
    for hname, severity in handles.items():
        handle = registry[hname]
        result = handle.apply(result, severity)
    return result


def with_handles(composite, handles: dict[str, float], *, registry):
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


# ── Where a handle could land: suggestions from annotations ─────────────

ROLES = ("production", "consumption", "modifier")


@dataclass(frozen=True)
class Target:
    """One parameter a handle could move: a rate of a reaction in which an
    annotated species plays ``role``. ``param`` is the dotted field a
    :class:`ParameterMapping` names; ``driven_by`` is the port that already
    drives it, when the composite wires the rate from outside; ``boundary``
    marks a constant input of the model rather than a rate constant."""

    process: str
    param: str
    species: str
    reaction: str
    role: str
    matched: tuple
    driven_by: str | None = None
    boundary: bool = False


def _matches(port_ontology: dict, ontology: dict) -> tuple | None:
    for key, value in ontology.items():
        if port_ontology.get(key) == value:
            return (key, value)
    return None


def targets_for(composite, ontology: dict, role: str = "any") -> list[Target]:
    """Every parameter of ``composite`` that a handle aimed at ``ontology``
    could move, read off the members' declared reaction channels.

    ``ontology`` is ``{"uniprot": "P04637"}``, ``{"go": ...}`` or any key a
    port carries; a port matches when one entry agrees. ``role`` is what the
    species does in the reaction whose rates are returned: ``production``,
    ``consumption``, ``modifier`` (read by the law, unchanged by it), or
    ``any``. A member with no symbolic form contributes nothing.
    """
    from hallsim.structure import parameter_field

    if role != "any" and role not in ROLES:
        raise ValueError(f"role must be one of {ROLES} or 'any'; got {role!r}")
    out: list[Target] = []
    for name, proc in composite.processes.items():
        channels = proc.reaction_channels()
        if not channels:
            continue
        schema = proc.ports_schema()
        matched = {
            port: m
            for port, spec in schema.items()
            if (m := _matches(spec.ontology or {}, ontology))
        }
        if not matched:
            continue
        values = proc.symbol_values()
        compartments = set(getattr(proc, "_compartment_names", ()))
        boundary = set(getattr(proc, "_w_names", ()))
        drivers = {
            d.input_port: d.param_name
            for d in getattr(proc, "_param_drivers", ())
        }
        params = set(values) - set(schema) - compartments
        for ch in channels:
            stoich = dict(ch.stoichiometry)
            free = {str(s) for s in ch.rate_law.free_symbols}
            for port, m in matched.items():
                coefficient = stoich.get(port, 0.0)
                if coefficient > 0:
                    r = "production"
                elif coefficient < 0:
                    r = "consumption"
                elif port in free:
                    r = "modifier"
                else:
                    continue
                if role != "any" and r != role:
                    continue
                for sym in sorted(free):
                    if sym in params:
                        out.append(
                            Target(
                                name,
                                parameter_field(proc, sym),
                                port,
                                ch.reaction_id,
                                r,
                                m,
                                boundary=sym in boundary,
                            )
                        )
                    elif sym in drivers:
                        out.append(
                            Target(
                                name,
                                parameter_field(proc, drivers[sym]),
                                port,
                                ch.reaction_id,
                                r,
                                m,
                                driven_by=sym,
                            )
                        )
    return out


def suggest_mappings(
    composite,
    ontology: dict,
    role: str = "any",
    *,
    floor: float = 1.0,
    slope: float = 1.0,
) -> list[ParameterMapping]:
    """:func:`targets_for` as ready :class:`ParameterMapping` entries, one
    per parameter, describing the reactions and the annotation it was found
    by. Paste into a registry after reading them: a suggestion is a place
    the biology could act, not a claim that it does."""
    grouped: dict[tuple[str, str], list[Target]] = {}
    for t in targets_for(composite, ontology, role):
        grouped.setdefault((t.process, t.param), []).append(t)
    out = []
    for (process, param), hits in grouped.items():
        first = hits[0]
        reactions = ", ".join(sorted({h.reaction for h in hits}))
        roles = "/".join(sorted({h.role for h in hits}))
        note = f"{roles} of {first.species} ({first.matched[0]}:{first.matched[1]}) in {reactions}"
        if first.driven_by:
            note += f"; driven through port {first.driven_by}"
        if first.boundary:
            note += "; a boundary input, not a rate constant"
        out.append(
            ParameterMapping(
                process_name=process,
                param_name=param,
                floor=floor,
                slope=slope,
                description=note,
            )
        )
    return out


@dataclass
class IntentTarget:
    """What a perturbation does, in biology: the species by ontology, its
    role in the reactions to scale, and the gain per unit severity."""

    ontology: dict
    role: str = "any"
    floor: float = 1.0
    slope: float = 1.0
    note: str = ""


@dataclass
class Intent:
    """A handle described without naming any model, so it can be suggested
    onto any composite: :func:`suggest_handle` turns it into a
    :class:`Handle` with named mappings for one composite."""

    name: str
    description: str = ""
    category: str = ""
    references: list[str] = field(default_factory=list)
    targets: list[IntentTarget] = field(default_factory=list)


def suggest_handle(intent: Intent, composite) -> Handle:
    """A :class:`Handle` for ``composite`` with every mapping the intent's
    targets suggest; empty ``mappings`` when nothing on the composite is
    annotated as the intent asks."""
    seen: set[tuple[str, str]] = set()
    mappings: list[ParameterMapping] = []
    for target in intent.targets:
        for m in suggest_mappings(
            composite,
            target.ontology,
            target.role,
            floor=target.floor,
            slope=target.slope,
        ):
            key = (m.process_name, m.param_name)
            if key in seen:
                continue
            seen.add(key)
            if target.note:
                m.description = f"{target.note}: {m.description}"
            mappings.append(m)
    return Handle(
        name=intent.name,
        description=intent.description,
        mappings=mappings,
        category=intent.category,
        references=list(intent.references),
    )


def suggest_registry(intents, composite) -> dict[str, Handle]:
    """``{name: Handle}`` for ``composite`` from a dict or iterable of
    :class:`Intent`; a handle that reaches nothing keeps empty mappings so
    the gap is visible rather than dropped."""
    items = intents.values() if isinstance(intents, dict) else intents
    return {i.name: suggest_handle(i, composite) for i in items}
