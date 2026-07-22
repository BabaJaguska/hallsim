"""Hallmark handles — high-level control interface for aging biology.

A hallmark of aging (Lopez-Otin et al., 2023) is represented as a 0-1
severity handle that modulates parameters across one or more Processes.
The abstraction targets all 12 hallmarks; :data:`HALLMARK_REGISTRY`
maps 4 today — Stem Cell Exhaustion, Mitochondrial Dysfunction,
Deregulated Nutrient Sensing, and Genomic Instability — and each new
one is a single :class:`HallmarkHandle` entry away.

    severity = 0.0  ->  healthy / optimal
    severity = 1.0  ->  severely impaired

Hallmark transforms are **multiplicative of the current calibrated
base value**, not absolute. A transform receives ``(severity, base)``
and returns ``base * f(severity)``. This lets a calibration loss
substitute mechanism parameters via ``parameters`` and then
apply hallmarks at the experimental severity profile without the
hallmark clobbering the calibrated values.

Because Process instances are immutable Equinox modules, hallmark
handles work by constructing *new* Process instances with modified
parameters. Severity is JAX-traceable (pass a ``jnp.ndarray`` and
``jax.grad`` flows through), and so is the base value (so Calibrator
can fit through ``parameters`` while hallmarks scale by
severity).

**Severity is an experimental-design knob, not a fittable parameter.**
The intended pattern is to set severity for each experimental
condition (DDIS=1.0, ctrl=0.0, RAPA-rescued=0.3) and fit mechanism
parameters via ``parameters`` with Calibrator. Severity-
differentiability is preserved for sensitivity analysis and severity-
sweep population studies, not for inferring "what severity does the
data show" — that would conflate experimental setup with model state.

Usage
-----
>>> from hallsim.hallmarks import HallmarkHandle, HALLMARK_REGISTRY
>>> handle = HALLMARK_REGISTRY["Mitochondrial Dysfunction"]
>>> modified_procs = handle.apply(composite.processes, severity=0.7)
>>> new_composite = Composite(modified_procs, composite.topology)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import equinox as eqx

from hallsim.process import Process


@dataclass(frozen=True)
class FittableCoeff:
    """A hallmark-mapping coefficient the Calibrator may fit.

    Stands in for a plain float in a mapping's ``floor``. The Calibrator
    discovers it via a :class:`hallsim.calibration.HallmarkCoeffRef` and
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
    """Maps a hallmark severity to a process parameter value.

    Two forms, resolved by :meth:`value`:

    - **Affine** (``floor`` set): ``value = base * (floor + slope * severity)``.
      ``floor`` is the severity=0 multiple of ``base``, ``slope`` the gain per
      unit severity. ``slope=None`` pins the map at severity=1
      (``slope = 1 - floor``) — the form for a rate whose *untreated* level is
      the fitted ``base`` and whose lower severities are a suppressed fraction
      of it (e.g. mTOR under rapamycin). Most hallmarks scale up from the
      control value instead: ``floor=1`` with an explicit ``slope``. ``floor``
      may be a :class:`FittableCoeff` to calibrate it.
    - **Custom** (``transform`` set): ``value = transform(severity, base)``.
      For a severity *dial* that sets the value directly and ignores ``base``
      (``lambda h, base: h``).

    Attributes
    ----------
    process_name:
        Name of the target process in the composite.
    param_name:
        Name of the Process field to modify. Either a plain attribute
        (``"alpha"``) or a dotted path into a dict-valued field
        (``"parameters.<key>"``).
    floor, slope:
        Affine coefficients (see above). ``base`` is read fresh from the
        target field on each application, so a prior calibration
        substitution into it flows through cleanly.
    transform:
        Escape hatch for non-affine / base-independent dials.
    description:
        Human-readable description of what this mapping does.
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

    def value(self, severity, base):
        """Resolve the parameter value at ``severity`` given current ``base``."""
        if self.transform is not None:
            return self.transform(severity, base)
        if self.floor is None:
            raise ValueError(
                f"ParameterMapping {self.process_name}.{self.param_name} "
                "needs either an affine `floor` or a `transform`."
            )
        floor = self.floor_value
        slope = (1.0 - floor) if self.slope is None else self.slope
        return base * (floor + slope * severity)


@dataclass
class HallmarkHandle:
    """A control knob for one hallmark of aging.

    Attributes
    ----------
    name:
        Human-readable name (e.g., "Mitochondrial Dysfunction").
    description:
        Brief description of the biology.
    mappings:
        List of ParameterMapping defining how severity affects processes.
    category:
        "Primary", "Antagonistic", or "Integrative" (Lopez-Otin taxonomy).
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
        """Return new process dict with parameters modified by severity.

        Processes not targeted by any mapping are returned unchanged.
        Target processes get new instances with updated parameter values
        via ``eqx.tree_at``.

        ``ParameterMapping.param_name`` may be either a plain attribute
        (e.g. ``"alpha"``) or a dotted path into a dict-valued field
        (e.g. ``"parameters.mTORC1_S2448_phos_by_AA"``). The
        dotted form lets a hallmark target a specific key inside the
        ``parameters`` dict on an :class:`hallsim.sbml_import.SBMLProcess`,
        which is the mechanism for parameterising specific SBML rate
        constants from a hallmark severity.

        Parameters
        ----------
        processes:
            ``{name: Process}`` from a Composite.
        severity:
            Hallmark severity in [0, 1].

        Returns
        -------
        New dict with modified Process instances.
        """
        result = dict(processes)
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
        """Show what parameters would be set at a given severity.

        If ``processes`` is provided, reads the actual current base
        values from each target process and returns the transformed
        result. If omitted, uses ``base=1.0`` as a placeholder
        (suitable for displaying the transform's severity-shape but
        not the absolute resulting value).
        """
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


def apply_hallmarks(
    processes: dict[str, Process],
    hallmarks: dict[str, float],
    registry: dict[str, HallmarkHandle] | None = None,
) -> dict[str, Process]:
    """Apply multiple hallmark severities to a process dict.

    Parameters
    ----------
    processes:
        ``{name: Process}`` from a Composite.
    hallmarks:
        ``{hallmark_name: severity}`` — which hallmarks to apply.
    registry:
        Hallmark registry to look up handles. Defaults to
        ``HALLMARK_REGISTRY``.

    Returns
    -------
    New process dict with all hallmark effects applied.
    """
    if registry is None:
        registry = HALLMARK_REGISTRY
    result = dict(processes)
    for hname, severity in hallmarks.items():
        handle = registry[hname]
        result = handle.apply(result, severity)
    return result


def with_hallmarks(composite, hallmarks: dict[str, float], *, registry=None):
    """Return a new Composite with ``hallmarks`` severities applied.

    Applies :func:`apply_hallmarks` to ``composite.processes`` and rewires
    them on the same topology, with topology + semantic checks off (the
    wiring is unchanged from the validated base — only parameter values
    move). The one call for "give me the treated/severity variant of this
    composite", e.g. ``Scheduler().run(with_hallmarks(base, {...}), ...)``.
    """
    from hallsim.composite import Composite

    return Composite(
        processes=apply_hallmarks(
            composite.processes, hallmarks, registry=registry
        ),
        topology=composite.topology,
        validate=False,
        semantic_validation={"check_semantics": False},
    )


# ── Registry ────────────────────────────────────────────────────────────

# Hallmark definitions for ERiQ-based processes.
# Process names match those in build_eriq_composite().

HALLMARK_REGISTRY: dict[str, HallmarkHandle] = {
    "Stem Cell Exhaustion": HallmarkHandle(
        name="Stem Cell Exhaustion",
        description=(
            "Age-dependent decline in stem cell niche signaling. "
            "Wnt, EGF, Shh, and Notch pathways deteriorate, reducing "
            "self-renewal capacity and regenerative potential."
        ),
        category="Integrative",
        references=[
            "Lopez-Otin et al. 2023",
            "Sivakumar et al. 2011 (BIOMD0000000398)",
        ],
        mappings=[
            # Stem-cell niche severity is the direct knob (no calibrated
            # base behind it) — base is ignored.
            ParameterMapping(
                process_name="niche",
                param_name="severity",
                transform=lambda h, base: h,
                description="Niche deterioration severity — scales decay of all ligands",
            ),
        ],
    ),
    "Mitochondrial Dysfunction": HallmarkHandle(
        name="Mitochondrial Dysfunction",
        description=(
            "Impairment in mitochondrial function leading to reduced ATP "
            "production, increased ROS generation, and accumulation of "
            "mitochondrial damage."
        ),
        category="Primary",
        references=["Lopez-Otin et al. 2023", "Alfego & Kriete 2017"],
        mappings=[
            # severity=0 → base (no perturbation); severity=1 → 3*base
            # (the published "severely impaired" factor).
            ParameterMapping(
                process_name="oxidative_stress",
                param_name="MDAMAGE_SA",
                floor=1.0,
                slope=2.0,
                description="Damage accumulation rate scales 1x→3x with dysfunction",
            ),
        ],
    ),
    "Deregulated Nutrient Sensing": HallmarkHandle(
        name="Deregulated Nutrient Sensing",
        description=(
            "Imbalance in nutrient-sensing pathways (mTOR, AMPK, sirtuins). "
            "Chronic mTOR activation, impaired AMPK response, declining NAD+. "
            "Pharmacological mTORC1 inhibitors (rapamycin and analogs) map "
            "to this hallmark as a downward severity shift."
        ),
        category="Primary",
        references=[
            "Lopez-Otin et al. 2023",
            "Alfego & Kriete 2017",
            "DallePezze 2014 (BIOMD0000000582)",
        ],
        mappings=[
            # ERiQ-based composites: severity=0 → base; severity=1 → 1.5*base.
            ParameterMapping(
                process_name="energy",
                param_name="GLYCOL_SA",
                floor=1.0,
                slope=0.5,
                description="Glycolytic flux scales 1x→1.5x with nutrient dysregulation (ERiQ-based composites)",
            ),
            # DP14-based composites: scale the mTORC1 phosphorylation rate
            # around the (fitted) untreated base. severity=1 → base (untreated
            # DDIS), severity=0 → floor*base (rapamycin-suppressed residual;
            # slope=None pins severity=1 at base). The floor is the residual
            # mTOR fraction under rapamycin — a real biological quantity, fitted
            # like GZ06's basal ψ rather than frozen. It sets the DDIS:control
            # mTOR contrast the EIF4EBP1 reporter reads.
            ParameterMapping(
                process_name="dp14",
                param_name=(
                    "parameters." "mTORC1_S2448_phos_by_AA_n_Akt_pS473"
                ),
                floor=FittableCoeff(
                    init=0.3,
                    clamp=(0.05, 0.95),
                    prior=0.3,
                    prior_sigma=0.3,
                    description="mTOR residual fraction under rapamycin",
                ),
                slope=None,
                description=(
                    "mTORC1 S2448 phosphorylation rate (DP14): "
                    "floor*base at severity=0 (rapa-suppressed), "
                    "full base at severity=1 (untreated)"
                ),
            ),
        ],
    ),
    "Genomic Instability": HallmarkHandle(
        name="Genomic Instability",
        description=(
            "Exogenous DNA damage exposure. Drives ERiQ's damage_repair "
            "(eta) and DP14's Irradiation exposure input — severity is the "
            "normalized exposure level (0=none, 1=full). GZ06's psi is no "
            "longer set here: it is driven by DP14's DNA_damage state "
            "through a topology edge (see multi_hallmark). The per-exposure "
            "damage potency is a mechanism parameter fit separately, not "
            "part of this dial."
        ),
        category="Primary",
        references=[
            "Lopez-Otin et al. 2023",
            "DallePezze 2014 (BIOMD0000000582)",
            "Geva-Zatorsky 2006 (BIOMD0000000157)",
        ],
        mappings=[
            # ERiQ-based composites: severity=0 → base; severity=1 → 5*base.
            ParameterMapping(
                process_name="damage_repair",
                param_name="eta",
                floor=1.0,
                slope=4.0,
                description="Damage production rate scales 1x→5x with instability (ERiQ-based composites)",
            ),
            # DP14-based composites: severity IS the exogenous-exposure
            # level. severity maps directly onto DP14's `Irradiation`
            # boundary input (0 = no exposure, 1 = full exposure) — an
            # identity dial, not a scaled rate constant. The damage
            # *potency* per unit exposure (`DNA_damaged_by_irradiation`) is
            # a mechanism parameter Calibrator fits separately; severity
            # never touches it. The exposure is delivered as a bounded dose
            # window (`with_pulse_window`, default 2 days for etoposide);
            # persistence past washout comes from the ROS→damage feedback,
            # not a sustained input.
            ParameterMapping(
                process_name="dp14",
                param_name="parameters.Irradiation",
                transform=lambda h, base: h,
                description=(
                    "Exogenous-exposure level (DP14 Irradiation input): "
                    "0 at severity=0 (no exposure), full at severity=1 "
                    "(sustained DDIS exposure)"
                ),
            ),
            # GZ06's psi is not mapped here — it is driven by DP14's
            # DNA_damage via a topology edge (see multi_hallmark), so GI
            # severity reaches GZ06 through Irradiation → DNA_damage → psi.
        ],
    ),
}
