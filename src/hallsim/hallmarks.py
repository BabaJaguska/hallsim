"""Hallmark handles — high-level control interface for aging biology.

A hallmark of aging (Lopez-Otin et al., 2023) is a signed severity handle in
[-1, 1] modulating parameters across one or more Processes: -1 is the full
opposite perturbation (mTOR suppression), 0 homeostasis, +1 severely impaired.
A hallmark with no meaningful opposite — there is no negative DNA damage —
uses the [0, 1] half. :data:`HALLMARK_REGISTRY` maps 5 of the 12 today; each
new one is a single :class:`HallmarkHandle` entry.

Transforms are **multiplicative of the current base**: ``base * f(severity)``,
not an absolute value, so a calibration can fit mechanism parameters and then
apply severities without the hallmark clobbering the fit. Processes are
immutable, so applying a handle builds *new* instances; both severity and base
are JAX-traceable, and ``jax.grad`` flows through either.

**Severity is an experimental-design knob, not a fittable parameter.** Set it
per condition (DDIS=1.0, ctrl=0.0) and fit mechanism parameters with
Calibrator. Its differentiability is there for sensitivity analysis and
severity sweeps, not for inferring "what severity does the data show" — that
would conflate experimental setup with model state.

>>> handle = HALLMARK_REGISTRY["Mitochondrial Dysfunction"]
>>> new_composite = Composite(
...     handle.apply(composite.processes, severity=0.7), composite.topology
... )
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
    """Maps a hallmark severity to a process parameter value, two forms:

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
        """New ``{name: Process}`` with this hallmark applied at ``severity``
        (in [-1, 1], 0 = homeostasis). Untargeted processes pass through
        unchanged; targeted ones are rebuilt via ``eqx.tree_at``. A dotted
        ``param_name`` reaches inside a dict-valued field, which is how a
        hallmark drives one SBML rate constant."""
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
            # DP14-based composites: severity is the nutrient/mTOR drive
            # level on DP14's `Amino_Acids` input (`forcing.drive_step` adds
            # the "nutrient_drive" StepSource), never a rate constant — so
            # arms differ only in u(t). The phosphorylation rate per unit
            # drive stays a mechanism parameter Calibrator fits.
            # Skipped for composites without the source.
            ParameterMapping(
                process_name="nutrient_drive",
                param_name="after",
                floor=1.0,
                slope=FittableCoeff(
                    init=0.7,
                    clamp=(0.05, 0.95),
                    prior=0.7,
                    prior_sigma=0.3,
                    description="mTOR/nutrient suppression gain (severity=-1 → (1-gain)x basal drive under rapamycin)",
                ),
                description=(
                    "Nutrient/mTOR drive level (DP14 Amino_Acids input): "
                    "basal at severity=0, (1-gain)x basal at severity=-1 "
                    "(rapa-suppressed), (1+gain)x at severity=+1"
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
            # level — an identity dial (0 = no exposure, 1 = full DDIS dose).
            # It sets the amplitude of the forcing source (`forcing.drive_pulse`
            # adds a PulseSource named "irradiation_pulse" driving DP14's
            # Irradiation input over the dose window). The damage *potency* per
            # unit exposure (`DNA_damaged_by_irradiation`) is a separate
            # mechanism parameter Calibrator fits; severity never touches it.
            # Skipped for composites without the pulse source (apply() ignores
            # mappings whose process is absent).
            ParameterMapping(
                process_name="irradiation_pulse",
                param_name="amplitude",
                transform=lambda h, base: h,
                description=(
                    "Exogenous-exposure level (irradiation PulseSource "
                    "amplitude): 0 at severity=0 (no exposure), full at "
                    "severity=1 (full DDIS dose)"
                ),
            ),
            # GZ06's psi is not mapped here — it is driven by DP14's
            # DNA_damage via a topology edge (see multi_hallmark), so GI
            # severity reaches GZ06 through Irradiation → DNA_damage → psi.
        ],
    ),
    "Altered Intercellular Communication": HallmarkHandle(
        name="Altered Intercellular Communication",
        description=(
            "Signal a cell receives from other cells, rather than from its "
            "own state. Drives the CD95L death-ligand challenge on "
            "Kallenberger 2014 — severity is the normalized ligand level "
            "(0 = unchallenged, which leaves that model exactly at rest, "
            "1 = the placed dose). The ligand's potency per unit "
            "concentration lives in the deposit's own rate constants and is "
            "never touched by this dial."
        ),
        category="Integrative",
        references=[
            "Lopez-Otin et al. 2023",
            "Kallenberger 2014 (BIOMD0000000524)",
        ],
        mappings=[
            # severity=0 → 0 (no ligand), severity=1 → the placed dose.
            # Skipped for composites without the challenge source.
            ParameterMapping(
                process_name="cd95l_challenge",
                param_name="amplitude",
                floor=0.0,
                slope=1.0,
                description=(
                    "CD95L ligand level (challenge PulseSource amplitude): "
                    "0 at severity=0, the placed dose at severity=1"
                ),
            ),
        ],
    ),
}
