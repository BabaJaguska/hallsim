"""Hallmark handles — high-level control interface for aging biology.

Each of the 12 hallmarks of aging (Lopez-Otin et al., 2023) is
represented as a 0-1 severity handle that modulates parameters across
one or more Processes.

    severity = 0.0  ->  healthy / optimal
    severity = 1.0  ->  severely impaired

Because Process instances are immutable Equinox modules, hallmark
handles work by constructing *new* Process instances with modified
parameters.  This means hallmark severity is differentiable:

    jax.grad(lambda s: simulate(apply_hallmark(composite, s)).loss)(0.5)

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


@dataclass
class ParameterMapping:
    """Maps a hallmark severity to a process parameter value.

    Attributes
    ----------
    process_name:
        Name of the target process in the composite.
    param_name:
        Name of the Process field to modify.
    transform:
        Function ``severity -> param_value``.
    description:
        Human-readable description of what this mapping does.
    """

    process_name: str
    param_name: str
    transform: Callable[[float], float]
    description: str = ""


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
            new_val = mapping.transform(severity)
            # Use eqx.tree_at to create a new Process with the modified param
            result[pname] = eqx.tree_at(
                lambda p: getattr(p, mapping.param_name),
                proc,
                new_val,
            )
        return result

    def summary(self, severity: float) -> dict[str, Any]:
        """Show what parameters would be set at a given severity."""
        return {
            f"{m.process_name}.{m.param_name}": m.transform(severity)
            for m in self.mappings
        }


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
            ParameterMapping(
                process_name="niche",
                param_name="severity",
                transform=lambda h: h,
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
            ParameterMapping(
                process_name="oxidative_stress",
                param_name="MDAMAGE_SA",
                transform=lambda h: 1.0 + h * 2.0,  # 1.0 -> 3.0
                description="Damage accumulation rate increases with dysfunction",
            ),
        ],
    ),

    "Deregulated Nutrient Sensing": HallmarkHandle(
        name="Deregulated Nutrient Sensing",
        description=(
            "Imbalance in nutrient-sensing pathways (mTOR, AMPK, sirtuins). "
            "Chronic mTOR activation, impaired AMPK response, declining NAD+."
        ),
        category="Primary",
        references=["Lopez-Otin et al. 2023", "Alfego & Kriete 2017"],
        mappings=[
            ParameterMapping(
                process_name="energy",
                param_name="GLYCOL_SA",
                transform=lambda h: 1.0 + h * 0.5,  # 1.0 -> 1.5
                description="Glycolytic flux increases with nutrient dysregulation",
            ),
        ],
    ),

    "Genomic Instability": HallmarkHandle(
        name="Genomic Instability",
        description="DNA damage accumulation due to genomic instability.",
        category="Primary",
        references=["Lopez-Otin et al. 2023"],
        mappings=[
            ParameterMapping(
                process_name="damage_repair",
                param_name="eta",
                transform=lambda h: 0.5 + h * 2.0,  # 0.5 -> 2.5
                description="Damage production rate increases with instability",
            ),
        ],
    ),
}
