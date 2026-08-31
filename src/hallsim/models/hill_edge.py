"""HillActivationEdge — generic Hill-gated additive coupling edge.

One primitive for every cross-model activation edge:

    d(target)/dt += k_act · ∏ᵢ hill_gate(sourceᵢ; Kᵢ, nᵢ)

Ports are generic (``target`` + one ``source`` per driver); the store
paths they connect live in the composite topology, so an agent adds a
coupling by *instantiating* this edge with ``(k_act, K, n)`` and its
metadata rather than authoring a new Process class. Single-source is the
common case; pass ``sources=(...)`` with matching ``K``/``n`` tuples for an
AND of drivers (the gates multiply).

``target`` is an EVOLVED pure source (``reads_value=False``): the term
depends on the sources, not on the path it writes, so it composes
additively with the target module's intrinsic dynamics without creating a
spurious feedback cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp

from hallsim.kinetics import hill_gate
from hallsim.process import Port, PortRole, Process, calibratable
from hallsim.tracing import is_traced


class HillActivationEdge(Process):
    """Hill-gated additive edge; see module docstring for the rate law."""

    timescale: float | None = None

    k_act: float = calibratable(
        1.0, description="Hill-edge strength; fit against the target reporter."
    )
    K: tuple = (1.0,)  # per-source half-saturation threshold
    n: tuple = (2.0,)  # per-source Hill cooperativity

    sources: tuple = eqx.field(static=True, default=("source",))
    # None abstains; set it only when this edge owns the target path.
    target_default: float | None = eqx.field(static=True, default=None)
    target_ontology: dict | None = eqx.field(static=True, default=None)
    target_description: str = eqx.field(static=True, default="")
    source_ontology: tuple | None = eqx.field(static=True, default=None)
    source_descriptions: tuple | None = eqx.field(static=True, default=None)
    hallmark: str | None = eqx.field(static=True, default=None)
    reference: str | None = eqx.field(static=True, default=None)
    description: str | None = eqx.field(static=True, default=None)

    def ports_schema(self):
        ont = self.source_ontology or ((None,) * len(self.sources))
        descs = self.source_descriptions or (("",) * len(self.sources))
        ports = {
            "target": Port(
                role=PortRole.EVOLVED,
                default=self.target_default,
                units="dimensionless",
                description=self.target_description,
                ontology=self.target_ontology or {},
                reads_value=False,
            )
        }
        for name, o, d in zip(self.sources, ont, descs):
            ports[name] = Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description=d,
                ontology=o or {},
            )
        return ports

    def derivative(self, t, state):
        drive = jnp.asarray(1.0)
        for name, K, n in zip(self.sources, self.K, self.n):
            drive = drive * hill_gate(state[name], K, n)
        return {"target": self.k_act * drive}


class HillSignalEdge(Process):
    """Assigns a signal store path from a source via a Hill — the algebraic
    (ASSIGNED) sibling of :class:`HillActivationEdge`. Each step:

        signal = basal + (hi − basal) · hill_gate(source; K, n)

    computed as a cross-process assignment rule (no integration, no timescale
    lag). An imported model reads the ``signal`` path through a plain
    parameter INPUT (``ImportedODEProcess.with_param_input``), so the Hill
    transform is a first-class composable edge rather than baked into a
    driver. ``basal`` and ``K`` are the fittable pair; ``hi``/``n`` are traced
    too but stay off the calibration surface.

    The edge interpolates from ``basal`` toward ``hi`` in either direction, so
    ``hi < basal`` is an inhibitory edge — a source that suppresses its target,
    which is as ordinary in biology as activation.
    """

    timescale: float | None = None
    basal: float = calibratable(
        0.3, description="signal floor at source→0; fit against the reporter."
    )
    hi: float = 1.0
    K: float = calibratable(
        1.0,
        description="source threshold; place at the operating point "
        "(hallsim.calibration.suggest_hill_gate) or fit against the readout.",
    )
    n: float = 2.0

    source_ontology: dict | None = eqx.field(static=True, default=None)
    source_description: str = eqx.field(static=True, default="")
    hallmark: str | None = eqx.field(static=True, default=None)
    reference: str | None = eqx.field(static=True, default=None)
    description: str | None = eqx.field(static=True, default=None)

    def __check_init__(self):
        super().__check_init__()
        if not is_traced(self.basal, self.hi) and float(self.basal) == float(
            self.hi
        ):
            raise ValueError(
                f"{type(self).__name__}: basal == hi == {float(self.hi):g}, so "
                "the edge assigns a constant and carries no signal. Separate "
                "them; hi below basal is a valid inhibitory edge."
            )

    def ports_schema(self):
        return {
            "signal": Port(
                role=PortRole.ASSIGNED,
                default=self.basal,
                units="dimensionless",
                description="Hill-bridged algebraic signal.",
            ),
            "source": Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description=self.source_description,
                ontology=self.source_ontology or {},
            ),
        }

    def assign(self, t, state):
        gate = hill_gate(state["source"], self.K, self.n)
        return {"signal": self.basal + (self.hi - self.basal) * gate}


@dataclass
class HillGateSuggestion:
    """Deterministically-placed Hill ``(K, n)`` for a gate that should be
    ~closed at ``off_level`` and ~open at ``on_level`` (see
    :func:`place_hill_gate`). ``ok`` is False when the levels overlap or are too
    close for a clean gate — read ``note`` for why."""

    K: float
    n: float
    off_level: float
    on_level: float
    off_occupancy: float
    on_occupancy: float
    ok: bool
    note: str
    # Set by place_hill_gate_for_crossing: the driver level at which the signal
    # reaches `critical`, the (lo, hi) K window straddling the conditions, and
    # the fractional distance from the crossing to the nearer one.
    crossing: float | None = None
    window: tuple[float, float] | None = None
    margin: float | None = None


def _occ(x, K, n):
    return x**n / (K**n + x**n)


def place_hill_gate_for_crossing(
    off_level,
    on_level,
    *,
    basal,
    hi,
    critical,
    n: float = 2.0,
) -> HillGateSuggestion:
    """Hill ``K`` placing a gate so its *signal* crosses ``critical`` between
    the driver's ``off_level`` and ``on_level``.

    For an edge assigning ``signal = basal + (hi - basal)·H(source; K, n)``,
    a downstream bifurcation at ``signal = critical`` is reached where
    ``H = (critical - basal)/(hi - basal)``, i.e. at the driver level
    ``D* = K·(H/(1-H))**(1/n)``. Straddling the conditions therefore needs
    ``off/r < K < on/r`` with ``r = (H/(1-H))**(1/n)``; ``K`` is placed at the
    geometric centre of that window. Works for ``hi`` above or below ``basal``.

    Where :func:`place_hill_gate` asks for a clean 10/90 switch — and so needs
    the levels well separated — this asks only that the crossing fall between
    them, which needs ``off < on`` and nothing more. ``note`` carries the
    window and the fractional margin to the nearer condition, because a
    crossing placed in a narrow window is correct but fragile.
    """
    off, on, basal, hi = (
        float(off_level),
        float(on_level),
        float(basal),
        float(hi),
    )
    crit = float(critical)
    span = hi - basal
    h = (crit - basal) / span if span else float("nan")
    if not 0.0 < h < 1.0:
        return HillGateSuggestion(
            max(on, 1e-9),
            n,
            off,
            on,
            float("nan"),
            float("nan"),
            False,
            f"critical {crit:.4g} is outside the edge's range "
            f"[{min(basal, hi):.4g}, {max(basal, hi):.4g}], so the signal "
            "never reaches it at any K",
        )
    if off <= 0.0 or on <= 0.0:
        return HillGateSuggestion(
            max(on, 1e-9),
            n,
            off,
            on,
            float("nan"),
            float("nan"),
            False,
            "non-positive operating level; cannot place a Hill gate",
        )
    r = (h / (1.0 - h)) ** (1.0 / n)
    lo_K, hi_K = off / r, on / r
    K = math.sqrt(lo_K * hi_K)
    crossing = K * r
    if on <= off:
        return HillGateSuggestion(
            K,
            n,
            off,
            on,
            _occ(off, K, n),
            _occ(on, K, n),
            False,
            f"off ({off:.3g}) >= on ({on:.3g}): operating ranges overlap — "
            "no monotone Hill gate separates these levels",
        )
    margin = min(crossing / off, on / crossing) - 1.0
    return HillGateSuggestion(
        K,
        n,
        off,
        on,
        _occ(off, K, n),
        _occ(on, K, n),
        True,
        f"crosses {crit:.4g} at driver {crossing:.4g}; K window "
        f"{lo_K:.4g}–{hi_K:.4g}, margin {100 * margin:.3g}% to the nearer "
        "condition",
        crossing=crossing,
        window=(lo_K, hi_K),
        margin=margin,
    )


def place_hill_gate(
    off_level, on_level, *, off_occupancy: float = 0.1, n_max: float = 8.0
) -> HillGateSuggestion:
    """Deterministic Hill ``(K, n)`` from a source's off/on operating levels.

    Places ``K`` at the geometric mean ``sqrt(off*on)`` — the operating midpoint —
    and picks the smallest ``n`` making the gate at most ``off_occupancy`` open at
    ``off_level`` (so it is ``1 - off_occupancy`` open at ``on_level``, by the
    symmetry of the Hill about ``K``). Flags (``ok=False``) when ``on <= off``
    (ranges overlap — no monotone gate separates them) or when the required ``n``
    exceeds ``n_max`` (levels too close for a clean gate). Pure arithmetic on
    measured operating points — no fitting, no heuristics beyond the stated rule;
    pair with :meth:`hallsim.calibration.CalibrationProblem.suggest_hill_gate`
    which supplies the levels from :meth:`operating_ranges`."""
    off = float(off_level)
    on = float(on_level)
    if off <= 0.0 or on <= 0.0:
        return HillGateSuggestion(
            max(on, 1e-9),
            2.0,
            off,
            on,
            float("nan"),
            float("nan"),
            False,
            "non-positive operating level; cannot place a Hill gate",
        )
    K = math.sqrt(off * on)
    if on <= off:
        return HillGateSuggestion(
            K,
            2.0,
            off,
            on,
            _occ(off, K, 2.0),
            _occ(on, K, 2.0),
            False,
            f"off ({off:.3g}) >= on ({on:.3g}): operating ranges overlap — "
            "no monotone Hill gate separates these levels",
        )
    r = on / off
    need = math.log((1.0 - off_occupancy) / off_occupancy) / (
        0.5 * math.log(r)
    )
    n = max(2.0, math.ceil(need))
    ok = n <= n_max
    note = (
        "ok"
        if ok
        else f"needs n={n:.0f} (>{n_max:.0f}); the driver's low and high "
        f"levels differ by only r={r:.2f}, too little for a Hill gate to "
        "resolve at any plausible cooperativity"
    )
    return HillGateSuggestion(
        K, float(n), off, on, _occ(off, K, n), _occ(on, K, n), ok, note
    )
