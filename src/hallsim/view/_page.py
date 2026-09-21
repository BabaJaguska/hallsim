"""What a page shows: the composite, its levers, panels and shaded periods.

:func:`page_for` derives all of it from a composite and a handle registry;
a caller narrows or renames any part by passing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hallsim.process import PortRole
from hallsim.view._theme import PALETTE

_WRITES = (
    PortRole.EVOLVED,
    PortRole.EXCLUSIVE,
    PortRole.ASSIGNED,
    PortRole.LATCHED,
)
_NO_UNIT = {None, "", "dimensionless", "item", "1"}


@dataclass(frozen=True)
class Lever:
    """One slider: a registry handle and the severity range it moves on."""

    handle: str
    lo: float = 0.0
    hi: float = 1.0
    label: str = ""

    @property
    def name(self) -> str:
        return self.label or self.handle


@dataclass(frozen=True)
class Panel:
    """One trajectory panel: the store paths it sums, and its title."""

    paths: tuple[str, ...]
    label: str


def panel(spec, label: str | None = None) -> Panel:
    paths = (spec,) if isinstance(spec, str) else tuple(spec)
    return Panel(paths, label or paths[0].split("/", 1)[-1])


@dataclass(frozen=True)
class Shade:
    """A period shaded in every panel while ``lever`` is on: from ``start``
    to ``end`` (``None`` runs to the end). ``when`` is ``"nonzero"``,
    ``"positive"`` or ``"negative"``; ``negative_label`` names the period
    when the lever is below zero."""

    label: str
    start: float
    end: float | None = None
    lever: str | None = None
    when: str = "nonzero"
    negative_label: str = ""
    color: str = "color-pulse"
    opacity: float = 0.14

    def on(self, severities: dict[str, float]) -> bool:
        if self.lever is None:
            return True
        s = severities.get(self.lever, 0.0)
        return {
            "nonzero": s != 0,
            "positive": s > 0,
            "negative": s < 0,
        }[self.when]

    def name(self, severities: dict[str, float]) -> str:
        if self.negative_label and severities.get(self.lever, 0.0) < 0:
            return self.negative_label
        return self.label


@dataclass
class Page:
    """Everything the page is built from. ``variants`` are the composites a
    segmented switch chooses between, the first the default; a page with one
    shows no switch. Presets are named severity vectors in lever order and
    ``reference`` is the one every panel draws dotted underneath."""

    variants: dict
    registry: dict
    levers: tuple[Lever, ...]
    t_end: float
    macro_dt: float
    presets: dict[str, tuple[float, ...]]
    reference: str
    panels: dict[str, tuple[Panel, ...]]
    titles: dict[str, str]
    save_dt: float | None = None
    preroll: float = 0.0
    reference_label: str = "reference"
    deposits: dict[str, str] = field(default_factory=dict)
    units: dict[str, str | None] = field(default_factory=dict)
    colors: dict[str, str] = field(default_factory=dict)
    population: tuple[str, ...] = ()
    cells: int = 0
    seed: int = 0
    shades: tuple[Shade, ...] = ()
    ticks: tuple[float, ...] | None = None
    variant_title: str = "variant"
    variant_labels: dict[str, tuple[str, str]] = field(default_factory=dict)
    title: str = "hallsim"
    tagline: str = ""
    about: tuple[str, ...] = ()
    logos: tuple[tuple[str, str], ...] = ()
    assets_folder: str | None = None

    @property
    def first(self) -> str:
        return next(iter(self.variants))

    @property
    def composite(self):
        return self.variants[self.first]

    @property
    def severity_names(self) -> tuple[str, ...]:
        return tuple(lv.handle for lv in self.levers)

    def severities(self, values) -> dict[str, float]:
        return {lv.handle: v for lv, v in zip(self.levers, values)}

    def variant_label(self, name: str) -> tuple[str, str]:
        return self.variant_labels.get(name, (name, ""))


def written_paths(composite, name: str) -> list[str]:
    """The store paths ``name`` writes, in port order."""
    schema = composite.processes[name].ports_schema()
    topo = composite.topology.get(name, {})
    out = []
    for port, spec in schema.items():
        if spec.role in _WRITES and port in topo:
            target = topo[port]
            out += [target] if isinstance(target, str) else list(target)
    return out


def _units_of(composite, name: str, panels: tuple[Panel, ...]):
    schema = composite.processes[name].ports_schema()
    topo = composite.topology.get(name, {})
    by_path = {}
    for port, target in topo.items():
        for p in (target,) if isinstance(target, str) else target:
            by_path[p] = getattr(schema.get(port), "units", None)
    seen = {by_path.get(p) for pn in panels for p in pn.paths}
    seen -= _NO_UNIT
    return seen.pop() if len(seen) == 1 else None


def _deposit(proc) -> str:
    source = getattr(proc, "source", None)
    if not source:
        return ""
    source = str(source)
    return Path(source).stem if "/" in source or "." in source else source


def page_for(
    composite,
    registry: dict | None = None,
    *,
    t_end: float,
    macro_dt: float = 1.0,
    save_dt: float | None = None,
    levers: tuple[Lever, ...] | None = None,
    presets: dict[str, tuple[float, ...]] | None = None,
    reference: str | None = None,
    panels: dict[str, tuple[Panel, ...]] | None = None,
    population: tuple[str, ...] = (),
    cells: int = 0,
    seed: int = 0,
    variants: dict | None = None,
    **kw,
) -> Page:
    """A page for ``composite``: one lever per registry handle that reaches
    it, every written path a panel grouped by process, a control preset at
    zero severity as the reference. Keyword arguments override any
    :class:`Page` field."""
    registry = dict(registry or {})
    variants = dict(variants or {"composite": composite})
    procs = composite.processes
    if levers is None:
        levers = tuple(
            Lever(name)
            for name, h in registry.items()
            if any(m.process_name in procs for m in h.mappings)
        )
    if presets is None:
        presets = {"control": (0.0,) * len(levers)}
    reference = reference or next(iter(presets))
    if panels is None:
        panels = {
            name: tuple(panel(p) for p in written_paths(composite, name))
            for name in procs
        }
        panels = {k: v for k, v in panels.items() if v}
    titles = {name: name for name in panels}
    deposits = {name: _deposit(procs[name]) for name in panels}
    units = {name: _units_of(composite, name, panels[name]) for name in panels}
    colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(panels)}
    fields = dict(
        variants=variants,
        registry=registry,
        levers=levers,
        t_end=float(t_end),
        macro_dt=float(macro_dt),
        save_dt=save_dt,
        presets=presets,
        reference=reference,
        panels=panels,
        titles=titles,
        deposits=deposits,
        units=units,
        colors=colors,
        population=tuple(population),
        cells=int(cells),
        seed=int(seed),
    )
    fields.update(kw)
    return Page(**fields)
