"""Objective validity check for coupling edges — the driver layer.

The sibling of :mod:`hallsim.reporter_wiring`. Where that guards the *readout*
layer (an observable mapped to a gene — activity≠transcript), this guards the
*coupling* layer: when a signal drives a model parameter (a coupling edge, a
``with_param_driver``, or a hallmark severity mapping), is the target
semantically the right place to inject that influence?

The load-bearing check is **structural and needs no external ontology**: it reads
the model's own assignment rules. If you drive a *constant* parameter that the
model already modulates through a rule involving a *variable* — e.g. Zhang 2007's
``kd2_0 = kd2·(1+DNAdamage)``, where ``kd2`` is the baseline rate constant and
``DNAdamage`` is the model's damage channel — it flags the driver **for review**:
the influence may belong on ``DNAdamage`` (and its upstream dose ``Dam0``)
instead. It is deliberately *advisory* — purely structural analysis cannot
distinguish a rate constant modulated by a signal (a likely bypass) from a
parameter that legitimately configures a computed schedule (e.g. an XPP forcing
period), since both look identical in the rule graph. Confirming which is a
genuine bypass needs the semantic (SBO/ontology) layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# SBO terms that denote a kinetic rate constant (a secondary, annotation-based
# signal; the structural bypass check below is primary). Common rate-constant
# leaves under SBO:0000009 "kinetic constant".
_RATE_CONSTANT_SBO = frozenset(
    {9, 16, 17, 25, 35, 36, 37, 153, 156, 261, 320, 321, 337}
)


@dataclass(frozen=True)
class CouplingVerdict:
    process: str
    param: str
    status: str  # ok | review | rate-constant | non-structural
    message: str
    suggested_target: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_warning(self) -> bool:
        return self.status in ("review", "rate-constant")


@dataclass(frozen=True)
class CouplingReport:
    verdicts: tuple[CouplingVerdict, ...]

    @property
    def warnings(self) -> tuple[CouplingVerdict, ...]:
        return tuple(v for v in self.verdicts if v.is_warning)


def classify_driver_target(proc, param_name: str) -> CouplingVerdict:
    """Verdict for driving ``param_name`` on ``proc`` with an external signal."""
    name = getattr(proc, "_name", type(proc).__name__)
    meta = (
        proc.coupling_structure()
        if hasattr(proc, "coupling_structure")
        else None
    )
    if not meta or param_name not in meta["param_constant"]:
        # Non-SBML process, or a plain Process attribute (not an SBML
        # parameter) — no rule graph to reason over. Left to unit/topology
        # validation; not a semantic pass/fail here.
        return CouplingVerdict(
            name,
            param_name,
            "non-structural",
            f"{name}.{param_name}: not an SBML parameter with a rule graph; "
            "structural coupling check skipped.",
        )

    is_constant = meta["param_constant"][param_name]

    # Primary check: a constant modulated by a variable in a rule → the rule is
    # the model's intended channel; driving the constant bypasses it.
    if is_constant:
        for target, deps in meta["rules"]:
            if param_name not in deps:
                continue
            modulators = sorted(
                d for d in deps if d != param_name and d in meta["variables"]
            )
            if modulators:
                mods = ", ".join(modulators)
                return CouplingVerdict(
                    name,
                    param_name,
                    "review",
                    f"{name}.{param_name}: the model also modulates "
                    f"'{target}' through {mods} in rule "
                    f"'{target} = f({param_name}, {mods})'. Confirm "
                    f"{param_name} is the intended entry point for this "
                    f"influence, rather than {modulators[0]} (or its "
                    f"upstream input).",
                    suggested_target=modulators[0],
                )

    # Secondary (annotation) check: an explicitly kinetic rate constant.
    sbo = meta["param_sbo"].get(param_name, -1)
    if is_constant and sbo in _RATE_CONSTANT_SBO:
        return CouplingVerdict(
            name,
            param_name,
            "rate-constant",
            f"{name}.{param_name}: SBO:{sbo:07d} marks this a kinetic rate "
            "constant. Making a rate constant a function of a biological "
            "signal should be an explicit rule in the model, not injected by "
            "an external driver.",
        )

    return CouplingVerdict(
        name,
        param_name,
        "ok",
        f"{name}.{param_name}: a valid driving target.",
    )


def classify_topology_edge(
    processes: dict, writer_name: str, port: str, path: str
) -> CouplingVerdict:
    """Verdict for a topology wire ``writer.port -> path`` that *writes* a
    derivative contribution (EVOLVED/EXCLUSIVE) to another model's namespace.

    Structural test: if ``path`` is a quantity the owning model **computes
    algebraically** (an assignment-rule target) or **holds fixed** (a boundary
    species), an added derivative contribution is either overwritten by the
    rule or fails to integrate — wire to the state it feeds instead. Writing to
    a genuine integrated state is fine.
    """
    if "/" not in path:
        return CouplingVerdict(
            writer_name,
            path,
            "non-structural",
            f"{writer_name}.{port} → {path}: not a namespaced path; skipped.",
        )
    owner_name, var = path.split("/", 1)
    owner = processes.get(owner_name)
    struct = (
        owner.coupling_structure()
        if owner is not None and hasattr(owner, "coupling_structure")
        else None
    )
    if not struct:
        # New path (e.g. an observer output) or an opaque owner — nothing to
        # reason over structurally.
        return CouplingVerdict(
            writer_name,
            path,
            "non-structural",
            f"{writer_name}.{port} → {path}: owner has no rule graph; skipped.",
        )
    rule_targets = {t for t, _ in struct["rules"]}
    if var in rule_targets:
        return CouplingVerdict(
            writer_name,
            path,
            "review",
            f"{writer_name}.{port} → {path}: {owner_name} computes '{var}' "
            f"via an assignment rule, so an added derivative contribution is "
            f"overwritten by the rule (or double-counts). Wire to the state "
            f"'{var}' feeds, or drive a parameter instead.",
            suggested_target=owner_name,
        )
    if var in struct.get("boundary", frozenset()):
        return CouplingVerdict(
            writer_name,
            path,
            "review",
            f"{writer_name}.{port} → {path}: {owner_name} holds '{var}' as a "
            f"boundary/constant species, so a derivative contribution to it "
            f"will not integrate. Wire to a dynamic state instead.",
            suggested_target=owner_name,
        )
    return CouplingVerdict(
        writer_name,
        path,
        "ok",
        f"{writer_name}.{port} → {path}: writes to a dynamic state.",
    )


def _driven_targets(composite) -> list[tuple[object, str]]:
    """(process, param_name) for every ``with_param_driver`` coupling edge."""
    out = []
    for proc in composite.processes.values():
        for d in getattr(proc, "_param_drivers", ()):
            out.append((proc, d.param_name))
    return out


def topology_writer_verdicts(processes: dict, topology: dict) -> list:
    """Verdicts for every topology wire that *writes* (EVOLVED/EXCLUSIVE) a
    derivative contribution into another model's namespace."""
    from hallsim.process import PortRole

    out = []
    for writer_name, ports in topology.items():
        writer = processes.get(writer_name)
        if writer is None:
            continue
        schema = writer.ports_schema()
        for port, path in ports.items():
            # Only cross-namespace writes are coupling edges; a model writing
            # its own states (path owned by the writer) is its own dynamics.
            if path.split("/", 1)[0] == writer_name:
                continue
            p = schema.get(port)
            if p is not None and p.role in (
                PortRole.EVOLVED,
                PortRole.EXCLUSIVE,
            ):
                out.append(
                    classify_topology_edge(processes, writer_name, port, path)
                )
    return out


def validate_couplings(composite, extra_targets=None) -> CouplingReport:
    """Check every coupling edge in ``composite`` — both parameter drivers
    (``with_param_driver``) and topology wires that write into another model.

    ``extra_targets`` — optional ``[(process, param_name), …]`` for driver
    targets not expressed as ``_param_drivers`` (e.g. hallmark severity
    mappings), so the same structural check covers them.
    """
    targets = _driven_targets(composite)
    if extra_targets:
        targets = targets + list(extra_targets)
    verdicts = [classify_driver_target(proc, p) for proc, p in targets]
    verdicts += topology_writer_verdicts(
        composite.processes, composite.topology
    )
    return CouplingReport(verdicts=tuple(verdicts))
