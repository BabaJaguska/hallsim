import json
import logging

import click
import equinox as eqx

#: -q, none, -v, -vv. The default is WARNING because the framework's warnings
#: are the ones a user must not miss; INFO carries the decisions it made
#: (auto-reduced save_dt, stiffness verdicts, group ordering).
_LEVELS = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]


@click.group()
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Show what the framework decided (-v) and how (-vv).",
)
@click.option(
    "-q", "--quiet", is_flag=True, help="Errors only; suppress warnings."
)
def simulate(verbose, quiet):
    """HallSim simulation commands."""
    level = _LEVELS[0 if quiet else min(1 + verbose, len(_LEVELS) - 1)]
    # force=True so the level a user asked for wins over any basicConfig a
    # demo module ran at import; without a handler every log.warning arrives
    # through logging.lastResort as bare stderr with no level or logger name.
    # The root stays at warnings so -v means "what HallSim decided" and not
    # JAX's backend probing; -q quietens everything.
    logging.basicConfig(
        level=logging.ERROR if quiet else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("hallsim").setLevel(level)


@simulate.group("demo")
def demo():
    """Worked examples: framework mechanics on toy processes, and the
    multi-hallmark case study. Test workloads, not models to build on."""


# ── Composable architecture commands ─────────────────────────────────────


@demo.command("compose")
@click.option("--t1", type=float, default=50.0, help="End time")
@click.option("--dt", type=float, default=1.0, help="Save interval")
@click.option(
    "--validate/--no-validate", default=True, help="Run semantic validation"
)
def compose(t1, dt, validate):
    """Demo: compose ROS production + antioxidant defense, solve ODE.

    Wires two processes to the same store path (cytoplasm/ROS) and runs
    the combined system. Shows the composable architecture in action.
    """
    import jax.numpy as jnp

    from hallsim.composite import Composite
    from hallsim.process import Port, PortRole, Process
    from hallsim.scheduler import Scheduler

    class ROSProduction(Process):
        rate: float = 0.05

        def ports_schema(self):
            return {
                "ros": Port(
                    role=PortRole.EVOLVED,
                    default=0.1,
                    units="uM",
                    description="Reactive oxygen species concentration",
                    ontology={"chebi": "CHEBI:26523"},
                ),
            }

        def derivative(self, t, state):
            return {"ros": jnp.array(self.rate)}

    class AntioxidantDefense(Process):
        scavenge_rate: float = 0.02

        def ports_schema(self):
            return {
                "ros": Port(
                    role=PortRole.EVOLVED,
                    default=0.1,
                    units="uM",
                    description="Reactive oxygen species concentration",
                    ontology={"chebi": "CHEBI:26523"},
                ),
            }

        def derivative(self, t, state):
            return {"ros": -self.scavenge_rate * state["ros"]}

    processes = {
        "ros_prod": ROSProduction(),
        "antioxidant": AntioxidantDefense(),
    }
    topology = {
        "ros_prod": {"ros": "cytoplasm/ROS"},
        "antioxidant": {"ros": "cytoplasm/ROS"},
    }

    click.echo("Processes:")
    for name, proc in processes.items():
        click.echo(f"  {name}: {type(proc).__name__}")
        for pname, port in proc.ports_schema().items():
            click.echo(f"    port '{pname}' -> {port}")

    click.echo(f"\nTopology: {json.dumps(topology, indent=2)}")

    composite = Composite(
        processes,
        topology,
        semantic_validation=validate,
    )

    click.echo(f"\nStore paths: {composite.store_paths()}")

    sim = Scheduler()
    result = sim.run(composite, t_span=(0.0, t1), macro_dt=dt, save_dt=dt)

    click.echo(f"\nSimulation t=[0, {t1}], dt={dt}")
    click.echo(f"Time points: {len(result.ts)}")

    ts = result.ts
    ros = result.get("cytoplasm/ROS")
    click.echo(f"\n{'t':>8s}  {'ROS (uM)':>10s}")
    click.echo(f"{'─' * 8}  {'─' * 10}")
    indices = list(range(min(5, len(ts)))) + list(
        range(max(5, len(ts) - 3), len(ts))
    )
    indices = sorted(set(i for i in indices if 0 <= i < len(ts)))
    prev = -1
    for i in indices:
        if prev >= 0 and i > prev + 1:
            click.echo(f"{'...':>8s}  {'...':>10s}")
        click.echo(f"{float(ts[i]):8.1f}  {float(ros[i]):10.4f}")
        prev = i

    ss = float(ros[-1])
    expected_ss = 0.05 / 0.02
    click.echo(
        f"\nSteady state ROS: {ss:.4f} uM (expected: {expected_ss:.1f} uM)"
    )


@demo.command("compose-kick")
@click.option("--t1", type=float, default=100.0, help="End time")
@click.option(
    "--kick-time", type=float, default=50.0, help="Time of perturbation"
)
@click.option("--kick-ros", type=float, default=5.0, help="ROS delta at kick")
def compose_kick(t1, kick_time, kick_ros):
    """Demo: composable simulation with a mid-run perturbation."""
    import jax.numpy as jnp

    from hallsim.composite import Composite
    from hallsim.process import Port, PortRole, Process
    from hallsim.scheduler import Scheduler

    class ROSProduction(Process):
        rate: float = 0.05

        def ports_schema(self):
            return {
                "ros": Port(role=PortRole.EVOLVED, default=0.1, units="uM")
            }

        def derivative(self, t, state):
            return {"ros": jnp.array(self.rate)}

    class AntioxidantDefense(Process):
        scavenge_rate: float = 0.02

        def ports_schema(self):
            return {
                "ros": Port(role=PortRole.EVOLVED, default=0.1, units="uM")
            }

        def derivative(self, t, state):
            return {"ros": -self.scavenge_rate * state["ros"]}

    processes = {
        "ros_prod": ROSProduction(),
        "antioxidant": AntioxidantDefense(),
    }
    topology = {
        "ros_prod": {"ros": "cytoplasm/ROS"},
        "antioxidant": {"ros": "cytoplasm/ROS"},
    }

    # Mid-run perturbation expressed as a first-class EVENT process: the
    # KickEvent fires once when its condition (t >= kick_time) becomes
    # true and applies its delta via the Scheduler's event dispatcher.
    from hallsim.models.kick_event import KickEvent

    processes["kick"] = KickEvent(
        kick_time=kick_time, deltas={"ros": kick_ros}
    )
    topology["kick"] = {"ros": "cytoplasm/ROS"}

    composite = Composite(processes, topology, semantic_validation=False)
    click.echo(
        f"Solving {len(processes)} processes over t=[0, {t1}] "
        f"(the first run compiles; a few seconds)..."
    )
    result = Scheduler().run(
        composite, t_span=(0.0, t1), macro_dt=1.0, save_dt=1.0
    )

    ts = result.ts
    ros = result.get("cytoplasm/ROS")

    click.echo(f"Perturbation: +{kick_ros} uM ROS at t={kick_time}")
    click.echo(f"\n{'t':>8s}  {'ROS (uM)':>10s}")
    click.echo(f"{'─' * 8}  {'─' * 10}")

    key_times = [0, kick_time - 1, kick_time, kick_time + 1, t1]
    for target in key_times:
        idx = int(jnp.argmin(jnp.abs(ts - target)))
        label = ""
        if (
            abs(float(ts[idx]) - kick_time) < 0.5
            and float(ts[idx]) >= kick_time
        ):
            label = "  <-- kick"
        click.echo(f"{float(ts[idx]):8.1f}  {float(ros[idx]):10.4f}{label}")

    click.echo(
        f"\nRecovery: ROS returns to ~{float(ros[-1]):.2f} uM (steady state: 2.5 uM)"
    )


@demo.command("validate")
@click.option("--strict", is_flag=True, help="Promote warnings to errors")
def validate_demo(strict):
    """Demo: show the semantic validation layer catching issues."""
    import jax.numpy as jnp

    from hallsim.process import Port, PortRole, Process
    from hallsim.validation import CompositeValidator

    class ProcessA(Process):
        rate: float = 0.1

        def ports_schema(self):
            return {
                "x": Port(
                    role=PortRole.EVOLVED,
                    default=1.0,
                    units="uM",
                    description="ROS concentration from mitochondrial damage",
                    ontology={"chebi": "CHEBI:26523"},
                ),
                "y": Port(role=PortRole.INPUT, default=0.5, units="uM"),
            }

        def derivative(self, t, state):
            return {"x": jnp.array(self.rate)}

    class ProcessB(Process):
        rate: float = 0.05

        def ports_schema(self):
            return {
                "x": Port(
                    role=PortRole.EVOLVED,
                    default=1.0,
                    units="nM",
                    description="ROS concentration from oxidative stress",
                ),
                "y": Port(role=PortRole.EVOLVED, default=0.5, units="uM"),
            }

        def derivative(self, t, state):
            return {"x": -self.rate * state["x"], "y": jnp.array(0.01)}

    processes = {"proc_a": ProcessA(), "proc_b": ProcessB()}
    topology = {
        "proc_a": {"x": "pool/ROS", "y": "pool/signal"},
        "proc_b": {"x": "pool/ROS", "y": "pool/signal"},
    }

    click.echo("Processes:")
    for name, proc in processes.items():
        click.echo(f"  {name}:")
        for pname, port in proc.ports_schema().items():
            ont = f", ontology={port.ontology}" if port.ontology else ""
            click.echo(
                f"    {pname}: {port.role.value}, units={port.units!r}{ont}"
            )

    click.echo("\nTopology:")
    for name, topo in topology.items():
        click.echo(f"  {name}: {topo}")

    validator = CompositeValidator(strict=strict)
    report = validator.validate(processes, topology)

    click.echo(f"\n{'═' * 60}")
    click.echo("VALIDATION REPORT")
    click.echo(f"{'═' * 60}")
    click.echo(report)
    click.echo(f"{'═' * 60}")
    click.echo(f"Valid: {report.is_valid}")

    if report.interaction_graph:
        click.echo(
            f"\nInteraction graph nodes: "
            f"{[n['id'] for n in report.interaction_graph.get('nodes', [])]}"
        )
        click.echo(
            f"Interaction graph edges: "
            f"{len(report.interaction_graph.get('links', []))}"
        )


@demo.command("multiscale")
@click.option("--t1", type=float, default=100.0, help="End time (seconds)")
@click.option(
    "--macro-dt", type=float, default=5.0, help="Macro step interval"
)
def multiscale(t1, macro_dt):
    """Demo: multi-timescale simulation with continuous + discrete + event processes."""
    import jax.numpy as jnp

    from hallsim.composite import Composite
    from hallsim.process import Port, PortRole, Process, ProcessKind
    from hallsim.scheduler import Scheduler

    class ROSProduction(Process):
        kind: ProcessKind = ProcessKind.CONTINUOUS
        timescale: float = eqx.field(static=True, default=1.0)
        rate: float = 0.5

        def ports_schema(self):
            return {
                "ros": Port(role=PortRole.EVOLVED, default=0.0, units="uM")
            }

        def derivative(self, t, state):
            return {"ros": jnp.array(self.rate)}

    class SlowDecay(Process):
        kind: ProcessKind = ProcessKind.CONTINUOUS
        timescale: float = eqx.field(static=True, default=100.0)
        rate: float = 0.01

        def ports_schema(self):
            return {
                "ros": Port(role=PortRole.EVOLVED, default=0.0, units="uM")
            }

        def derivative(self, t, state):
            return {"ros": -self.rate * state["ros"]}

    class HeartbeatCounter(Process):
        kind: ProcessKind = ProcessKind.DISCRETE
        dt_step: float = eqx.field(static=True, default=20.0)

        def ports_schema(self):
            return {
                "beats": Port(
                    role=PortRole.LATCHED, default=0.0, units="dimensionless"
                )
            }

        def update(self, t, state):
            return {"beats": jnp.array(1.0)}

    class ROSAlarm(Process):
        kind: ProcessKind = ProcessKind.EVENT
        threshold: float = 30.0

        def ports_schema(self):
            return {
                "ros": Port(role=PortRole.INPUT, default=0.0, units="uM"),
                "alarm": Port(
                    role=PortRole.LATCHED, default=0.0, units="dimensionless"
                ),
            }

        def condition(self, t, state):
            return state["ros"] > self.threshold

        def handler(self, t, state):
            return {"alarm": 1.0 - state["alarm"]}

    composite = Composite(
        processes={
            "ros_prod": ROSProduction(),
            "slow_decay": SlowDecay(),
            "heartbeat": HeartbeatCounter(),
            "alarm": ROSAlarm(),
        },
        topology={
            "ros_prod": {"ros": "cell/ROS"},
            "slow_decay": {"ros": "cell/ROS"},
            "heartbeat": {"beats": "state/heartbeats"},
            "alarm": {"ros": "cell/ROS", "alarm": "state/alarm"},
        },
    )

    click.echo("Multi-Timescale Demo")
    click.echo("=" * 50)
    click.echo()
    click.echo("Processes:")
    click.echo(
        "  ros_prod    [CONTINUOUS, ts=1s]    ROS production (rate=0.5 uM/s)"
    )
    click.echo(
        "  slow_decay  [CONTINUOUS, ts=100s]  first-order decay (rate=0.01/s)"
    )
    click.echo(
        "  heartbeat   [DISCRETE, dt=20s]     increments counter every 20s"
    )
    click.echo("  alarm       [EVENT]                fires when ROS > 30 uM")
    click.echo()

    groups = composite.auto_groups()
    click.echo(f"Auto-groups: {json.dumps({k: v for k, v in groups.items()})}")
    click.echo()

    scheduler = Scheduler()
    result = scheduler.run(composite, t_span=(0.0, t1), macro_dt=macro_dt)

    ts = result.ts
    ros = result.get("cell/ROS")
    beats = result.get("state/heartbeats")
    alarm = result.get("state/alarm")

    click.echo(f"{'t':>8s}  {'ROS (uM)':>10s}  {'beats':>6s}  {'alarm':>6s}")
    click.echo(f"{'─' * 8}  {'─' * 10}  {'─' * 6}  {'─' * 6}")

    for i in range(len(ts)):
        click.echo(
            f"{float(ts[i]):8.1f}  {float(ros[i]):10.4f}  "
            f"{int(float(beats[i])):6d}  {int(float(alarm[i])):6d}"
        )

    click.echo()
    click.echo(f"Events fired: {len(result.events)}")
    for ev in result.events:
        click.echo(f"  t={ev.time:.1f}: {ev.process} -> {ev.delta}")

    click.echo()
    click.echo(
        f"Final ROS: {float(ros[-1]):.2f} uM (steady state: {0.5 / 0.01:.0f} uM)"
    )
    click.echo(f"Final heartbeats: {int(float(beats[-1]))}")
    click.echo(f"Alarm triggered: {'yes' if float(alarm[-1]) > 0.5 else 'no'}")


@demo.command("clamp")
@click.option(
    "--level", type=float, default=2.0, show_default=True, help="Setpoint"
)
@click.option("--t1", type=float, default=60.0, show_default=True)
@click.option(
    "--rel-error",
    type=float,
    default=0.01,
    show_default=True,
    help="Residual offset the placed clamp rate must hold to.",
)
@click.option(
    "--k",
    "k_clamps",
    type=float,
    multiple=True,
    help="Clamp rates to overlay (repeatable). Default: 0.05, 0.2, 1.",
)
def clamp(level, t1, rel_error, k_clamps):
    """Chronic vs transient exposure: hold a consumed species at a setpoint.

    Runs a ligand-uptake model with and without a ClampEdge, sweeps the clamp
    rate against its residual offset, and writes the three-panel figure to
    ``outputs/clamp_setpoint/``.
    """
    from demos.clamp_setpoint import run_demo

    overrides = {"level": level, "t1": t1, "rel_error": rel_error}
    if k_clamps:
        overrides["k_clamps"] = list(k_clamps)
    run_demo(**overrides)


@simulate.command("find")
@click.argument("query", nargs=-1, required=True)
@click.option(
    "--produces",
    "pattern",
    required=True,
    help="Regex a deposit must EMIT, matched against species id and display "
    "name (e.g. 'IL6|CXCL8|MMP1'). Word-boundary it or TNF matches TNFR.",
)
@click.option(
    "--limit",
    default=15,
    show_default=True,
    help="Hits per query per repository.",
)
@click.option(
    "--sources",
    default=None,
    help="Comma-separated repositories; default is all of them.",
)
@click.option(
    "--triage/--no-triage",
    default=False,
    help="Run the numerical screen on the producers.",
)
@click.option(
    "--repos/--no-repos",
    default=True,
    help="Follow the code-availability links of the papers found and say "
    "what each repository holds: importable files, or source that needs "
    "translating.",
)
@click.option(
    "--repo-limit",
    type=int,
    default=10,
    help="repositories to classify per run (the forges rate-limit keyless "
    "calls)",
)
def find(query, pattern, limit, sources, triage, repos, repo_limit):
    """Search every repository for a model that EMITS a quantity.

    A text search answers "is this deposit about IL6"; composing needs "does
    it emit IL6". A module imported to supply an output it only ever consumes
    contributes nothing, which is the single most common way a candidate
    fails. Each query term is searched separately and the hits are pooled.

        simulate find NFkB inflammation --produces '\bIL6\b|\bCXCL8\b'

    Every candidate yields a row, including the ones that could not be
    screened: `no-reactions` is a qualitative deposit, `no-rate-laws` a drawn
    pathway map, and neither is a screened negative.
    """
    import re
    from collections import Counter

    from hallsim.discovery import screen_produced_species, search_for_model

    src = sources.split(",") if sources else None
    seen, cands = set(), []
    for term in query:
        for c in search_for_model(term, limit=limit, sources=src):
            if c.id not in seen:
                seen.add(c.id)
                cands.append(c)
    click.echo(
        f"{len(cands)} distinct candidates from {len(query)} quer"
        f"{'y' if len(query) == 1 else 'ies'}"
    )
    rows = screen_produced_species(cands, pattern)
    # A re-hosted deposit is screened under its BioModels accession, so a
    # row can carry that id rather than the candidate's own; two candidates
    # can also resolve to the same deposit.
    by = {}
    for c in cands:
        by.setdefault(c.id, c)
        embedded = re.search(r"BIOMD\d{10}", c.id)
        if embedded:
            by.setdefault(embedded.group(0), c)

    producers, listed = [], set()
    for r in rows:
        if r.status == "produces" and r.model_id not in listed:
            listed.add(r.model_id)
            producers.append(r)
    click.echo(f"\n=== PRODUCES /{pattern}/ ===")
    for r in sorted(producers, key=lambda r: -len(r.produced)):
        c = by.get(r.model_id)
        mark = "cur" if c is not None and c.curated else "UNC"
        click.echo(
            f"  {mark} {r.model_id:20s} n={r.n_species:4d} "
            f"rx={r.n_reactions:4d} {list(r.produced)[:8]}"
        )
        click.echo(f"      {c.name[:96] if c is not None else ''}")
    if not producers:
        click.echo("  (none)")

    click.echo(f"\noutcome: {dict(Counter(r.status for r in rows))}")
    for r in rows:
        if r.status not in ("produces", "no-match"):
            click.echo(f"  [{r.status}] {r.model_id}: {r.note[:88]}")

    papers = [c for c in cands if c.source == "europepmc"]
    if repos and papers:
        from hallsim.literature import repositories_cited

        click.echo(f"\n=== REPOSITORIES cited by {len(papers)} paper(s) ===")
        cited = repositories_cited(papers, limit=repo_limit)
        for cand, by_papers in cited:
            click.echo(
                f"  {cand.kind:18s} {cand.source}:{cand.id}  "
                f"{cand.description[:90]}"
            )
            click.echo(f"      cited by {', '.join(by_papers)}")
        if not cited:
            click.echo("  (none)")

    if triage and producers:
        from hallsim.intake import triage_sbml

        click.echo(f"\n=== TRIAGE of {len(producers)} producer(s) ===")
        for r in producers:
            click.echo(f"\n{by[r.model_id].name[:88]}")
            click.echo(str(triage_sbml(r.model_id)))


@simulate.command("find-data")
@click.argument("query", nargs=-1, required=True)
@click.option("--limit", type=int, default=20)
@click.option("--organism", default=None, help='e.g. "Homo sapiens"')
@click.option(
    "--check/--no-check",
    default=False,
    help="Read the head of each series' platform table from GEO and say "
    "how the loader would map its probes to genes.",
)
def find_data(query, limit, organism, check):
    """Search GEO for a dataset to calibrate against.

    Lists series with their kind, organism, platform and sample titles (the
    arms and timepoints are usually in the titles). Array expression series
    carry their values in the series matrix the loader reads; sequencing
    series usually do not.
    """
    from hallsim.datasets import (
        loader_route,
        platform_head,
        search_for_dataset,
    )

    seen, hits = set(), []
    for term in query:
        for d in search_for_dataset(term, limit=limit, organism=organism):
            if d.accession not in seen:
                seen.add(d.accession)
                hits.append(d)
    hits.sort(key=lambda d: not d.series_matrix_has_values)
    click.echo(f"{len(hits)} series")
    for d in hits:
        click.echo(
            f"\n{d.accession:11s} {d.short_kind:<14.14s} {d.organism:<16.16s} "
            f"{d.n_samples:4d} samples  {d.platform}"
        )
        click.echo(f"    {d.title[:100]}")
        if d.samples:
            click.echo(f"    samples: {', '.join(d.samples[:6])}")
        if check and d.series_matrix_has_values:
            click.echo(
                f"    platform: {loader_route(platform_head(d.accession))}"
            )


@simulate.command("screen")
@click.argument("model")
@click.option(
    "--t-end",
    type=float,
    default=10.0,
    help="solo solve window, in the model's native time unit",
)
def screen(model, t_end):
    """Screen one model on its own before it joins a composite.

    Imports MODEL (a BioModels id, or a path to an SBML or COPASI file),
    triages it, and runs the numerical screen: exploding, vanishing,
    tolerance-sensitive, not at rest, non-tunable. Nothing should be
    composed unscreened.
    """
    from pathlib import Path

    from hallsim.intake import triage_sbml

    if model.isdigit():
        target, name = int(model), f"biomodel_{model}"
    else:
        target, name = model, Path(model).stem
    verdict = triage_sbml(target, t_end=t_end, name=name)
    click.echo(str(verdict))
    if verdict.screen is not None:
        click.echo(str(verdict.screen))


@simulate.command("handles")
@click.argument("models", nargs=-1)
@click.option(
    "--demo",
    is_flag=True,
    help="Use the multi-hallmark demo composite instead of MODELS.",
)
def handles(models, demo):
    """Where the hallmarks of aging would land on a model.

    MODELS are BioModels ids or SBML/COPASI paths, imported side by side.
    For each hallmark intent (hallsim.hallmarks) the table lists the rates
    its annotated species take part in, as ready ParameterMapping entries,
    or says that nothing on the model is annotated the way it asks. A
    suggestion is a place the biology could act; review it before it goes
    into a registry.
    """
    from pathlib import Path

    from hallsim.composite import Composite
    from hallsim.handles import suggest_registry
    from hallsim.hallmarks import HALLMARK_INTENTS

    if demo:
        from demos.models.multi_hallmark import (
            build_multi_hallmark_composite,
        )

        comp = build_multi_hallmark_composite(validate=False)
    elif models:
        from hallsim.sbml_import import process_from_sbml

        procs = {}
        for m in models:
            target = int(m) if m.isdigit() else m
            name = f"biomodel_{m}" if m.isdigit() else Path(m).stem
            procs[name] = process_from_sbml(target, name=name)
        comp = Composite(
            processes=procs,
            topology={},
            validate=False,
            semantic_validation=False,
        )
    else:
        raise click.UsageError("give MODELS or --demo")
    for name, handle in suggest_registry(HALLMARK_INTENTS, comp).items():
        click.echo(f"\n{name}  [{handle.category}]  {handle.description}")
        if not handle.mappings:
            click.echo("    no annotated species matches")
        for m in handle.mappings:
            click.echo(
                f"    {m.process_name}.{m.param_name:<46s} "
                f"x(1 {'+' if m.slope >= 0 else '-'} {abs(m.slope):g}*s)"
            )
            click.echo(f"        {m.description}")


@simulate.command("rejections")
@click.option(
    "--class", "cls", default=None, help="Show only this failure class."
)
@click.option("--slot", default=None, help="Show only this slot.")
def rejections(cls, slot):
    """Which deposits were screened out of a slot, and why.

    A search that ends in "nothing suitable" is a result only when the reasons
    are recorded. The distribution of failure classes is a finding about the
    field, and the registry stops the same deposit being re-screened by the
    next session.
    """
    from hallsim.rejections import load, summary

    if cls is None and slot is None:
        click.echo(summary())
        return
    rows = [
        r
        for r in load()
        if (cls is None or r.failure_class == cls)
        and (slot is None or r.slot == slot)
    ]
    click.echo(f"{len(rows)} matching")
    for r in rows:
        click.echo(f"\n  {r.id}  {r.model}")
        click.echo(f"    {r.failure_class} ({r.slot}) — {r.reason}")
        click.echo(f"    evidence: {r.evidence}")


@demo.command("stiffness")
@click.option(
    "--macro-dt",
    default=5.0,
    show_default=True,
    help="Solve interval the stability-substep budget is measured against.",
)
def stiffness(macro_dt):
    """Report the per-group stiffness verdict of the multi-hallmark composite.

    Linearizes each auto-group at its initial state, measures the
    Jacobian spectrum, and shows which solver class the Scheduler would
    auto-select. Use it on any composite via
    ``hallsim.stiffness.analyze_groups``.
    """
    from demos.models.multi_hallmark import build_multi_hallmark_composite
    from hallsim.stiffness import analyze_groups

    comp = build_multi_hallmark_composite()
    report = analyze_groups(comp, dt=macro_dt)
    click.echo(f"Per-group stiffness @ macro_dt={macro_dt}")
    click.echo("=" * 60)
    for verdict in report.values():
        click.echo(str(verdict))
    click.echo()
    click.echo(
        "STIFF groups are integrated implicitly (Kvaerno5 + scaled vector "
        "atol); the rest explicitly (Tsit5). Stiffness index = "
        "max|Re λ| × macro_dt = stability-limited substeps per interval."
    )


@demo.command("gz06-damage-scan")
def gz06_damage_scan():
    """Which Geva-Zatorsky 2006 parameter should carry DNA damage?

    Scans the three degradation channels the DNA-damage literature implicates
    for a Hopf bifurcation, and reports which one a damage-signed move
    actually crosses. The pattern generalises: place a coupling edge on the
    parameter whose bifurcation the perturbation reaches, not on whichever
    parameter is exposed.
    """
    from demos.gz06_damage_channel_scan import main as run_scan

    run_scan()


@demo.command("multi-hallmark")
@click.argument(
    "command",
    type=click.Choice(
        [
            "run",
            "calibrate",
            "score",
            "screen",
            "sweep",
            "export",
            "fetch-data",
        ]
    ),
    default="run",
)
@click.option(
    "--lr",
    type=float,
    default=None,
    help="Adam learning rate / cosine start (default 0.005)",
)
@click.option(
    "--steps", type=int, default=None, help="fit steps (default 150)"
)
@click.option(
    "--no-cosine",
    "no_cosine",
    is_flag=True,
    help="constant LR + reduce-on-plateau instead of cosine decay",
)
@click.option(
    "--grad-clip",
    "grad_clip",
    type=float,
    default=None,
    help="clip gradient global-norm to this value",
)
@click.option(
    "--no-plateau",
    "no_plateau",
    is_flag=True,
    help="disable reduce-on-plateau LR schedule",
)
@click.option(
    "--equilibrate",
    is_flag=True,
    help="Newton-solve the whole composite to a fixed point and share it as "
    "t=0 (off by default: DP14 senescence is progressive and has none)",
)
@click.option(
    "--run",
    type=click.Path(),
    default=None,
    help="score: the calibrate run directory to re-score (default: latest)",
)
@click.option(
    "--pool",
    type=click.Choice(["all", "declared"]),
    default="all",
    help="screen: every calibratable parameter of the composite, or only "
    "the demo's declared candidate set",
)
@click.option(
    "--no-clamp",
    "no_clamp",
    is_flag=True,
    help="calibrate: let Mdm2 degradation leave the treated arm's "
    "oscillatory window (default: boxed inside it)",
)
@click.option(
    "--rapa-intensity",
    type=float,
    default=None,
    help="how hard rapamycin hits mTORC1 S2448 phosphorylation: severity -1 "
    "holds the rate at (1 - this) x published. A property of the dose, "
    "invisible to the etoposide arm, so it comes from the rapamycin arm "
    "(default: the demo's declared value)",
)
@click.option(
    "--fit",
    multiple=True,
    metavar="PARAM",
    help="fit exactly these parameters (repeatable); the rest stay at "
    "their placed values. A name from the demo's declared set, or any "
    "parameter as `process.key` the way `screen` names it (e.g. "
    "dp14.DNA_repair, p07.Source, mtor_synthesis.gain). Default: the "
    "four the paper reports",
)
def multi_hallmark(
    command,
    lr,
    steps,
    no_cosine,
    grad_clip,
    no_plateau,
    equilibrate,
    fit,
    rapa_intensity,
    run,
    pool,
    no_clamp,
):
    """The multi-hallmark composite (Dalle Pezze 2014 + Geva-Zatorsky 2006 +
    Proctor 2007) scored against GSE248823.

    \b
      run        score the composite out of the box, no fitting (default)
      calibrate  fit the mechanism parameters, evaluate on held-out arms
      score      re-score a saved fit under changed conditions, no refit
      sweep      two-hallmark severity sweep
      export     write a saved fit's composite as SBML, one document per arm
      fetch-data download GSE248823 from GEO (run does this on first use)
    """
    from types import SimpleNamespace

    from demos.multi_hallmark_calibrate import cmd_run, cmd_sweep

    # Fitting is the out-of-the-box run continued: cmd_run scores every arm,
    # then fits when args.calibrate is set, reusing that score as the baseline.
    args = SimpleNamespace(
        command=command,
        calibrate=command == "calibrate",
        lr=lr,
        steps=steps,
        cosine=not no_cosine,
        grad_clip=grad_clip,
        no_plateau=no_plateau,
        equilibrate=equilibrate,
        fit=fit,
        rapa_intensity=rapa_intensity,
        run=run,
        pool=pool,
        no_clamp=no_clamp,
    )
    from demos.multi_hallmark_calibrate import cmd_score, cmd_screen

    from demos.multi_hallmark_calibrate import cmd_export, cmd_fetch_data

    dispatch = {
        "sweep": cmd_sweep,
        "score": cmd_score,
        "screen": cmd_screen,
        "export": cmd_export,
        "fetch-data": cmd_fetch_data,
    }
    dispatch.get(command, cmd_run)(args)


@demo.command("proctor2007-ssa")
@click.option(
    "--runs",
    type=click.IntRange(min=2),
    default=6,
    show_default=True,
    help="Number of independent stochastic runs.",
)
@click.option(
    "--hours",
    type=click.FloatRange(min=0, min_open=True),
    default=24.0,
    show_default=True,
    help="Simulation duration in hours.",
)
@click.option(
    "--samples",
    type=click.IntRange(min=2),
    default=289,
    show_default=True,
    help="Number of saved time points.",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="First random seed; incremented for each run.",
)
@click.option(
    "--max-events",
    type=click.IntRange(min=1),
    default=2_000_000,
    show_default=True,
    help="Reaction event capacity per run.",
)
@click.option(
    "--inhibited",
    is_flag=True,
    help="Use inhibited proteasome activity (k69=0).",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory (default: repository outputs/proctor2007_ssa).",
)
def proctor2007_ssa(runs, hours, samples, seed, max_events, inhibited, output):
    """Plot repeated standalone Proctor 2007 Gillespie simulations."""
    from demos.proctor2007_ssa import run

    run(
        runs=runs,
        hours=hours,
        samples=samples,
        seed=seed,
        max_events=max_events,
        inhibited=inhibited,
        output=output,
    )


@demo.command("multi-hallmark-ssa")
@click.option("--t-end", type=float, default=14.0, show_default=True)
@click.option("--save-dt", type=float, default=0.1, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--max-events", type=int, default=10_000_000, show_default=True)
def multi_hallmark_ssa(t_end, save_dt, seed, max_events):
    """Run the one-way DP14/GZ06 + Proctor Gillespie hybrid."""
    from demos.multi_hallmark_ssa import run

    result, _ = run(
        t_end=t_end,
        save_dt=save_dt,
        seed=seed,
        max_events=max_events,
    )
    click.echo(
        f"SSA events={result.reaction_indices.size} "
        f"samples={len(result.times)}"
    )


@demo.command("hallmark-levers")
@click.option("--port", type=int, default=8050, show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--debug", is_flag=True, help="Dash debug mode")
@click.option(
    "--cells",
    type=click.IntRange(min=0),
    default=16,
    show_default=True,
    help="cells in the Proctor 2007 population (one Gillespie path each, "
    "run one host thread each). The presets are sampled at startup and "
    "each new setting once; a sample costs one chain per wave of cells "
    "over available cores. 0 draws the mean field instead",
)
@click.option(
    "--seed", type=int, default=0, show_default=True, help="population seed"
)
def hallmark_levers(port, host, debug, cells, seed):
    """Serve the hallmark-lever page: one slider per hallmark of aging,
    wired into the multi-hallmark composite. Every pull applies the
    severity through the hallmark layer, re-solves Dalle Pezze 2014,
    Geva-Zatorsky 2006 and Proctor 2007 as one system and redraws them
    against the etoposide arm. The etoposide exposure window is shaded,
    with longer windows on a switch; Proctor 2007 is drawn as a population
    of cells at reaction level, with the population mean over it.

    Needs the `app` extra: pip install "hallsim[app]".
    """
    from demos.hallmark_levers import main

    main(port=port, host=host, debug=debug, cells=cells, seed=seed)


@simulate.command("info")
def info():
    """Show info about the composable architecture."""
    click.echo("HallSim Composable Architecture")
    click.echo("================================")
    click.echo()
    click.echo("Core concepts:")
    click.echo(
        "  Process   — Equinox module declaring ports + computing derivatives/updates"
    )
    click.echo(
        "  Port      — Named connection point (INPUT / EVOLVED / EXCLUSIVE / LATCHED)"
    )
    click.echo("  Topology  — Wires process ports to shared store paths")
    click.echo(
        "  Composite — Bundles processes + topology, auto-groups by timescale"
    )
    click.echo(
        "  Scheduler — Default runner: multi-rate, native batched populations,"
    )
    click.echo("              JIT/grad-friendly, single-group fast path")
    click.echo()
    click.echo("Process kinds:")
    click.echo(
        "  CONTINUOUS — derivative(t, state) -> dy/dt, solved by Diffrax ODE"
    )
    click.echo(
        "  DISCRETE   — update(t, state) -> delta, called every dt_step seconds"
    )
    click.echo(
        "  EVENT      — condition + handler, fires on False->True crossing"
    )
    click.echo()
    click.echo("Validation layer:")
    click.echo("  UnitChecker      — pint-based dimensional analysis")
    click.echo("  SemanticChecker  — ontology-based species disambiguation")
    click.echo("  GraphAnalyzer    — feedback loops, fan-in, coupling density")
    click.echo("  CouplingAuditor  — duplicate reaction detection")
    click.echo()
    click.echo("CLI commands:")
    rows = []
    for name, cmd in sorted(simulate.commands.items()):
        if isinstance(cmd, click.Group):
            rows += [
                (f"{name} {sub}", c) for sub, c in sorted(cmd.commands.items())
            ]
        else:
            rows.append((name, cmd))
    width = max(len(n) for n, _ in rows)
    for name, cmd in rows:
        click.echo(
            f"  simulate {name:<{width}}  — {cmd.get_short_help_str(limit=70)}"
        )
    click.echo()
    click.echo("Python usage:")
    click.echo(
        "  from hallsim.process import Process, Port, PortRole, ProcessKind"
    )
    click.echo("  from hallsim.composite import Composite")
    click.echo("  from hallsim.scheduler import Scheduler")
    click.echo("  from hallsim.validation import CompositeValidator")


if __name__ == "__main__":
    simulate()
