import json

import click


@click.group()
def simulate():
    """HallSim simulation commands."""
    pass


# ── Composable architecture commands ─────────────────────────────────────


@simulate.command()
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


@simulate.command("compose-kick")
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


@simulate.command("validate-demo")
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


@simulate.command("multiscale")
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
        timescale: float = 1.0
        rate: float = 0.5

        def ports_schema(self):
            return {
                "ros": Port(role=PortRole.EVOLVED, default=0.0, units="uM")
            }

        def derivative(self, t, state):
            return {"ros": jnp.array(self.rate)}

    class SlowDecay(Process):
        kind: ProcessKind = ProcessKind.CONTINUOUS
        timescale: float = 100.0
        rate: float = 0.01

        def ports_schema(self):
            return {
                "ros": Port(role=PortRole.EVOLVED, default=0.0, units="uM")
            }

        def derivative(self, t, state):
            return {"ros": -self.rate * state["ros"]}

    class HeartbeatCounter(Process):
        kind: ProcessKind = ProcessKind.DISCRETE
        dt_step: float = 20.0

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


@simulate.command("clamp")
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
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "demos"))
    from clamp_setpoint import main

    overrides = {"level": level, "t1": t1, "rel_error": rel_error}
    if k_clamps:
        overrides["k_clamps"] = list(k_clamps)
    main(**overrides)


@simulate.command("stiffness")
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
    from hallsim.models.multi_hallmark import build_multi_hallmark_composite
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


@simulate.command("multi-hallmark")
@click.argument(
    "command",
    type=click.Choice(["run", "calibrate", "sweep"]),
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
def multi_hallmark(
    command, lr, steps, no_cosine, grad_clip, no_plateau, equilibrate
):
    """The multi-hallmark composite (DallePezze 2014 + Geva-Zatorsky 2006 +
    Ihekwaba 2004) scored against GSE248823.

    \b
      run        score the composite out of the box, no fitting (default)
      calibrate  fit the mechanism parameters, evaluate on held-out arms
      sweep      two-hallmark severity sweep
    """
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "demos"))
    from multi_hallmark_calibrate import _COMMANDS

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
    )
    _COMMANDS["run" if command == "calibrate" else command](args)


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
    click.echo(
        "  simulate compose          — demo: ROS production + antioxidant defense"
    )
    click.echo("  simulate compose-kick     — demo: perturbation + recovery")
    click.echo(
        "  simulate multiscale       — demo: continuous + discrete + event scheduling"
    )
    click.echo(
        "  simulate validate-demo    — demo: validation catching unit/semantic issues"
    )
    click.echo(
        "  simulate stiffness        — per-group Jacobian-spectrum solver verdict"
    )
    click.echo(
        "  simulate calibrate        — fit the multi-hallmark composite to data"
    )
    click.echo("  simulate info             — this help")
    click.echo()
    click.echo("Python usage:")
    click.echo(
        "  from hallsim.process import Process, Port, PortRole, ProcessKind"
    )
    click.echo("  from hallsim.composite import Composite")
    click.echo("  from hallsim.scheduler import Scheduler")
    click.echo("  from hallsim.validation import CompositeValidator")
