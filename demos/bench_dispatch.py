"""Dispatch-overhead benchmark: what does a repeated ``Scheduler.run`` cost?

Measures the host-side cost of getting to the solver, separately from the
solve itself, across the three axes that decide whether a run hits Diffrax's
JIT cache or recompiles:

1. **Repeat** — the same run issued N times. Flat-and-fast means the cache
   hits; flat-and-slow means every call retraces and recompiles.
2. **Parameter sweep** — the same value varied three ways (rerun unchanged,
   ``eqx.tree_at``, composite rebuilt per point). A parameter value is data,
   so none of them should recompile after the first point.
3. **Solver routing** — explicit (default) vs ``auto_stiffness=True`` on a
   stiff SBML import, where the cost is solver steps rather than compilation.

Also tracks RSS across the repeat sweep: retained compiled executables that
can never be reused show up as monotonic growth.

    python demos/bench_dispatch.py [config.json]
"""

import jax

jax.config.update("jax_enable_x64", True)

import gc
import json
import logging
import sys
import time

import equinox as eqx
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil

from hallsim.composite import single_process_composite
from hallsim.io import outdir
from hallsim.models.eriq import build_eriq_composite
from hallsim.sbml_import import process_from_sbml
from hallsim.scheduler import Scheduler

log = logging.getLogger("bench_dispatch")

DEFAULTS = {
    "t_end": 100.0,
    "macro_dt": 1.0,
    "n_repeat": 8,
    "sweep_values": [1.0, 1.1, 1.2, 1.3, 1.4],
    "stiff_model": "BIOMD0000000582",
    "stiff_macro_dt": 10.0,
    "outdir": "bench_dispatch",
    "plot": "bench_dispatch.png",
}

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

PROC = psutil.Process()


def rss_mb():
    return PROC.memory_info().rss / 1e6


# ── A pytree RHS: processes are dynamic leaves, index maps static ──────


def timed(fn):
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    return (time.perf_counter() - t0) * 1e3


# ── Measurements ───────────────────────────────────────────────────────


def measure_repeat(comp, cfg):
    """Wall time + RSS over N identical runs.

    Flat-and-fast means the compiled solve is being reused; flat-and-slow means
    every call retraces and recompiles, and RSS climbs with executables that can
    never be reused.
    """
    span = (0.0, cfg["t_end"])
    sched = Scheduler()
    ms, rss = [], []
    for _ in range(cfg["n_repeat"]):
        ms.append(
            timed(lambda: sched.run(comp, span, macro_dt=cfg["macro_dt"]).ys)
        )
        gc.collect()
        rss.append(rss_mb())
    log.info("repeat: %s", [f"{m:.0f}" for m in ms])
    return {"as shipped": (ms, rss)}


def measure_sweep(cfg):
    """Per-point cost of a parameter sweep, by how the value is varied.

    A parameter value is data and the compiled solve is keyed on structure, so
    all three routes should be flat after the first point. Any of them rising
    means something is being treated as structure that shouldn't be.
    """
    sched = Scheduler()
    span = (0.0, cfg["t_end"])
    base = build_eriq_composite(validate=False, semantic_validation=False)

    def rebuilt(sa):
        return build_eriq_composite(
            GLYCOL_SA=sa, validate=False, semantic_validation=False
        )

    def substituted(sa):
        return eqx.tree_at(
            lambda c: c.processes["energy"].GLYCOL_SA, base, jnp.asarray(sa)
        )

    variants = {
        "identical\nrerun": lambda sa: base,
        "eqx.tree_at\nper point": substituted,
        "composite rebuilt\nper point": rebuilt,
    }

    def run(c):
        return sched.run(c, span, macro_dt=cfg["macro_dt"]).ys

    out = {}
    for label, make in variants.items():
        run(make(cfg["sweep_values"][0]))  # absorb the first compile
        ms = [
            timed(lambda s=sa: run(make(s))) for sa in cfg["sweep_values"][1:]
        ]
        out[label] = sum(ms) / len(ms)
        log.info("%s: %.1f ms/point", label.replace("\n", " "), out[label])
    return out


def measure_routing(cfg):
    """Wall time + solver steps on a stiff SBML import, explicit vs auto."""
    comp = single_process_composite(
        process_from_sbml(cfg["stiff_model"]), "stiff"
    )
    span = (0.0, cfg["t_end"])
    out = {}
    for label, kw in (
        ("default\n(explicit)", {}),
        ("auto_stiffness\n=True", dict(auto_stiffness=True)),
    ):
        sched = Scheduler(max_steps=2_000_000, **kw)
        sched.warm_up(comp, span, macro_dt=cfg["stiff_macro_dt"])
        ms = timed(
            lambda: sched.run(comp, span, macro_dt=cfg["stiff_macro_dt"]).ys
        )
        res = sched.run(comp, span, macro_dt=cfg["stiff_macro_dt"])
        steps = sum(
            int(v["num_solver_steps"])
            for v in res.stats.values()
            if isinstance(v, dict) and "num_solver_steps" in v
        )
        solver = next(
            v["solver"]
            for v in res.stats.values()
            if isinstance(v, dict) and "solver" in v
        )
        out[label] = (ms, steps, solver)
        log.info(
            "%s: %.0f ms, %d steps, %s",
            label.replace("\n", " "),
            ms,
            steps,
            solver,
        )
    return out


# ── Figure ─────────────────────────────────────────────────────────────


def _style(ax):
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _bars(ax, labels, values, colors, fmt, log_scale=True):
    bars = ax.bar(range(len(labels)), values, color=colors, width=0.62)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5, color=MUTED)
    if log_scale:
        ax.set_yscale("log")
    top = max(values)
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v * 1.12 if log_scale else v + top * 0.02,
            fmt(v),
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
    ax.set_ylim(top=top * (4 if log_scale else 1.18))


def make_figure(repeat, sweep, routing, cfg, path):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    fig.patch.set_facecolor("#fcfcfb")

    # (a) repeated identical runs
    ax = axes[0, 0]
    for (label, (ms, _)), c in zip(repeat.items(), SERIES):
        ax.plot(
            range(1, len(ms) + 1), ms, "-o", color=c, lw=2, ms=6, label=label
        )
        ax.annotate(
            f"{ms[-1]:.1f} ms",
            (len(ms), ms[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            fontsize=9,
            color=INK,
            va="center",
        )
    ax.set_yscale("log")
    ax.set_xlabel("call number (identical run)")
    ax.set_ylabel("wall time (ms, log)")
    ax.set_title("Repeated identical Scheduler.run", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    ax.set_xlim(0.6, len(ms) + 0.9)
    _style(ax)

    # (b) resident memory over the same sweep
    ax = axes[0, 1]
    for (label, (_, rss)), c in zip(repeat.items(), SERIES):
        ax.plot(
            range(1, len(rss) + 1),
            [r - rss[0] for r in rss],
            "-o",
            color=c,
            lw=2,
            ms=6,
            label=label,
        )
    ax.set_xlabel("call number (identical run)")
    ax.set_ylabel("RSS growth from first call (MB)")
    ax.set_title(
        "Resident memory over the same calls", fontsize=11, loc="left"
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    ax.axhline(0, color=GRID, lw=1)
    _style(ax)

    # (c) parameter-sweep matrix
    ax = axes[1, 0]
    labels = list(sweep)
    vals = [sweep[k] for k in labels]
    # Linear, not log: these are meant to come out equal, and a log axis on
    # near-equal values renders them as slivers.
    _bars(
        ax,
        labels,
        vals,
        [SERIES[2]] * len(labels),
        lambda v: f"{v:.1f} ms",
        log_scale=False,
    )
    ax.set_ylabel("wall time per sweep point (ms)")
    ax.set_title(
        "Parameter sweep: the executable is reused however the value is varied",
        fontsize=11,
        loc="left",
    )
    _style(ax)

    # (d) solver routing on a stiff import
    ax = axes[1, 1]
    keys_ = list(routing)
    vals = [routing[k][0] for k in keys_]
    labels = [f"{k}\n{routing[k][2]} · {routing[k][1]:,} steps" for k in keys_]
    _bars(
        ax,
        labels,
        vals,
        [SERIES[1], SERIES[2]],
        lambda v: f"{v/1000:.2f} s",
    )
    ax.set_ylabel("wall time (ms, log)")
    ax.set_title(
        f"Stiff SBML import ({cfg['stiff_model']}): solver routing",
        fontsize=11,
        loc="left",
    )
    _style(ax)

    fig.suptitle(
        "HallSim dispatch overhead — host-side cost of reaching the solver",
        fontsize=13.5,
        color=INK,
        x=0.008,
        ha="left",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    log.info("wrote %s", path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("hallsim").setLevel(logging.ERROR)
    cfg = dict(DEFAULTS)
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            cfg.update(json.load(f))

    comp = build_eriq_composite(validate=False, semantic_validation=False)
    repeat = measure_repeat(comp, cfg)
    sweep = measure_sweep(cfg)
    routing = measure_routing(cfg)
    make_figure(
        repeat, sweep, routing, cfg, outdir(cfg["outdir"]) / cfg["plot"]
    )


if __name__ == "__main__":
    main()
