"""Batched population of the full multi-hallmark composite (paper Sec 3.2).

The single-model population (``gz06_population.py``) makes the mechanistic
point; this is the capability claim: the entire multi-process composite
(three published SBML models + custom coupling branches, on one shared
clock) run as a heterogeneous population of ``N`` cells in one batched,
differentiable ``Scheduler.run`` -- no per-cell Python loop.

Cells differ in two literature-grounded parameters (``alpha_y``, GZ06's
Mdm2 degradation rate, which sets the p53 period; ``CDKN1A`` transcription,
the p21 induction gain) and in their initial state, each drawn per cell
from a lognormal. Every cell is equilibrated to its own control fixed
point before the perturbation, the same baseline protocol the calibration
uses. Every embedded p53-Mdm2 oscillator sustains, but the cells' periods
differ, so the single-cell oscillations dephase and the population (bulk)
p53 mean damps. That variability propagates to the hallmark reporters of
:data:`hallsim.gene_reporters.MULTI_HALLMARK_REPORTERS`, read through their
canonical summaries, which become distributions rather than single numbers.

Two outputs, both written under ``outputs/composite_population/``:
  * a publication figure (p53 dephasing -> damped bulk; reporter
    distributions at the readout day),
  * a stats table (runtime, per-cell period spread, bulk coherence decay,
    per-reporter distribution summary) as JSON + CSV.

The solve caches its raw arrays to an ``.npz``; re-run with ``--from-npz``
to regenerate figures and stats without re-solving.

Run:
    .venv/bin/python demos/composite_population.py --n-cells 1000
    .venv/bin/python demos/composite_population.py \
        --from-npz outputs/composite_population/composite_population.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from hallsim.gene_reporters import MULTI_HALLMARK_REPORTERS  # noqa: E402
from hallsim.hallmarks import with_hallmarks  # noqa: E402
from hallsim.io import outdir  # noqa: E402
from hallsim.models.multi_hallmark import (  # noqa: E402
    build_multi_hallmark_composite,
)
from hallsim.oscillation import coherence_curve, dominant_period  # noqa: E402
from hallsim.scheduler import Scheduler  # noqa: E402
from hallsim.steady_state import conservation_laws, steady_state  # noqa: E402

log = logging.getLogger("composite_population").info

# Heterogeneous parameters: (process, param, population mean).
ALPHA_Y = ("gz06", "alpha_y", 0.8)
CDKN1A = ("dp14", "CDKN1A_transcr_by_FoxO3a_n_DNA_damage", 0.085)

# The pre-perturbation baseline every cell is equilibrated to. Nutrient
# sensing is a rapamycin dial, so the baseline and the perturbation hold it
# at the same severity: the damage arm is then the only difference between
# them, and any mTOR response comes from DP14's own mechanisms.
CTRL = {"Genomic Instability": 0.0, "Deregulated Nutrient Sensing": 0.5}

# p53/Mdm2 carry the oscillation the bulk damps; the reporters are the
# canonical gene readouts of the composite, read via their own summaries.
OSC = {"p53": "gz06/x", "Mdm2": "gz06/y"}
REPORTERS = {r.gene_symbol: r for r in MULTI_HALLMARK_REPORTERS}
CAPTURE = {**OSC, **{g: r.observable for g, r in REPORTERS.items()}}

COHERENCE_WINDOW = 1.5

# Palette, congruent with the other multi_hallmark figures.
INK = "#1f2937"
ACCENT = "#d97706"
POP = "#c9ccd1"


def lognormal(key, mean, cv, shape):
    if cv == 0.0:
        return jnp.full(shape, mean)
    s = np.sqrt(np.log(1.0 + cv**2))
    return mean * jnp.exp(s * jax.random.normal(key, shape) - s**2 / 2)


def _cell(comp, ay, cd):
    return eqx.tree_at(
        lambda c: (
            c.processes[ALPHA_Y[0]].parameters[ALPHA_Y[1]],
            c.processes[CDKN1A[0]].parameters[CDKN1A[1]],
        ),
        comp,
        (ay, cd),
    )


def solve(a):
    """Sweep the population size; returns the largest run's raw arrays plus
    the wall time of every size.

    Each size is one ``jax.vmap`` over ``Scheduler.run`` (optionally chunked),
    and every size is a prefix of the same drawn population, so the timings
    are the same cells solved in bigger batches.
    """
    base = build_multi_hallmark_composite()
    ctrl = with_hallmarks(base, CTRL)
    pert = with_hallmarks(
        base,
        {
            "Genomic Instability": a.gi,
            "Deregulated Nutrient Sensing": a.dns,
        },
    )
    # Both timescale groups are stiff (index ~1e4); the default explicit
    # solver rejects a fifth of its steps here and fails outright at DDIS
    # severity, so route each group through the stiffness analyzer.
    sched = Scheduler(auto_stiffness=True)
    keys = pert.store_keys()
    cap_idx = jnp.asarray([keys.index(p) for p in CAPTURE.values()])
    # Conservation laws are structural, so the nominal cell's hold for all.
    laws = conservation_laws(ctrl, ctrl.initial_state_vec())
    # Newton on this network has a spurious root with negative NF-kB species;
    # a forward pre-solve seeds it onto the physical branch. The nominal
    # cell's seed serves the whole population -- cells differ only in two
    # parameters, so each cell's own Newton converges from it.
    seed = sched.run(
        ctrl,
        (0.0, a.presolve_days),
        macro_dt=a.macro_dt,
        save_dt=a.presolve_days,
        y0=ctrl.initial_state_vec(),
    ).ys[-1]
    # The per-group verdict needs concrete eigenvalues, so resolve it eagerly
    # before the population runs under vmap.
    for name, integ in sched.warm_up(
        pert, (0.0, a.t_end), a.macro_dt, y0=seed
    ).items():
        log(f"{name}: {type(integ.solver).__name__} stiff={integ.stiff}")
    sizes = sorted(a.n_sweep)
    N = sizes[-1]
    log(f"composite: {len(pert.processes)} processes, {len(keys)} vars; "
        f"sizes {sizes}")

    def run(ay, cd, jitter):
        # Jitter sets this cell's conserved totals, then its own ctrl fixed
        # point is solved for them: every cell opens the run at its own
        # homeostasis, so the response is the perturbation and not a
        # relaxation from a state no cell actually sits in.
        ref = seed * jitter
        y0 = steady_state(
            _cell(ctrl, ay, cd), y_guess=ref, laws=laws, y_ref=ref
        )
        ys = sched.run(
            _cell(pert, ay, cd),
            (0.0, a.t_end),
            macro_dt=a.macro_dt,
            save_dt=a.save_dt,
            y0=y0,
        ).ys
        return ys[:, cap_idx], y0

    batched = eqx.filter_jit(jax.vmap(run))

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(a.seed), 3)
    ay = lognormal(k1, ALPHA_Y[2], a.param_cv, (N,))
    cd = lognormal(k2, CDKN1A[2], a.param_cv, (N,))
    jitter = lognormal(k3, 1.0, a.ic_cv, (N, len(keys)))

    def population(n):
        # Chunking bounds the batch a stiff outlier can gate under the shared
        # adaptive step controller; chunk=0 solves all n cells in one batch.
        # Compilation is timed apart from execution (ahead-of-time lowering)
        # so the sweep compares solves, not XLA compiles.
        step = a.chunk or n
        caps, base_y0, compile_s, run_s = [], [], 0.0, 0.0
        for s in range(0, n, step):
            args = (ay[s:s + step], cd[s:s + step], jitter[s:s + step])
            t0 = time.time()
            compiled = batched.lower(*args).compile()
            t1 = time.time()
            out, y0 = compiled(*args)
            jax.block_until_ready(out)
            compile_s += t1 - t0
            run_s += time.time() - t1
            caps.append(np.asarray(out))
            base_y0.append(np.asarray(y0)[:, cap_idx])
        return (
            np.concatenate(caps, 0),
            np.concatenate(base_y0, 0),
            compile_s,
            run_s,
        )

    dev = str(jax.devices()[0])
    timings = []
    for n in sizes:
        caps, baseline, compile_s, run_s = population(n)
        timings.append((n, compile_s, run_s))
        log(
            f"batched solve: {caps.shape} in {run_s:.1f}s "
            f"({n / run_s:.2f} cells/s, +{compile_s:.1f}s compile)  "
            f"finite={np.isfinite(caps).all()}  dev={dev}"
        )
    return {
        "caps": caps,
        "baseline": baseline,
        "cap_labels": np.array(list(CAPTURE.keys())),
        "cap_paths": np.array(list(CAPTURE.values())),
        "ay": np.asarray(ay),
        "cd": np.asarray(cd),
        "timings": np.array(timings, dtype=float),
        "runtime": timings[-1][1],
        "n_cells": N,
        "chunk": a.chunk,
        "gi": a.gi,
        "dns": a.dns,
        "param_cv": a.param_cv,
        "ic_cv": a.ic_cv,
        "save_dt": a.save_dt,
        "readout_day": a.readout_day if a.readout_day else a.t_end,
        "device": dev,
    }


def _tg(data):
    """Reconstruct the save grid from array length (the scheduler runs to the
    next macro-step boundary, so it can exceed the requested t_end)."""
    n = data["caps"].shape[1]
    return np.arange(n) * float(data["save_dt"])


def _col(data, label):
    """One captured path as ``(n_time, n_cells)`` -- summaries and the
    oscillation readouts are time-first."""
    return data["caps"][:, :, list(data["cap_labels"]).index(label)].T


def _readout(data, tg, gene):
    """Per-cell reporter value at the readout day, through the reporter's own
    canonical summary (vmapped over the population's cell axis)."""
    r = REPORTERS[gene]
    t_read = float(data["readout_day"])
    summarize = jax.vmap(r.summary, in_axes=(None, 1, None))
    return np.asarray(
        summarize(jnp.asarray(tg), jnp.asarray(_col(data, gene)), t_read)
    ).ravel()


def compute_stats(data):
    tg = _tg(data)
    p53 = _col(data, "p53")
    periods = np.asarray(dominant_period(jnp.asarray(tg), jnp.asarray(p53)))
    fp = periods[np.isfinite(periods)]
    tc, coh, sc, bk = (
        np.asarray(v)
        for v in coherence_curve(
            jnp.asarray(tg), jnp.asarray(p53), COHERENCE_WINDOW
        )
    )

    reporters = {}
    for gene in REPORTERS:
        v = _readout(data, tg, gene)
        q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
        reporters[gene] = {
            "observable": REPORTERS[gene].observable,
            "mean": float(v.mean()),
            "median": float(q[2]),
            "std": float(v.std()),
            "cv_pct": float(v.std() / abs(v.mean()) * 100),
            "iqr": float(q[3] - q[1]),
            "q05": float(q[0]),
            "q95": float(q[4]),
            "min": float(v.min()),
            "max": float(v.max()),
        }
    t = np.atleast_2d(np.asarray(data["timings"], dtype=float))
    return {
        "n_cells": int(data["n_cells"]),
        "runtime_s": float(data["runtime"]),
        "chunk": int(data["chunk"]),
        "scaling": [
            {
                "n_cells": int(n),
                "compile_s": float(c),
                "solve_s": float(r),
                "cells_per_s": float(n / r),
            }
            for n, c, r in t
        ],
        "device": str(data["device"]),
        "genomic_instability": float(data["gi"]),
        "deregulated_nutrient_sensing": float(data["dns"]),
        "param_cv": float(data["param_cv"]),
        "ic_cv": float(data["ic_cv"]),
        "t_end_days": float(tg[-1]),
        "readout_day": float(data["readout_day"]),
        "p53_period_days_median": float(np.median(fp)),
        "p53_period_hours_median": float(np.median(fp) * 24),
        "p53_period_cv_pct": float(fp.std() / fp.mean() * 100),
        "bulk_coherence_start": float(coh[0]),
        "bulk_coherence_end": float(coh[-1]),
        "reporters": reporters,
        "_curves": {"tc": tc, "coh": coh, "sc": sc, "bk": bk},
        "_periods": periods,
    }


def write_stats(stats, out_dir):
    flat = {k: v for k, v in stats.items() if not k.startswith("_")}
    (out_dir / "composite_population_stats.json").write_text(
        json.dumps(flat, indent=2)
    )
    cols = [
        "gene",
        "observable",
        "mean",
        "median",
        "std",
        "cv_pct",
        "iqr",
        "q05",
        "q95",
        "min",
        "max",
    ]
    lines = [",".join(cols)]
    for gene, r in stats["reporters"].items():
        lines.append(
            ",".join(
                [gene, r["observable"]]
                + [f"{r[c]:.6g}" for c in cols[2:]]
            )
        )
    (out_dir / "composite_population_reporters.csv").write_text(
        "\n".join(lines) + "\n"
    )


def make_figures(data, stats, out_dir, n_traces=60):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        }
    )

    tg = _tg(data)
    p53 = _col(data, "p53")
    bulk = p53.mean(1)
    N = int(data["n_cells"])
    periods = stats["_periods"]
    rng = np.random.default_rng(0)
    sub = rng.choice(N, size=min(n_traces, N), replace=False)
    exemplar = int(np.nanargmin(np.abs(periods - np.nanmedian(periods))))

    # ---- Main figure: dephasing + reporter distributions ----------------
    fig, (axA, axB) = plt.subplots(
        1,
        2,
        figsize=(13.5, 4.6),
        gridspec_kw={"width_ratios": [1.35, 1]},
        layout="constrained",
    )

    axA.plot(tg, p53[:, sub], color=POP, lw=0.5, zorder=1)
    axA.plot(
        tg, p53[:, exemplar], color=INK, lw=1.0, zorder=3,
        label="one cell (sustained)",
    )
    axA.plot(
        tg, bulk, color=ACCENT, lw=2.3, zorder=4,
        label=f"bulk mean (N={N})",
    )
    axA.set_xlabel("time (days)")
    axA.set_ylabel("p53  (gz06/x)")
    axA.set_xlim(tg[0], tg[-1])
    axA.margins(y=0.02)
    axA.set_title(
        "Single cells keep oscillating;\n"
        "the bulk mean damps as they dephase",
        fontsize=10.5,
    )
    axA.legend(loc="upper right", fontsize=9, frameon=False)

    genes = list(REPORTERS)
    values = {g: _readout(data, tg, g) for g in genes}
    long = pd.DataFrame(
        {
            "gene": np.repeat(genes, N),
            "value": np.concatenate(
                [values[g] / np.median(values[g]) for g in genes]
            ),
        }
    )
    sns.violinplot(
        long, x="gene", y="value", order=genes, ax=axB,
        color=ACCENT, saturation=1.0, inner="box", cut=0, linewidth=0.9,
        width=0.85,
    )
    for coll in axB.collections:
        coll.set_alpha(0.35)
    sns.stripplot(
        long, x="gene", y="value", order=genes, ax=axB,
        color=ACCENT, size=1.4, alpha=0.3, jitter=0.18, zorder=2,
    )
    axB.axhline(1.0, color="0.6", lw=0.8, ls="--", zorder=0)
    lo, hi = long.value.min(), long.value.max()
    axB.set_ylim(lo - 0.04 * (hi - lo), hi + 0.16 * (hi - lo))
    for i, g in enumerate(genes):
        axB.text(
            i, hi + 0.04 * (hi - lo),
            f"CV {stats['reporters'][g]['cv_pct']:.0f}%",
            ha="center", va="bottom", fontsize=8.5, color=INK,
        )
    axB.set_xlabel("")
    axB.set_xticks(range(len(genes)))
    axB.set_xticklabels(genes, rotation=20, ha="right", fontsize=9)
    axB.set_ylabel(
        "per-cell value / population median\n"
        f"(day {stats['readout_day']:.0f})"
    )
    axB.set_title(
        "Each hallmark reporter is now\na distribution", fontsize=10.5
    )
    sns.despine(fig=fig)
    fig.suptitle(
        f"The full composite as a {N}-cell heterogeneous population, "
        "one batched solve",
        fontsize=12,
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"composite_population.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)

    # ---- Supplementary: coherence decay + period spread -----------------
    c = stats["_curves"]
    figs, (s0, s1) = plt.subplots(
        1, 2, figsize=(11, 3.9), layout="constrained"
    )
    s0.plot(c["tc"], c["sc"], color=INK, lw=1.8, marker="o", ms=4,
            label="single-cell amplitude")
    s0.plot(c["tc"], c["bk"], color=ACCENT, lw=2.2, marker="s", ms=4,
            label="bulk amplitude")
    s0.set_xlabel("time (days)")
    s0.set_ylabel("p53 peak-to-trough amplitude")
    s0.set_title("Bulk amplitude decays while single cells persist",
                 fontsize=10.5)
    s0.legend(fontsize=9, frameon=False)

    fp = periods[np.isfinite(periods)] * 24
    s1.hist(fp, bins=30, color=ACCENT, alpha=0.85, edgecolor=INK,
            linewidth=0.5)
    s1.axvline(
        np.median(fp), color=INK, lw=1.6, ls="--",
        label=f"median {np.median(fp):.2f} h",
    )
    s1.set_xlabel("per-cell p53 period (hours)")
    s1.set_ylabel("cells")
    s1.set_title(
        f"Period spread drives dephasing "
        f"(CV {stats['p53_period_cv_pct']:.1f}%)",
        fontsize=10.5,
    )
    s1.legend(fontsize=9, frameon=False)
    for ext in ("png", "pdf"):
        figs.savefig(out_dir / f"composite_population_coherence.{ext}",
                     dpi=200, bbox_inches="tight")
    plt.close(figs)

    # ---- Supplementary: batch scaling -----------------------------------
    t = np.atleast_2d(np.asarray(data["timings"], dtype=float))
    n, compile_s, run_s = t[:, 0], t[:, 1], t[:, 2]
    figb, (b0, b1) = plt.subplots(
        1, 2, figsize=(10, 3.7), layout="constrained"
    )
    b0.plot(n, run_s, "-o", color=ACCENT, lw=2, ms=5, label="solve")
    b0.plot(n, compile_s, "-s", color=INK, lw=1.4, ms=4, label="XLA compile")
    b0.set_xscale("log", base=2)
    b0.set_yscale("log")
    b0.set_xlabel("cells in the batch")
    b0.set_ylabel("wall time (s)")
    b0.set_title("One vmapped Scheduler.run per point", fontsize=10.5)
    b0.legend(fontsize=9, frameon=False)
    b1.plot(n, n / run_s, "-o", color=ACCENT, lw=2, ms=5)
    b1.set_xscale("log", base=2)
    b1.set_xlabel("cells in the batch")
    b1.set_ylabel("cells / s")
    b1.set_title("Throughput vs batch size", fontsize=10.5)
    for ext in ("png", "pdf"):
        figb.savefig(out_dir / f"composite_population_scaling.{ext}",
                     dpi=200, bbox_inches="tight")
    plt.close(figb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n-sweep",
        type=lambda s: [int(n) for n in s.split(",")],
        default=[128, 256, 512, 1024],
        help="population sizes to time; the largest one makes the figure",
    )
    # Etoposide-level damage, nutrient sensing held at the CTRL severity.
    ap.add_argument("--gi", type=float, default=1.0)
    ap.add_argument(
        "--dns", type=float, default=CTRL["Deregulated Nutrient Sensing"]
    )
    ap.add_argument("--param-cv", type=float, default=0.30)
    ap.add_argument("--ic-cv", type=float, default=0.10)
    ap.add_argument("--t-end", type=float, default=14.0)
    ap.add_argument(
        "--presolve-days",
        type=float,
        default=20.0,
        help="ctrl forward pre-solve that seeds the per-cell Newton baseline",
    )
    ap.add_argument("--readout-day", type=float, default=None)
    ap.add_argument("--macro-dt", type=float, default=5.0)
    ap.add_argument("--save-dt", type=float, default=0.02)
    ap.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="cells per batch; 0 (default) is one batch for the whole "
             "population -- set it only to fit a smaller GPU",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--from-npz",
        default=None,
        help="load a cached solve and only regenerate figures + stats",
    )
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_dir = outdir("composite_population")
    npz = out_dir / "composite_population.npz"

    if a.from_npz:
        raw = np.load(a.from_npz, allow_pickle=True)
        data = {k: raw[k] for k in raw.files}
    else:
        data = solve(a)
        np.savez_compressed(npz, **data)
        log(f"saved arrays -> {npz}")

    stats = compute_stats(data)
    write_stats(stats, out_dir)
    make_figures(data, stats, out_dir)
    log(f"saved figures + stats -> {out_dir}")
    log(
        f"p53 period: median {stats['p53_period_hours_median']:.2f} h, "
        f"CV {stats['p53_period_cv_pct']:.1f}%   "
        f"bulk coherence {stats['bulk_coherence_start']:.2f} "
        f"-> {stats['bulk_coherence_end']:.2f}"
    )
    for gene, r in stats["reporters"].items():
        log(f"  {gene:8s} {r['observable']:22s} CV {r['cv_pct']:5.1f}%  "
            f"median {r['median']:.4g}  IQR {r['iqr']:.4g}")


if __name__ == "__main__":
    main()
