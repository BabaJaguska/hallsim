"""Batched population of the FULL multi-hallmark composite.

The single-model population (``gz06_population.py``) makes a mechanistic
point; this is the capability claim: the entire 7-process composite (three
published models + custom coupling branches, on one shared clock) run as a
heterogeneous population of ``N`` cells in a single differentiable
``Scheduler.run`` -- no per-cell Python loop.

Cells differ in two literature-grounded parameters (``alpha_y``, GZ06's
Mdm2 degradation / p53 period; ``CDKN1A_transcr``, p21 induction gain) and
in their initial state. The output is the population distribution of each
hallmark reporter, with the pooled mean a bulk assay would report. The
p53 arm desynchronises through the composite for the same reason it does
standalone -- ``alpha_y`` sets the period (see ``docs/gz06-population.md``).

Run:
    .venv_hallsim/bin/python demos/composite_population.py --n-cells 256
"""
from __future__ import annotations

import argparse
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from hallsim.composite import Composite
from hallsim.hallmarks import apply_hallmarks
from hallsim.models.multi_hallmark import build_multi_hallmark_composite
from hallsim.scheduler import Scheduler

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA_Y = ("gz06", "alpha_y", 0.8)
CDKN1A = ("dp14", "CDKN1A_transcr_by_FoxO3a_n_DNA_damage", 0.085)
# Reporters read at the shared macro-step resolution (the coupled state a
# bulk assay would sample). p53 pulse dynamics need sub-hour resolution and
# are covered standalone in gz06_population.py; here we read the slow,
# integrating hallmark reporters the composite couples together.
REPORTERS = {
    "p21 (CDKN1A)": "dp14/CDKN1A",
    "NFKBIA (IkBat)": "nfkb/IkBat",
    "mTORC1": "dp14/mTORC1_pS2448",
    "ROS": "dp14/ROS",
}


def lognormal(key, mean, cv, shape):
    s = np.sqrt(np.log(1.0 + cv**2))
    return mean * jnp.exp(s * jax.random.normal(key, shape) - s**2 / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=256)
    ap.add_argument("--gi", type=float, default=0.5)
    ap.add_argument("--dns", type=float, default=0.5)
    ap.add_argument("--param-cv", type=float, default=0.30)
    ap.add_argument("--ic-cv", type=float, default=0.10)
    ap.add_argument("--t-end", type=float, default=60.0)
    ap.add_argument("--macro-dt", type=float, default=5.0)
    ap.add_argument("--save-dt", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    from outdir import outdir
    ap.add_argument("--out", default=str(
        outdir("composite_population") / "composite_population.png"))
    a = ap.parse_args()

    base = build_multi_hallmark_composite()
    procs = apply_hallmarks(base.processes, {
        "Genomic Instability": a.gi, "Deregulated Nutrient Sensing": a.dns})
    comp = Composite(processes=procs, topology=base.topology, validate=False,
                     semantic_validation={"check_semantics": False})
    sched = Scheduler()
    keys = comp.store_keys()
    y0 = np.asarray(comp.initial_state_vec())
    N = a.n_cells
    print(f"composite: {len(procs)} processes, {len(keys)} state vars; "
          f"{N} cells")

    def run(ay, cd, y0i):
        c = eqx.tree_at(
            lambda c: (
                c.processes[ALPHA_Y[0]].parameters[ALPHA_Y[1]],
                c.processes[CDKN1A[0]].parameters[CDKN1A[1]],
            ),
            comp, (ay, cd),
        )
        return sched.run(c, (0.0, a.t_end), macro_dt=a.macro_dt,
                         save_dt=a.save_dt, y0=y0i).ys

    k = jax.random.PRNGKey(a.seed)
    k1, k2, k3 = jax.random.split(k, 3)
    ay = lognormal(k1, ALPHA_Y[2], a.param_cv, (N,))
    cd = lognormal(k2, CDKN1A[2], a.param_cv, (N,))
    y0b = jnp.asarray(y0)[None, :] * lognormal(k3, 1.0, a.ic_cv, (N, len(y0)))

    t0 = time.time()
    ys = np.asarray(jax.jit(jax.vmap(run))(ay, cd, y0b))  # (N, n_time, n_vars)
    print(f"batched solve: {ys.shape}  in {time.time()-t0:.1f}s  "
          f"finite={np.isfinite(ys).all()}")

    idx = {k: i for i, k in enumerate(keys)}
    reporters = {l: p for l, p in REPORTERS.items() if p in idx}
    print(f"\nPopulation reporters (N={N}, param CV={a.param_cv:.2f}):")
    dist, cvs = {}, {}
    for label, path in reporters.items():
        val = ys[:, -1, idx[path]]  # per-cell endpoint
        dist[label] = val
        cvs[label] = val.std() / abs(val.mean()) * 100
        print(f"  {label:20s} mean={val.mean():.4g}  "
              f"CV={cvs[label]:5.1f}%  "
              f"range=[{val.min():.4g}, {val.max():.4g}]")

    _plot(dist, cvs, a.out, N)
    print(f"\nsaved figure -> {a.out}")


def _plot(dist, cvs, out, N):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(dist.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    # per-cell reporter distributions, normalized to population median
    ax = axes[0]
    data = [dist[l] / np.median(dist[l]) for l in labels]
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#0e7490")
        pc.set_alpha(0.35)
    for i, l in enumerate(labels, 1):
        ax.scatter(np.full(len(dist[l]), i)
                   + np.random.uniform(-0.06, 0.06, len(dist[l])),
                   dist[l] / np.median(dist[l]), s=4, color="#0e7490",
                   alpha=0.25, zorder=3)
    ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("per-cell value / population median")
    ax.set_title(f"Every hallmark reporter is now a distribution\n"
                 f"({N} cells, one batched composite solve)", fontsize=10)
    # CV bars
    ax = axes[1]
    y = np.arange(len(labels))
    ax.barh(y, [cvs[l] for l in labels], color="#0e7490", alpha=0.8)
    for i, l in enumerate(labels):
        ax.text(cvs[l] + 0.5, i, f"{cvs[l]:.0f}%", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("population CV (%)")
    ax.set_title("Cell-to-cell spread by reporter", fontsize=10)
    fig.suptitle("The full 7-process composite, run as a heterogeneous "
                 "population in one differentiable solve", fontsize=12, y=1.03)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
