"""Batched GZ06 population: which heterogeneity desynchronizes the bulk.

Lifts the deterministic Geva-Zatorsky 2006 p53-Mdm2 oscillator
(BIOMD0000000157) to a heterogeneous population of ``N`` cells, runs them
in one batched Scheduler solve, and pools them the way a bulk assay does.
It contrasts three sources of cell-to-cell heterogeneity:

    beta_x   production rate  -> amplitude spread, NO desync
    alpha_y  Mdm2 degradation -> period spread     -> bulk damps
    both + random initial phase (realistic)        -> spread AND damping

The point: a bulk measurement is a *pooled* readout, and whether the pool
oscillates or damps is set by the spread in oscillation *period* (a phase
phenomenon), not the spread in amplitude. beta_x moves only amplitude;
alpha_y moves the period. See ``docs/gz06-population.md`` for the algebra.

The whole sweep runs through ``eqx.tree_at`` + ``jax.vmap`` over a single
``Scheduler.run`` -- native batched populations, no per-cell Python loop,
fully differentiable.

Run:
    .venv_hallsim/bin/python demos/gz06_population.py --n-cells 1000
"""

from __future__ import annotations

import argparse
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from hallsim.composite import Composite
from hallsim.scheduler import Scheduler
from hallsim.sbml_import import process_from_sbml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(
    _REPO, "models/zatorsky2006/zatorsky2006_BIOMD0000000157.xml"
)


def build():
    proc = process_from_sbml(XML, name="gz06")
    topo = {"gz06": {p: f"gz06/{p}" for p in proc.ports_schema()}}
    return Composite({"gz06": proc}, topology=topo)


def batched_runner(comp, sched, t_end, dt):
    """vmap one Scheduler.run over per-cell (beta_x, alpha_y, y0)."""

    def run(beta_x, alpha_y, y0):
        c = eqx.tree_at(
            lambda c: (
                c.processes["gz06"].parameters["beta_x"],
                c.processes["gz06"].parameters["alpha_y"],
            ),
            comp,
            (beta_x, alpha_y),
        )
        return sched.run(
            c, (0.0, t_end), macro_dt=t_end, save_dt=dt, y0=y0
        ).get("gz06/x")

    return jax.jit(jax.vmap(run))


def lognormal(key, mean, cv, n):
    if cv == 0.0:
        return jnp.full((n,), mean)
    s = np.sqrt(np.log(1.0 + cv**2))
    return mean * jnp.exp(s * jax.random.normal(key, (n,)) - s**2 / 2)


def measure(xs, tg, dt, t_end):
    """Per-cell period & amplitude, bulk coherence, pooled reporter."""
    mask = tg > 30.0
    periods, amps = [], []
    for x in xs:
        hi = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & mask[1:-1]
        lo = (x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]) & mask[1:-1]
        pk, tr = np.where(hi)[0] + 1, np.where(lo)[0] + 1
        if len(pk) > 1 and len(tr) > 0:
            periods.append(np.mean(np.diff(pk)) * dt)
            amps.append(x[pk].mean() - x[tr].mean())
    periods, amps = np.array(periods), np.array(amps)
    xbar = xs.mean(0)
    lo_i = tg > t_end - 30.0  # late window
    late = xbar[lo_i]
    pooled_amp = late.max() - late.min()
    tw = tg > t_end / 2
    rms_percell = float(np.sqrt((xs[:, tw] ** 2).mean(1)).mean())
    rms_pooled = float(np.sqrt((xbar[tw] ** 2).mean()))
    return {
        "period_cv": periods.std() / periods.mean(),
        "amp_cv": amps.std() / amps.mean(),
        "coherence": pooled_amp / amps.mean(),
        "rms_percell": rms_percell,
        "rms_pooled": rms_pooled,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=1000)
    ap.add_argument("--cv", type=float, default=0.30)
    ap.add_argument("--t-end", type=float, default=120.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    from hallsim.io import outdir

    ap.add_argument(
        "--out",
        default=str(outdir("gz06_population") / "gz06_population.png"),
    )
    a = ap.parse_args()

    comp, sched = build(), Scheduler()
    run = batched_runner(comp, sched, a.t_end, a.dt)
    tg = np.arange(0.0, a.t_end + a.dt / 2, a.dt)
    n, key = a.n_cells, jax.random.PRNGKey(a.seed)
    base_y0 = np.tile(np.asarray(comp.initial_state_vec()), (n, 1))

    # Random-phase initial conditions: sample the baseline limit cycle.
    cyc = np.asarray(
        sched.run(comp, (0.0, a.t_end), macro_dt=a.t_end, save_dt=a.dt).ys
    )
    on_cycle = cyc[tg > a.t_end - 30.0]
    kph, key = jax.random.split(key)
    idx = np.asarray(jax.random.randint(kph, (n,), 0, len(on_cycle)))
    phase_y0 = on_cycle[idx]

    conditions = {}
    k1, k2, k3, key = jax.random.split(key, 4)
    conditions["vary beta_x only\n(amplitude)"] = run(
        lognormal(k1, 0.9, a.cv, n), jnp.full((n,), 0.8), base_y0
    )
    conditions["vary alpha_y only\n(period)"] = run(
        jnp.full((n,), 0.9), lognormal(k2, 0.8, a.cv, n), base_y0
    )
    conditions["both + random phase\n(realistic bulk)"] = run(
        lognormal(k3, 0.9, a.cv, n),
        lognormal(jax.random.fold_in(k3, 1), 0.8, a.cv, n),
        phase_y0,
    )

    print(
        f"\nGZ06 population: {n} cells, per-cell CV={a.cv:.2f}, "
        f"t_end={a.t_end:g} h\n" + "=" * 68
    )
    stats = {}
    for name, xs in conditions.items():
        s = stats[name] = measure(np.asarray(xs), tg, a.dt, a.t_end)
        tag = name.replace("\n", " ")
        print(f"\n  {tag}")
        print(
            f"    amplitude CV = {s['amp_cv']*100:5.1f}%    "
            f"period CV = {s['period_cv']*100:5.1f}%"
        )
        print(
            f"    bulk coherence = {s['coherence']:.2f}  "
            f"(1=coherent, ->0=fully damped)"
        )
        print(
            f"    reporter: per-cell-RMS-then-avg = {s['rms_percell']:.3f}   "
            f"avg-then-RMS = {s['rms_pooled']:.3f}   "
            f"({s['rms_percell']/s['rms_pooled']:.2f}x)"
        )

    _plot(conditions, stats, tg, a.out)
    print(f"\nsaved figure -> {a.out}")


def _plot(conditions, stats, tg, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, (name, xs) in zip(axes, conditions.items()):
        xs = np.asarray(xs)
        for i in range(0, min(len(xs), 250), 25):
            ax.plot(tg, xs[i], color="0.78", lw=0.5)
        ax.plot(tg, xs.mean(0), color="C3", lw=2.3, label="bulk (pop. mean)")
        s = stats[name]
        ax.set_title(f"{name}\ncoherence {s['coherence']:.2f}", fontsize=10)
        ax.set_xlabel("time (h)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, 1.45)
    axes[0].set_ylabel("p53  (x)")
    fig.suptitle(
        "GZ06 lifted to a heterogeneous population: period spread "
        "(alpha_y) damps the bulk; amplitude spread (beta_x) does not",
        fontsize=12,
        y=1.03,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
