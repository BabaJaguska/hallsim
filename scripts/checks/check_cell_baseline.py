"""Why do heterogeneous cells trip the implicit solver when the nominal cell
does not?

The population builds each cell's start state as (its own ctrl fixed point) x
(a 10% lognormal jitter). That jitter moves the cell OFF the fixed point it
was just equilibrated to -- and moves its conserved totals -- so every cell
opens the run with a relaxation kick the nominal cell never sees.

The alternative: let the jitter set each cell's conserved totals, then solve
that cell's OWN fixed point for those totals (steady_state's y_ref pins the
conserved quantities). Every cell then starts AT its homeostasis.

Compares both constructions on 16 cells: residual at t=0, negativity, and
whether the 14-day DDIS solve reports a clean result.
"""

from __future__ import annotations

import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from hallsim.hallmarks import with_hallmarks  # noqa: E402
from hallsim.models.multi_hallmark import (  # noqa: E402
    build_multi_hallmark_composite,
)
from hallsim.scheduler import Scheduler  # noqa: E402
from hallsim.steady_state import (  # noqa: E402
    accumulator_mask,
    conservation_laws,
    steady_state,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("check").info

CTRL = {"Genomic Instability": 0.0, "Deregulated Nutrient Sensing": 0.5}
DDIS = {"Genomic Instability": 1.0, "Deregulated Nutrient Sensing": 0.5}
ALPHA_Y = ("gz06", "alpha_y", 0.8)
CDKN1A = ("dp14", "CDKN1A_transcr_by_FoxO3a_n_DNA_damage", 0.085)
N, PARAM_CV, IC_CV = 16, 0.30, 0.10


def cell(comp, ay, cd):
    return eqx.tree_at(
        lambda c: (
            c.processes[ALPHA_Y[0]].parameters[ALPHA_Y[1]],
            c.processes[CDKN1A[0]].parameters[CDKN1A[1]],
        ),
        comp,
        (ay, cd),
    )


def lognormal(key, mean, cv, shape):
    s = np.sqrt(np.log(1.0 + cv**2))
    return mean * jnp.exp(s * jax.random.normal(key, shape) - s**2 / 2)


def main():
    base = build_multi_hallmark_composite()
    ctrl, pert = with_hallmarks(base, CTRL), with_hallmarks(base, DDIS)
    keys = ctrl.store_keys()
    rhs, _ = ctrl.build_rhs()
    mask = np.asarray(accumulator_mask(ctrl, keys))
    ic = ctrl.initial_state_vec()
    laws = conservation_laws(ctrl, ic)
    sched = Scheduler(auto_stiffness=True, throw=False)
    seed = sched.run(
        ctrl, (0.0, 20.0), macro_dt=5.0, save_dt=20.0, y0=ic
    ).ys[-1]

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    ay = lognormal(k1, ALPHA_Y[2], PARAM_CV, (N,))
    cd = lognormal(k2, CDKN1A[2], PARAM_CV, (N,))
    jitter = lognormal(k3, 1.0, IC_CV, (N, len(keys)))

    def kicked(a, c, j):
        return steady_state(cell(ctrl, a, c), y_guess=seed, laws=laws) * j

    def own_fp(a, c, j):
        ref = seed * j
        return steady_state(cell(ctrl, a, c), y_guess=ref, laws=laws,
                            y_ref=ref)

    for name, fn in (("re-equilibrated", own_fp),):
        y0 = np.asarray(eqx.filter_jit(jax.vmap(fn))(ay, cd, jitter))
        r = np.abs(np.asarray(jax.vmap(rhs, in_axes=(None, 0))(0.0, y0)))
        log(f"\n=== {name} ===")
        log(f"  residual at t=0: max {r[:, ~mask].max():.3e}  "
            f"median-cell {np.median(r[:, ~mask].max(1)):.3e}")
        log(f"  worst state: {keys[int(np.argmax(r[:, ~mask].max(0)))]}  "
            f"negatives: {int((y0 < -1e-9).sum())} cells-states")

        ok = 0
        for i in range(3):
            res = sched.run(
                cell(pert, ay[i], cd[i]), (0.0, 14.0), macro_dt=5.0,
                save_dt=0.5, y0=jnp.asarray(y0[i]),
            )
            good = all(
                str(s["result"]) == "RESULTS.successful"
                for s in res.stats.values()
            )
            ok += good
            if not good and i < 3:
                log(f"    cell {i} failed: "
                    + "  ".join(f"{g}:{s['result']}"
                                for g, s in res.stats.items()))
        log(f"  clean 14-day solves: {ok}/3")


if __name__ == "__main__":
    main()
