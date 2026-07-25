"""Does seeding Newton with a forward pre-solve land it on the physical root?

Unseeded, steady_state at ctrl converges to |dy/dt| ~ 1e-11 with several NF-kB
species negative -- a spurious algebraic branch. Its docstring prescribes a
short forward pre-solve as the seed. This compares the unseeded root, the
burn-in endpoint alone, and burn-in-seeded Newton, on residual and positivity,
for a few burn-in horizons.
"""

from __future__ import annotations

import logging

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


def main():
    ctrl = with_hallmarks(build_multi_hallmark_composite(), CTRL)
    keys = ctrl.store_keys()
    rhs, _ = ctrl.build_rhs()
    mask = np.asarray(accumulator_mask(ctrl, keys))
    ic = ctrl.initial_state_vec()
    laws = conservation_laws(ctrl, ic)
    sched = Scheduler()

    def report(name, y):
        y = np.asarray(y)
        r = np.abs(np.asarray(rhs(0.0, jnp.asarray(y))))[~mask]
        neg = np.flatnonzero(y < -1e-9)
        log(
            f"{name:28s} max|dy/dt|={r.max():.3e}  n_negative={len(neg)}  "
            f"min={y.min():.4e}"
        )
        if len(neg):
            log(f"    {[(keys[i], round(float(y[i]), 5)) for i in neg][:6]}")
        return y

    report("Newton (unseeded)", steady_state(ctrl, laws=laws))

    for horizon in (20.0, 60.0, 200.0):
        end = sched.run(
            ctrl, (0.0, horizon), macro_dt=5.0, save_dt=1.0, y0=ic
        ).ys[-1]
        report(f"burn-in {horizon:.0f} d", end)
        report(
            f"Newton seeded @ {horizon:.0f} d",
            steady_state(ctrl, y_guess=end, laws=laws),
        )


if __name__ == "__main__":
    main()
