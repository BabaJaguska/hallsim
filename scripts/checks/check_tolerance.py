"""Intake gate for the DDIS population run: does the implicit routing solve,
and is the trajectory tolerance-insensitive?

With auto_stiffness the two groups go to Kvaerno5. This runs the nominal cell
from the equilibrated ctrl baseline at the default tolerance and at 100x
tighter, and compares both the raw captured paths and the six canonical
reporter readouts. A material change between the two means the result is
solver-dependent and is not yet a result.
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from hallsim.gene_reporters import MULTI_HALLMARK_REPORTERS  # noqa: E402
from hallsim.hallmarks import with_hallmarks  # noqa: E402
from hallsim.models.multi_hallmark import (  # noqa: E402
    build_multi_hallmark_composite,
)
from hallsim.scheduler import Scheduler  # noqa: E402
from hallsim.steady_state import conservation_laws, steady_state  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("check").info

CTRL = {"Genomic Instability": 0.0, "Deregulated Nutrient Sensing": 0.5}
DDIS = {"Genomic Instability": 1.0, "Deregulated Nutrient Sensing": 0.5}
T_END, MACRO_DT, SAVE_DT, READOUT = 14.0, 5.0, 0.02, 14.0


def main():
    base = build_multi_hallmark_composite()
    ctrl, pert = with_hallmarks(base, CTRL), with_hallmarks(base, DDIS)
    keys = pert.store_keys()
    ic = ctrl.initial_state_vec()
    laws = conservation_laws(ctrl, ic)

    out = {}
    for label, rtol, atol in (
        ("default 1e-6/1e-9", 1e-6, 1e-9),
        ("tight   1e-8/1e-11", 1e-8, 1e-11),
    ):
        sched = Scheduler(auto_stiffness=True, rtol=rtol, atol=atol)
        seed = sched.run(
            ctrl, (0.0, 20.0), macro_dt=MACRO_DT, save_dt=20.0, y0=ic
        ).ys[-1]
        y0 = steady_state(ctrl, y_guess=seed, laws=laws)
        for name, integ in sched.warm_up(
            pert, (0.0, T_END), MACRO_DT, y0=y0
        ).items():
            log(f"[{label}] {name}: {type(integ.solver).__name__} "
                f"stiff={integ.stiff}")
        t0 = time.time()
        res = sched.run(
            pert, (0.0, T_END), macro_dt=MACRO_DT, save_dt=SAVE_DT, y0=y0
        )
        ys = np.asarray(res.ys)
        log(f"[{label}] {time.time() - t0:.1f}s  finite={np.isfinite(ys).all()}")
        for g, s in res.stats.items():
            log(f"    {g}: solver={s['solver']} stiff={s['stiff']} "
                f"steps={int(s['num_solver_steps'])} "
                f"rejected={int(s['num_rejected_steps'])} "
                f"result={s['result']}")
        out[label] = (np.asarray(res.ts), ys)

    (ts, a), (_, b) = out["default 1e-6/1e-9"], out["tight   1e-8/1e-11"]
    rel = np.abs(a - b) / np.maximum(np.abs(b).max(0), 1e-12)
    log(f"\nmax relative trajectory difference: {rel.max():.3e} "
        f"at {keys[int(np.argmax(rel.max(0)))]}")
    worst = np.argsort(rel.max(0))[::-1][:5]
    for i in worst:
        log(f"    {keys[i]:26s} rel {rel.max(0)[i]:.3e}")

    log("\nreporter readouts (default vs tight):")
    for r in MULTI_HALLMARK_REPORTERS:
        i = keys.index(r.observable)
        va = float(np.ravel(r.summary(jnp.asarray(ts), jnp.asarray(a[:, i]),
                                      READOUT))[0])
        vb = float(np.ravel(r.summary(jnp.asarray(ts), jnp.asarray(b[:, i]),
                                      READOUT))[0])
        log(f"    {r.gene_symbol:8s} {va:.6g}  vs  {vb:.6g}   "
            f"rel {abs(va - vb) / max(abs(vb), 1e-12):.2e}")


if __name__ == "__main__":
    main()
