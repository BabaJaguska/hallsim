"""Is the ctrl fixed point stable, or does NF-kB ride a limit cycle?

Two independent answers on the SAME physical (pre-solve-seeded) fixed point:

1. the spectrum of the FULL-system Jacobian -- not restricted to one
   timescale group, since freezing off-group variables can manufacture an
   unstable pair that the coupled system does not have;
2. a forward run started AT that fixed point, sampled finely enough to
   resolve a ~1.9 h NF-kB oscillation. An unstable focus with Re(lam)=+4.4/d
   e-folds in 0.23 d, so 5 days from the fixed point is ~21 e-folds -- any
   real instability is unmissable, and staying put is equally decisive.
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
from hallsim.io import outdir  # noqa: E402
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

    seed = sched.run(ctrl, (0.0, 20.0), macro_dt=5.0, save_dt=20.0, y0=ic).ys[-1]
    y_fp = steady_state(ctrl, y_guess=seed, laws=laws)
    log(f"fixed point: max|dy/dt| (non-accumulator) = "
        f"{np.abs(np.asarray(rhs(0.0, y_fp)))[~mask].max():.3e}")

    keep = np.flatnonzero(~mask)
    J = np.asarray(jax.jacfwd(lambda y: rhs(0.0, y))(y_fp))[np.ix_(keep, keep)]
    lam = np.linalg.eigvals(J)
    order = np.argsort(lam.real)[::-1]
    log("full-system spectrum, 6 least-stable eigenvalues (1/day):")
    for i in order[:6]:
        period = 2 * np.pi / abs(lam[i].imag) if lam[i].imag else np.inf
        log(f"    {lam[i].real:+.4e} {lam[i].imag:+.4e}i   "
            f"period {period * 24:.3f} h" if np.isfinite(period)
            else f"    {lam[i].real:+.4e} {lam[i].imag:+.4e}i   (no rotation)")
    unstable = lam[lam.real > 1e-8]
    log(f"eigenvalues with Re > 0: {len(unstable)}")
    for z in unstable:
        log(f"    UNSTABLE {z.real:+.4e} {z.imag:+.4e}i")

    res = sched.run(ctrl, (0.0, 5.0), macro_dt=5.0, save_dt=0.002, y0=y_fp)
    ys, ts = np.asarray(res.ys), np.asarray(res.ts)
    rel = (ys.max(0) - ys.min(0)) / np.maximum(np.abs(ys.mean(0)), 1e-12)
    log("\nforward 5 d starting AT the fixed point "
        "(save_dt=2.9 min, resolves a 1.9 h cycle):")
    for i in np.argsort(rel * ~mask)[::-1][:8]:
        log(f"    {keys[i]:26s} rel swing {rel[i]:.3e}  "
            f"start {ys[0, i]:.5g}  end {ys[-1, i]:.5g}")
    drift = np.abs(ys[-1] - ys[0]) / np.maximum(np.abs(ys[0]), 1e-12)
    log(f"max relative drift over 5 d (non-accumulator): "
        f"{drift[~mask].max():.3e} at {keys[int(np.argmax(drift * ~mask))]}")

    nf = [i for i, k in enumerate(keys) if k.startswith("nfkb/")]
    log(f"NF-kB max rel swing over 5 d: {rel[nf].max():.3e} "
        f"at {keys[nf[int(np.argmax(rel[nf]))]]}")
    np.savez(
        outdir("checks") / "ctrl_stability.npz",
        ts=ts, ys=ys, keys=np.array(keys), lam=lam, y_fp=np.asarray(y_fp),
    )


if __name__ == "__main__":
    main()
