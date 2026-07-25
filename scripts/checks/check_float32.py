"""Can the FORWARD population solve run in float32 on this composite?

Gradients are not the question here -- the calibration keeps float64 either
way. The question is whether a forward-only run survives single precision,
given the analyzer measures cond=9.7e18 / inf on the two groups that now go
through an implicit 54x54 solve per stage.

Run twice; the second pass compares against the first:
    python check_float32.py f64
    python check_float32.py f32
Both start from the SAME float64-computed equilibrated baseline (cast down
for the f32 pass), so this isolates the forward solve from the Newton
equilibration, and both are compared on the six canonical reporters.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import jax

MODE = sys.argv[1] if len(sys.argv) > 1 else "f64"

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from hallsim.gene_reporters import MULTI_HALLMARK_REPORTERS  # noqa: E402
from hallsim.hallmarks import with_hallmarks  # noqa: E402
from hallsim.models.multi_hallmark import (  # noqa: E402
    build_multi_hallmark_composite,
)
from hallsim.io import outdir  # noqa: E402
from hallsim.scheduler import Scheduler  # noqa: E402
from hallsim.steady_state import conservation_laws, steady_state  # noqa: E402

# hallsim/__init__.py force-enables x64 at import, so the override has to come
# after it -- the composite and every solve array are built later.
jax.config.update("jax_enable_x64", MODE == "f64")

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("check").info

OUT = outdir("checks")
CTRL = {"Genomic Instability": 0.0, "Deregulated Nutrient Sensing": 0.5}
DDIS = {"Genomic Instability": 1.0, "Deregulated Nutrient Sensing": 0.5}
T_END, MACRO_DT, SAVE_DT, READOUT = 14.0, 5.0, 0.02, 14.0
# f32 cannot service atol=1e-9 on states of order 1e3, so the f32 pass also
# gets the loosest tolerance the oscillators tolerate per the intake notes.
TOLS = {"f64": [(1e-6, 1e-9)], "f32": [(1e-6, 1e-9), (1e-5, 1e-7)]}


def main():
    base = build_multi_hallmark_composite()
    ctrl, pert = with_hallmarks(base, CTRL), with_hallmarks(base, DDIS)
    keys = pert.store_keys()
    log(f"mode={MODE}  default float dtype={jnp.zeros(1).dtype}")

    y0_path = OUT / "f32_baseline_y0.npy"
    if MODE == "f64":
        ic = ctrl.initial_state_vec()
        laws = conservation_laws(ctrl, ic)
        seed = Scheduler(auto_stiffness=True).run(
            ctrl, (0.0, 20.0), macro_dt=MACRO_DT, save_dt=20.0, y0=ic
        ).ys[-1]
        y0 = np.asarray(steady_state(ctrl, y_guess=seed, laws=laws))
        np.save(y0_path, y0)
    else:
        y0 = np.load(y0_path)
    log(f"baseline y0 range [{y0.min():.4g}, {y0.max():.4g}]")

    for rtol, atol in TOLS[MODE]:
        sched = Scheduler(auto_stiffness=True, rtol=rtol, atol=atol,
                          throw=False)
        sched.warm_up(pert, (0.0, T_END), MACRO_DT, y0=jnp.asarray(y0))
        t0 = time.time()
        res = sched.run(
            pert, (0.0, T_END), macro_dt=MACRO_DT, save_dt=SAVE_DT,
            y0=jnp.asarray(y0),
        )
        ys, ts = np.asarray(res.ys), np.asarray(res.ts)
        tag = f"{MODE} rtol={rtol:g}"
        log(f"[{tag}] {time.time() - t0:.1f}s  dtype={ys.dtype}  "
            f"finite={np.isfinite(ys).all()}")
        for g, s in res.stats.items():
            log(f"    {g}: steps={int(s['num_solver_steps'])} "
                f"rejected={int(s['num_rejected_steps'])} "
                f"result={s['result']}")
        vals = {}
        for r in MULTI_HALLMARK_REPORTERS:
            i = keys.index(r.observable)
            vals[r.gene_symbol] = float(
                np.ravel(r.summary(jnp.asarray(ts), jnp.asarray(ys[:, i]),
                                   READOUT))[0]
            )
        np.savez(OUT / f"prec_{MODE}_{rtol:g}.npz", ts=ts, ys=ys,
                 genes=np.array(list(vals)), vals=np.array(list(vals.values())))
        log(f"    reporters: " + "  ".join(f"{k}={v:.6g}"
                                           for k, v in vals.items()))

    if MODE == "f32":
        ref = np.load(OUT / "prec_f64_1e-06.npz")
        for rtol, _ in TOLS["f32"]:
            cur = np.load(OUT / f"prec_f32_{rtol:g}.npz")
            a, b = cur["ys"].astype(np.float64), ref["ys"]
            n = min(len(a), len(b))
            rel = np.abs(a[:n] - b[:n]) / np.maximum(np.abs(b[:n]).max(0), 1e-12)
            log(f"\nf32 rtol={rtol:g} vs f64: max rel trajectory diff "
                f"{rel.max():.3e} at {keys[int(np.argmax(rel.max(0)))]}")
            for g, v, vr in zip(cur["genes"], cur["vals"], ref["vals"]):
                log(f"    {str(g):8s} {v:.6g} vs {vr:.6g}   "
                    f"rel {abs(v - vr) / max(abs(vr), 1e-12):.2e}")


if __name__ == "__main__":
    main()
