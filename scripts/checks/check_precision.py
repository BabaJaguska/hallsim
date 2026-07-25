"""Is the population run actually float64 on the GPU, and what does that cost?

Confirms x64 survives all the way to the solved trajectory (not just the
config flag), then measures this card's f64 vs f32 throughput so the sweep
timings can be read against the hardware's real double-precision ceiling.
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from hallsim.hallmarks import with_hallmarks  # noqa: E402
from hallsim.models.multi_hallmark import (  # noqa: E402
    build_multi_hallmark_composite,
)
from hallsim.scheduler import Scheduler  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("check").info

CTRL = {"Genomic Instability": 0.0, "Deregulated Nutrient Sensing": 0.5}


def throughput(dtype, n=2048, reps=20):
    a = jnp.asarray(np.random.rand(n, n), dtype=dtype)
    f = jax.jit(lambda x: x @ x)
    jax.block_until_ready(f(a))
    t0 = time.time()
    for _ in range(reps):
        out = f(a)
    jax.block_until_ready(out)
    dt = (time.time() - t0) / reps
    return 2 * n**3 / dt / 1e12, out.dtype  # TFLOP/s


def main():
    d = jax.devices()[0]
    log(f"device: {d} kind={d.device_kind}  x64={jax.config.jax_enable_x64}")

    comp = with_hallmarks(build_multi_hallmark_composite(), CTRL)
    y0 = comp.initial_state_vec()
    res = Scheduler(auto_stiffness=True).run(
        comp, (0.0, 1.0), macro_dt=1.0, save_dt=0.02, y0=y0
    )
    log(f"initial_state_vec: {y0.dtype}   ts: {res.ts.dtype}   "
        f"ys: {res.ys.dtype}   device={list(res.ys.devices())}")
    eps = np.finfo(np.asarray(res.ys).dtype).eps
    log(f"trajectory eps = {eps:.3e} "
        f"({'float64' if eps < 1e-15 else 'NOT float64'})")

    for dt in (jnp.float64, jnp.float32):
        tf, out_dtype = throughput(dt)
        log(f"{np.dtype(dt).name}: {tf:.3f} TFLOP/s  (result {out_dtype})")


if __name__ == "__main__":
    main()
