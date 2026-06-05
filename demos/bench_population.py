"""Population benchmark: hallsim batched GPU solve vs tellurium serial loop.

Population = N individuals, same SBML model, heterogeneous initial conditions
(cell-to-cell variation). hallsim integrates the whole batch in one vmapped,
JIT-compiled float64 diffrax solve on GPU (chunked to fit T4 memory);
tellurium (libRoadRunner) loops one CVODE solve per individual on CPU.

float64 throughout (hallsim requires x64; CVODE is double-precision) so the
comparison is like-for-like on precision.

Usage:
    .venv/bin/python demos/bench_population.py            # default sweep
    .venv/bin/python demos/bench_population.py 1,8,64,256 # custom N sweep
"""
import jax
jax.config.update("jax_enable_x64", True)  # MUST precede any array creation

import sys
import time

import numpy as np
import jax.numpy as jnp
import equinox as eqx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hallsim.sbml_import import process_from_sbml
from hallsim.composite import Composite
from hallsim.scheduler import Scheduler

MODEL = "models/sivakumar2011/wnt_BIOMD0000000397.xml"
T_END = 20.0
N_SAVE = 21
CHUNK = 64  # max batch per GPU solve before T4 float64 OOM


def log(*a):
    print(*a, flush=True)


# ----- hallsim setup -----
proc = process_from_sbml(MODEL)
species = list(proc.ports_schema().keys())
topo = {"model": {p: f"pool/{p}" for p in species}}
comp = Composite(processes={"model": proc}, topology=topo)
sched = Scheduler()
base_y0 = comp.flatten(comp.initial_state())
keys = comp.store_keys()
n_vars = base_y0.shape[0]
log(f"hallsim: {len(species)} species, n_vars={n_vars}, "
    f"x64={jax.config.jax_enable_x64}, backend={jax.default_backend()}")

# Resolve the stiffness/solver verdict eagerly (concrete Jacobian) so the
# JIT-traced solve hits the warm cache. run() calls dfx.diffeqsolve directly,
# so XLA fuses the whole adaptive while-loop into one GPU kernel ONLY when we
# wrap it in jit — without this the ~thousands of steps dispatch op-by-op.
sched.warm_up(comp, (0.0, T_END), 1.0, y0=base_y0)


@eqx.filter_jit
def _solve(y0):
    return sched.run(comp, t_span=(0.0, T_END), macro_dt=1.0,
                     save_dt=T_END / (N_SAVE - 1), y0=y0).ys


def make_population(N, dim, seed=0):
    """N per-species perturbation factors: 1 + 0.1 * normal, shape (N, dim).

    Each tool gets its own matrix sized to its own species count (hallsim and
    roadrunner enumerate species differently); same distribution + seed, so the
    workload is equivalent even though the exact values differ.
    """
    rng = np.random.default_rng(seed)
    return 1.0 + 0.1 * rng.standard_normal((N, dim))


def hallsim_run(factors):
    """Integrate the whole population on GPU, chunked. Returns trajectory array."""
    N = factors.shape[0]
    y0 = jnp.asarray(base_y0)[None, :] * jnp.asarray(factors)
    outs = [_solve(y0[i:i + CHUNK]) for i in range(0, N, CHUNK)]
    ys = jnp.concatenate(outs, axis=1) if len(outs) > 1 else outs[0]
    ys.block_until_ready()
    return ys


# ----- tellurium setup -----
import tellurium as te
rr = te.loadSBMLModel(MODEL)
fs = rr.model.getFloatingSpeciesIds()


def tellurium_run(factors):
    """Serial loop: per individual, perturb init concs, CVODE simulate."""
    N = factors.shape[0]
    finals = np.empty((N, len(fs)))
    for i in range(N):
        rr.reset()
        for j, sp in enumerate(fs):
            k = f"[{sp}]"
            rr[k] = rr[k] * float(factors[i, j])
        res = rr.simulate(0, T_END, N_SAVE)
        finals[i] = res[-1, 1:1 + len(fs)]
    return finals


# ----- validation: nominal trajectory agreement -----
log("\n=== validation: nominal run, both tools ===")
rh = sched.run(comp, t_span=(0.0, T_END), macro_dt=1.0,
               save_dt=T_END / (N_SAVE - 1))
rr.reset()
rt = rr.simulate(0, T_END, N_SAVE)
# compare a few species shared by id
tel_ids = list(fs)
for sp in species[:6]:
    if sp in tel_ids:
        h = float(rh.get(f"pool/{sp}")[-1])
        t = float(rt[-1, 1 + tel_ids.index(sp)])
        rel = abs(h - t) / (abs(t) + 1e-12)
        log(f"  {sp}: hallsim={h:.4g}  tellurium={t:.4g}  rel_diff={rel:.2%}")

# ----- benchmark sweep -----
Ns = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 \
    else [1, 8, 32, 128, 512]
log(f"\n=== benchmark sweep: N = {Ns} (CHUNK={CHUNK}) ===")

rows = []
for N in Ns:
    fac_h = make_population(N, n_vars)
    fac_t = make_population(N, len(fs))
    hallsim_run(fac_h)  # compile this N's chunk shape(s) before timing
    t0 = time.time(); hallsim_run(fac_h); th = time.time() - t0
    t0 = time.time(); tellurium_run(fac_t); tt = time.time() - t0
    speedup = tt / th
    rows.append((N, th, tt, speedup))
    log(f"  N={N:>5}  hallsim={th:7.3f}s  tellurium={tt:7.3f}s  "
        f"speedup={speedup:5.1f}x")

# ----- plot -----
arr = np.array(rows)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.loglog(arr[:, 0], arr[:, 1], "o-", label="hallsim (GPU, batched f64)")
ax1.loglog(arr[:, 0], arr[:, 2], "s-", label="tellurium (CPU, loop)")
ax1.set_xlabel("population size N"); ax1.set_ylabel("wall time (s)")
ax1.set_title("Population simulation: wall time vs N"); ax1.legend(); ax1.grid(True, which="both", alpha=0.3)
ax2.semilogx(arr[:, 0], arr[:, 3], "d-", color="green")
ax2.axhline(1.0, ls="--", color="gray")
ax2.set_xlabel("population size N"); ax2.set_ylabel("speedup (tellurium / hallsim)")
ax2.set_title("hallsim speedup over tellurium"); ax2.grid(True, which="both", alpha=0.3)
plt.tight_layout()
out = "demos/plots/bench_population.png"
plt.savefig(out, dpi=120)
log(f"\nsaved {out}")
log("DONE")
