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

import json

# Config file (JSON, configs/ convention). Override the path as argv[1].
CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "configs/bench_population.json"
with open(CONFIG_PATH) as f:
    cfg = json.load(f)
MODEL = cfg["model"]
T_END = float(cfg["t_end"])
N_SAVE = int(cfg["n_save"])
CHUNK = int(cfg["chunk"])  # batch per GPU solve (memory cap)
NS = list(cfg["n_sweep"])
SIGMA = float(cfg.get("perturb_sigma", 0.1))
SEED = int(cfg.get("seed", 0))
NPROC = int(cfg.get("nproc", 8))
PLOT_PATH = cfg.get("plot", "demos/plots/bench_population.png")


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


def make_population(N, dim, seed=SEED):
    """N per-species perturbation factors: 1 + SIGMA * normal, shape (N, dim).

    Each tool gets its own matrix sized to its own species count (hallsim and
    roadrunner enumerate species differently); same distribution + seed, so the
    workload is equivalent even though the exact values differ.
    """
    rng = np.random.default_rng(seed)
    return 1.0 + SIGMA * rng.standard_normal((N, dim))


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

# ----- hallsim sweep (this process, GPU) -----
log(f"\n=== hallsim sweep: N = {NS} (CHUNK={CHUNK}) ===")
htimes = {}
for N in NS:
    fac_h = make_population(N, n_vars)
    hallsim_run(fac_h)  # compile this N's chunk shape(s) before timing
    t0 = time.time(); hallsim_run(fac_h); htimes[N] = time.time() - t0
    log(f"  N={N:>5}  hallsim={htimes[N]:7.3f}s")

# ----- tellurium sweep (separate JAX-free process: serial + NPROC-core) -----
# A subprocess (fork+exec) so its multiprocessing pool never inherits this
# process's CUDA context — fork-after-CUDA-init is unsafe.
import subprocess
log(f"\n=== tellurium sweep (subprocess, serial + {NPROC}-core) ===")
proc_out = subprocess.run(
    [sys.executable, "demos/_bench_tellurium.py", CONFIG_PATH],
    capture_output=True, text=True)
if proc_out.returncode != 0:
    log("tellurium subprocess FAILED:\n", proc_out.stderr[-2000:])
    sys.exit(1)
tel = json.loads(proc_out.stdout.strip().splitlines()[-1])

# ----- combine + report -----
rows = []
for N in NS:
    th = htimes[N]
    ts = tel[str(N)]["serial"]
    tp = tel[str(N)]["parallel"]
    rows.append((N, th, ts, tp))
    log(f"  N={N:>5}  hallsim(1 T4)={th:7.3f}s  tellurium(1)={ts:7.3f}s  "
        f"tellurium({NPROC})={tp:7.3f}s  | vs1core={ts/th:4.1f}x  "
        f"vs{NPROC}core={tp/th:4.1f}x")

# ----- plot -----
arr = np.array(rows)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.loglog(arr[:, 0], arr[:, 1], "o-", label="hallsim (1 Tesla T4, batched f64)")
ax1.loglog(arr[:, 0], arr[:, 2], "s-", label="tellurium (1 CPU core)")
ax1.loglog(arr[:, 0], arr[:, 3], "^-", label=f"tellurium ({NPROC} CPU cores)")
ax1.set_xlabel("population size N"); ax1.set_ylabel("wall time (s)")
ax1.set_title("Population simulation: wall time vs N"); ax1.legend(); ax1.grid(True, which="both", alpha=0.3)
ax2.semilogx(arr[:, 0], arr[:, 2] / arr[:, 1], "s-", label="vs 1 core")
ax2.semilogx(arr[:, 0], arr[:, 3] / arr[:, 1], "^-", label=f"vs {NPROC} cores")
ax2.axhline(1.0, ls="--", color="gray")
ax2.set_xlabel("population size N"); ax2.set_ylabel("hallsim speedup (tellurium / hallsim)")
ax2.set_title("hallsim speedup over tellurium"); ax2.legend(); ax2.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=120)
log(f"\nsaved {PLOT_PATH}")
log("DONE")
