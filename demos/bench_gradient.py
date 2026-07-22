"""Gradient benchmark: HallSim autodiff vs tellurium finite differences.

A population-level scalar loss L(theta) = mean over the cell population of a
readout species at t_end. We need dL/dtheta over P SBML rate constants theta.

- HallSim: one reverse-mode `jax.grad` returns the exact gradient w.r.t. all P
  parameters in a single backward pass through the batched-population solve —
  cost ~independent of P.
- tellurium / libRoadRunner: no native parameter gradients, so the realistic
  route is central finite differences — 2*P population solves, and only
  approximate.

Reports: correctness (autodiff vs hallsim's own central differences) and
wall time vs P (flat vs linear). float64 throughout. Run from repo root with
the hallsim venv.

    .venv/bin/python demos/bench_gradient.py [config.json]
"""

import jax

jax.config.update("jax_enable_x64", True)

import json
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

CONFIG_PATH = (
    sys.argv[1] if len(sys.argv) > 1 else "configs/bench_population.json"
)
with open(CONFIG_PATH) as f:
    cfg = json.load(f)
MODEL = cfg["model"]
T_END = float(cfg["t_end"])
N_SAVE = int(cfg["n_save"])
N_POP = 128
P_SWEEP = [1, 4, 8, 16]
H_FD = 1e-4  # relative step for central differences


def log(*a):
    print(*a, flush=True)


# ----- hallsim setup -----
proc = process_from_sbml(MODEL)
species = list(proc.ports_schema().keys())
topo = {"model": {p: f"pool/{p}" for p in species}}
comp = Composite(processes={"model": proc}, topology=topo)
sched = Scheduler()
base_y0 = comp.flatten(comp.initial_state())
sched.warm_up(comp, (0.0, T_END), 1.0, y0=base_y0)
SAVE_DT = T_END / (N_SAVE - 1)

# ----- tellurium setup -----
import tellurium as te

rr = te.loadSBMLModel(MODEL)
rr_params = set(rr.model.getGlobalParameterIds())
rr_species = list(rr.model.getFloatingSpeciesIds())

# Parameters both tools can set: SBML global constants present in roadrunner.
PNAMES = [n for n in proc._param_names if n in rr_params][: max(P_SWEEP)]
assert len(PNAMES) >= max(
    P_SWEEP
), f"only {len(PNAMES)} shared settable params"
THETA0 = np.array([float(proc.parameters[n]) for n in PNAMES])
READOUT = species[5]
readout_i = species.index(READOUT)
readout_t = rr_species.index(READOUT)
log(f"model={proc._name}  P params={PNAMES}\nreadout={READOUT}  N_pop={N_POP}")


def population(dim, seed=0):
    rng = np.random.default_rng(seed)
    return 1.0 + 0.1 * rng.standard_normal((N_POP, dim))


# ----- hallsim gradient -----
def hallsim_loss_factory(pnames, yb):
    def loss(theta):
        proc_i = eqx.tree_at(
            lambda p: [p.parameters[n] for n in pnames],
            proc,
            [theta[i] for i in range(len(pnames))],
        )
        comp_i = Composite(
            processes={"model": proc_i},
            topology=topo,
            validate=False,
            semantic_validation=False,
        )
        ys = sched.run(
            comp_i, t_span=(0.0, T_END), macro_dt=1.0, save_dt=SAVE_DT, y0=yb
        ).ys
        return jnp.mean(ys[-1, :, readout_i])

    return loss


# ----- tellurium finite-difference gradient -----
def tellurium_loss(theta, pnames, factors):
    for n, v in zip(pnames, theta):
        rr[n] = float(v)
    vals = np.empty(len(factors))
    for c in range(len(factors)):
        rr.reset()
        for j, sp in enumerate(rr_species):
            k = f"[{sp}]"
            rr[k] = rr[k] * float(factors[c, j])
        res = rr.simulate(0, T_END, N_SAVE)
        vals[c] = res[-1, 1 + readout_t]
    return float(vals.mean())


def tellurium_fd_grad(theta, pnames, factors):
    g = np.empty(len(theta))
    for i in range(len(theta)):
        h = H_FD * (abs(theta[i]) + H_FD)
        tp = theta.copy()
        tp[i] += h
        tm = theta.copy()
        tm[i] -= h
        g[i] = (
            tellurium_loss(tp, pnames, factors)
            - tellurium_loss(tm, pnames, factors)
        ) / (2 * h)
    return g


# ----- correctness: hallsim autodiff vs hallsim's OWN finite difference -----
# Cross-tool numerical agreement is not meaningful here: sbmltoodejax and
# roadrunner number reaction rate constants differently, so a given name moves
# different physical constants in each tool (trajectories still match at the
# nominal defaults). The rigorous check that the autodiff is exact is therefore
# against hallsim's own central differences.
log(
    "\n=== autodiff correctness: hallsim autodiff vs hallsim finite-diff (P=6) ==="
)
pn = PNAMES[:6]
th = THETA0[:6]
loss1 = hallsim_loss_factory(pn, jnp.asarray(base_y0[None, :]))
g_auto = np.asarray(eqx.filter_jit(jax.grad(loss1))(jnp.asarray(th)))
fwd1 = eqx.filter_jit(loss1)
g_hfd = np.empty(len(pn))
for i in range(len(pn)):
    h = 1e-4 * (abs(th[i]) + 1e-4)
    tp = th.copy()
    tp[i] += h
    tm = th.copy()
    tm[i] -= h
    g_hfd[i] = (
        float(fwd1(jnp.asarray(tp))) - float(fwd1(jnp.asarray(tm)))
    ) / (2 * h)
for n, a, b in zip(pn, g_auto, g_hfd):
    log(f"  {n:>10}: autodiff={a:+.4e}  hallsim_FD={b:+.4e}")
cos = float(
    np.dot(g_auto, g_hfd)
    / (np.linalg.norm(g_auto) * np.linalg.norm(g_hfd) + 1e-30)
)
log(f"  cosine similarity = {cos:.6f}  (1.0 => autodiff is exact)")

# ----- timing vs number of parameters P -----
log(f"\n=== gradient wall time vs P (N_pop={N_POP}) ===")
yb = jnp.asarray(base_y0[None, :] * population(base_y0.shape[0]))
fac_t = population(len(rr_species))
rows = []
for P in P_SWEEP:
    pn = PNAMES[:P]
    th = jnp.asarray(THETA0[:P])
    vg = eqx.filter_jit(jax.value_and_grad(hallsim_loss_factory(pn, yb)))
    vg(th)[1].block_until_ready()  # compile
    t0 = time.time()
    vg(th)[1].block_until_ready()
    t_had = time.time() - t0
    t0 = time.time()
    tellurium_fd_grad(np.asarray(THETA0[:P]), pn, fac_t)
    t_fd = time.time() - t0
    rows.append((P, t_had, t_fd))
    log(
        f"  P={P:>3}  hallsim_autodiff={t_had:7.4f}s  tellurium_FD={t_fd:7.3f}s  "
        f"({t_fd/t_had:5.1f}x)"
    )

# ----- plot -----
arr = np.array(rows)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(arr[:, 0], arr[:, 1], "o-", label="HallSim autodiff (reverse-mode)")
ax.plot(arr[:, 0], arr[:, 2], "s-", label="tellurium finite differences")
ax.set_xlabel("number of parameters P")
ax.set_ylabel("gradient wall time (s)")
ax.set_title(f"dL/dθ over a {N_POP}-cell population: autodiff vs finite diff")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
from hallsim.io import outdir

out = str(outdir("bench_gradient") / "bench_gradient.png")
plt.savefig(out, dpi=120)
log(f"\nsaved {out}\nDONE")
