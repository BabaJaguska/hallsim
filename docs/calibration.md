# Calibration & data validation

How HallSim fits mechanism parameters to data and validates mechanistic
states against transcriptomics — gene reporters, the calibration API,
held-out splits, priors, and what it takes to differentiate through a stiff
multi-model composite. The runnable end-to-end example is
[`demos/multi_hallmark_calibrate.py`](../demos/multi_hallmark_calibrate.py).

## Gene reporters

Mechanistic states are validated against transcriptomic data via
**single-gene reporters** ([`hallsim.gene_reporters`](../src/hallsim/gene_reporters.py)):
one canonical reporter gene per mechanistic store path, with a
literature-anchored sign and a per-reporter trajectory summary. The
multi-hallmark composite's reporters:

<!-- reporters:start — checked against MULTI_HALLMARK_REPORTERS + PROTEOSTASIS_REPORTERS by
     tests/unit/test_gene_reporters.py; edit the code, then this table. -->

| Gene | Store path | Summary | Note |
|---|---|---|---|
| `CDKN1A` (p21) | `dp14/CDKN1A` | zero-phase mean, τ=2.0 | senescence/arrest marker; DP14 transcribes it via FoxO3a × damage (no explicit p53) |
| `GLB1` (SA-β-gal) | `dp14/SA_beta_gal` | zero-phase mean, τ=2.0 | the canonical senescence marker, which DP14 models directly |
| `BNIP3` | `dp14/FoxO3a` | zero-phase mean, τ=2.0 | FoxO3 target; reads the FoxO-driven mitophagy arm downstream of nutrient sensing |
| `DDB2` | `gz06/x` | zero-phase **RMS** `√⟨x²⟩`, τ=0.75 | p53 target; GZ06's mean p53 is analytically damage-blind, so DDB2 reads pulse amplitude — see [Why the p53 reporters read RMS](#why-the-p53-reporters-read-rms) |
| `MDM2` | `gz06/y0` | zero-phase RMS amplitude, τ=0.75 | p53 target; `y0` is the paper's Mdm2 precursor, "representing, for example, Mdm2 mRNA" — the transcript, not the protein `y`. RMS as for DDB2: the mean of `y0` is damage-blind under GZ06, the pulse amplitude is not |
| `HSPA1A` | `p07/MisP` | zero-phase mean, τ=2.0 | HSP70, the canonical HSF1 target induced by misfolded load; reads Proctor's free misfolded pool — the load the heat-shock response would answer, not the response |

<!-- reporters:end -->

Oscillating species are read phase-insensitively, **post-hoc on the raw saved
trajectory**: a forward–backward exponential filter, so the summary carries no
phase lag and the composite gains no extra state (`zerophase_mean`,
`zerophase_rms_raw`). The save grid has to resolve the oscillation for this to
be faithful, which `Scheduler.run(antialias=True)` enforces.

The alternative is still available and is the right choice when a summary must
be read *inside* the solve rather than after it: `window_mean` / `window_rms`
over a co-solved [`RunningIntegral`](../src/hallsim/models/running_integral.py),
where `∫x` differenced over a trailing window gives the mean and `∫x²` gives
`√⟨x²⟩`. That buys grid-independence at the cost of extra integrated state and
a window's worth of lag.

### Why the p53 reporters read RMS

DDB2 and MDM2 read the Geva-Zatorsky 2006 (GZ06) p53–Mdm2 oscillator
(BIOMD0000000157), and two properties of that model decide the summary.

**Where damage enters.** Damage reaches GZ06 on `alpha_x`, the
Mdm2-independent p53 degradation that ATM blocks (Banin et al. 1998,
*Science*). Of the model's three degradation channels it is the one whose
damage direction crosses a supercritical Hopf point (at 0.1662), so lowering
it takes p53 from a fixed point into pulsing — the textbook damage response
(`demos/gz06_damage_channel_scan.py`). The paper's `psi` (ξ) is a
multiplicative noise gain on protein production, not a damage variable, and
the composite leaves it alone. GZ06 was built for damage-*induced* pulses and
has no unstressed baseline of its own; basal p53 in real cells is low but
nonzero, sustained by ~50 spontaneous double-strand breaks per cell cycle
(Wang et al. 2011, PLoS ONE, PMC3218058), and the coupling edge's resting
level supplies it.

**The mean p53 is analytically damage-blind.** At steady state the Mdm2
equations give `y = beta_y·x·psi / alpha_y`, and substituting into
`dx/dt = 0` the production gain cancels: the long-window mean p53 is fixed by
the rate constants alone, because the Mdm2 negative feedback buffers the mean
against the drive (simulated: mean p53 0.385 at psi = 0.05 and 0.394 at
psi = 1.0). What the drive controls is the **pulsing** — below the Hopf point
p53 sits at a fixed point, above it it oscillates — which is how DNA damage
is encoded in p53 (Lahav et al. 2004, *Nat Genet*; Purvis et al. 2012,
*Science*). A window-mean readout, the natural phase-insensitive summary for
bulk transcriptomics, is therefore exactly the summary that cannot see damage
in this model.

**RMS, not mean.** DDB2 is read as the zero-phase RMS of p53, `√⟨x²⟩`, and
MDM2 as the RMS of `y0`, the paper's Mdm2 precursor ("representing, for
example, Mdm2 mRNA" — the transcript, not the protein `y`). RMS rises with
pulse amplitude, so it reads the damage-encoding pulsing, and it keeps the
mean as a floor (`√⟨x²⟩ = √(mean² + variance)`), so a quiescent baseline
gives a finite fold-change instead of the divergence a bare amplitude would
produce below the Hopf point. By Parseval's theorem the variance is the total
non-DC spectral power, so RMS is the differentiable form of "how much is p53
pulsing"; zero-crossing counts and peak frequency are not differentiable and
cannot enter a gradient-based fit. Biologically, a p53 target integrates
pulsatile p53 weighted toward peaks, which is how pulse dynamics drive
target-gene programs (Purvis 2012). Out of the box, with the coupling edge
placed by `check_hill_gates` and no fitting, DDB2 reads +0.163 log2 at day 7
against a measured +0.202.

GZ06's Hopf is sharp, so the model predicts a near-switch damage response
where the measured DDB2 change is gradual (1.2×); GZ06 is parameterised for
acute pulsing while the data is day-14 chronic senescence, and DDB2 mRNA
integrates over pulses. WT p53 protein induction on topo-II damage is itself
a few-fold peak quantity (doxorubicin ~4×; PMC3702437), consistent with the
mean being buffered.

## Calibration API

[`hallsim.calibration`](../src/hallsim/calibration.py) provides `Calibrator`
(the low-level autodiff loop) plus a declarative layer —
`CalibrationProblem` + `Condition` + `FitParam` + `GeneExpressionDataset` —
for wiring any composite to any held-out gene-expression dataset:

```python
from hallsim.calibration import Arm, CalibrationProblem, Condition, FitParam
from hallsim.gene_reporters import GeneExpressionDataset, MULTI_HALLMARK_REPORTERS
from demos.models.hallmarks import HALLMARK_REGISTRY
from demos.models.multi_hallmark import build_multi_hallmark_composite
from demos.multi_hallmark_calibrate import (
    PLATFORM, SAMPLE_POSITION_GROUPS, SERIES_MATRIX, fetch_dataset)

composite = build_multi_hallmark_composite()

# Self-documenting parameter discovery: walks every Process, enumerates each
# SBML constant / scalar attr, and hides handle-controlled knobs (they're
# set by Condition.handles per arm, not learned from data).
for p in composite.calibration_targets():
    print(p.process_name, p.field, p.default, p.clamp)

fetch_dataset()   # GSE248823 from GEO into data/, once
ds = GeneExpressionDataset.from_series_matrix(
    SERIES_MATRIX, PLATFORM, sample_position_groups=SAMPLE_POSITION_GROUPS)

problem = CalibrationProblem(
    composite=composite,
    readouts=MULTI_HALLMARK_REPORTERS,
    conditions={
        "ctrl": Condition("ctrl", {"Genomic Instability": 0.0}),
        "DDIS": Condition("DDIS", {"Genomic Instability": 1.0}),
        # Rapamycin is a downward shift on nutrient sensing: -1 holds DP14's
        # mTORC1 phosphorylation rate below published from the dosing day on.
        "RAPA": Condition("RAPA", {"Genomic Instability": 1.0,
                                   "Deregulated Nutrient Sensing": -1.0}),
    },
    # Each arm reads against a reference: its own day 0 (the default), another
    # condition at the matched time, or None for values in the data's units.
    # A single arm needs no pair.
    arms={"DDIS_vs_ctrl": Arm("DDIS"), "RAPA_vs_ctrl": Arm("RAPA")},
    # Trajectory-native: each arm is a {day: Δlog2FC} time course (model time
    # units). A plain `ds.delta(...)` Series is the degenerate single-point
    # case, auto-normalized to {t_end: series}.
    # Every arm is normalized within itself, to its own day 0 — the rapamycin
    # culture's day 0 *is* ETOP_D00, since the drug goes in on day 2. The drug
    # contrast is recovered afterwards by differencing the two arm curves.
    data={
        "DDIS_vs_ctrl": {7.0: ds.delta("ETOPOSIDE_D07", "ETOPOSIDE_D00"),
                         14.0: ds.delta("ETOPOSIDE_D14", "ETOPOSIDE_D00")},
        "RAPA_vs_ctrl": {7.0: ds.delta("ETOPOSIDE_RAPA_D07", "ETOPOSIDE_D00"),
                         14.0: ds.delta("ETOPOSIDE_RAPA_D14", "ETOPOSIDE_D00")},
    },
    params={
        # No starting value: the fit begins at the composite's own value for
        # the field, so there is nowhere to declare one that could disagree.
        "CDKN1A_transcr": FitParam(
            "dp14", "parameters.CDKN1A_transcr_by_FoxO3a_n_DNA_damage",
            clamp=(0.001, 5.0), prior=0.085, prior_sigma=0.5),
        "alpha_y": FitParam(
            "gz06", "parameters.alpha_y", clamp=(0.01, 10.0),
            prior=0.8, prior_sigma=0.5),
    },
    fit_arms=["DDIS_vs_ctrl"],       # in the loss
    held_out_arms=["RAPA_vs_ctrl"],  # evaluated, not fit
    registry=HALLMARK_REGISTRY,      # what the condition names mean
)

history = problem.fit(steps=150, mode="reverse")
results = problem.evaluate(history.best_params)
```

### Fitting a parameter to a trajectory

The gene-expression layer above is one loss. `Calibrator` takes any
JAX-traceable scalar of the parameter pytree, with the solve inside it, so
the commonest fitting task — recover a rate constant from a time series —
needs no reporters, conditions or arms. Proctor 2007's synthesis rate from a
synthetic native-protein trajectory over one native hour, start 0.005, true
0.012:

```python
import jax.numpy as jnp
from hallsim.calibration import Calibrator
from hallsim.composite import single_process_composite
from hallsim.sbml_import import process_from_sbml
from hallsim.scheduler import Scheduler

base = single_process_composite(process_from_sbml(105), name="p07")
T_END = 3600.0

def natp(k1):
    comp = base.with_params({"p07.parameters.k1": k1})
    return Scheduler().run(comp, t_span=(0.0, T_END), macro_dt=T_END,
                           y0=comp.initial_state_vec(),
                           save_dt=T_END / 50).get("p07/NatP")

target = natp(0.012)

def loss(params):
    return jnp.mean(((natp(params["k1"]) - target) / target.mean()) ** 2)

hist = Calibrator(loss_fn=loss, init_params={"k1": jnp.asarray(0.005)},
                  log_params=True, clamps={"k1": (1e-4, 1e-1)},
                  mode="reverse", learning_rate=0.05).fit(steps=60)
hist.best_params["k1"]   # 0.01199 after 60 steps, ~16 s on a laptop CPU
```

`log_params=True` fits in log10 so a rate constant spanning decades takes
even steps; `clamps` is the box; `mode="reverse"` is the adjoint through the
solve. The block runs as written.

The same fit through the problem: an observed trajectory is a reporter set
that reads store paths as values, one condition, and one arm with no
reference, its data the paths' values at each sampled time. It then has
what the hand-written loss lacks: priors, held-out arms, best-iterate
tracking and the identifiability gate.

```python
import pandas as pd
from hallsim.gene_reporters import trajectory_readouts

ts = jnp.linspace(0.0, T_END, 51)
k1_problem = CalibrationProblem(
    composite=base,
    readouts=trajectory_readouts("p07/NatP"),
    conditions={"obs": Condition("obs", {})},
    data={"obs": {float(t): pd.Series({"p07/NatP": float(v)})
                  for t, v in zip(ts, target)}},
    arms={"obs": Arm("obs", reference=None)},
    params={"k1": FitParam("p07", "parameters.k1", clamp=(1e-4, 1e-1))},
    fit_arms=["obs"],
    t_end=T_END, macro_dt=T_END, n_save=51,
)
hist = k1_problem.fit(steps=60, mode="reverse", learning_rate=0.05)
hist.best_params["k1"]
```

A condition may also carry its own `start` state, `{path: value}` over the
shared start, and `window`, its own `(t_start, t_end)`, for an observation
that begins from a measured state partway through. A `start` value with a
leading axis runs one member per initial state through the Scheduler's batch
axis; a timepoint's data is then a `DataFrame` with one row per member.
`shooting_conditions(ts, ys, paths, segments=4)` builds exactly these from a
trajectory set, `(n_traj, n_t, len(paths))`: one batched condition per
window, consecutive windows sharing their boundary sample so a window's end
is fitted to the next window's start. Multiple shooting is then the same
loss over more arms, and a curriculum is the subset of arms passed to
`data_loss`. `Collocation(ts, ys, paths)` is the term without a solve: the
composite's field at each observed state against the trajectory's
central-difference slope, each path scaled by its slope's spread. Passed as
`collocation=` it enters `loss` at its `weight`; `collocation_loss` on its
own is a pretraining stage, cheap and free of phase drift, for a stiff or
oscillating field before the shooting fit. A learned block in the composite
is fitted the same way: `FitBlock("m")` in `params` makes the block's
trainable leaves one flat fittable, in linear space with no prior, outside
the identifiability report, and `fit` runs in reverse mode when one is
present. Mechanism constants and the block then descend one loss together.
The two NeuralODE trainers, `fit_neuralode_derivative` and
`fit_neuralode_shooting`, are this path with the block as the only
fittable: derivative matching is the collocation term alone, and shooting is
`shooting_conditions` over the trajectory set, one stage per curriculum
step, each warm-started from the previous stage's best iterate.
Minibatching is a PRNG key the `Calibrator` threads into the loss each
step (`minibatch_seed`): a `Collocation` draws `batch` samples from it and
a batched condition `member_batch` members. Iterates are then ranked on the
whole objective, or on `eval_loss_fn` where that is too costly, every
`eval_every` steps, so the best iterate is not the luckiest batch.

### Principles the API enforces

- **Handle targets aren't fittable by default** — a guard rail raises if you
  try to fit a parameter a handle's severity drives (severity would overwrite
  the fit).
- **One route for changing a parameter** — `with_overrides`, fitted or not. An
  edit that substitution would overwrite raises instead of running unablated.
- **Parameter discovery is self-documenting** via `calibration_targets()`.
- **Held-out splits are mandatory** — calibrate on one arm, report concordance
  on another. Same-data fit-and-evaluate is curve-fitting, not concordance.
- **Hill gates are checked where they are declared** — see below.

### Hill gate placement

A coupling edge gated on `xⁿ/(Kⁿ+xⁿ)` is silent when misplaced: a `K` above
everything its driver reaches never opens, one below the floor is always open,
and both read downstream as a weak coupling rather than as an error.
`CalibrationProblem.__init__` therefore runs `check_hill_gates()` — it compares
every gate `Composite.hill_gates()` finds against the range its driver reaches,
warns when `K` is outside, and reports the replacement:

```
Hill gate psi_bridge.K = 52 on 'dp14/DNA_damage' is above everything its driver
reaches ([1, 27.18] across conditions), so the edge is dead. K=10.79 would open
it, but needs n=19 (>8); the driver's low and high levels differ by only
r=1.26, too little for a Hill gate to resolve at any plausible cooperativity
```

`n` is the required cooperativity, and it is a pure function of the separation
ratio `r` between the driver's low and high levels:
`n = ln((1−p)/p) / (½·ln r)` for an off-occupancy `p` (default 0.1). It reads as
a difficulty score — real cooperativity tops out near 4, and `n_max` is 8, so a
large `n` says the driver does not separate its own regimes and no gate on it
will switch.

The check needs a trajectory, hence a horizon, which is why it lives on the
calibration problem rather than on `Composite` — the problem has a horizon
because its data was measured at particular times. It costs one condition-set
solve, and `hill_gates()` is structural, so a composite declaring no gate pays
nothing.

`suggest_hill_gate(params, path, off_conditions, on_conditions)` is the same
placement with the two levels named explicitly, for when you want the contrast
to be a specific pair of conditions rather than the driver's own extremes.

### Changing a parameter for a run — ablations

`with_overrides` is the one route, and it works the same whether or not the
parameter is fitted:

```python
# Is the p53 -> CDKN1A edge load-bearing? Switch it off and re-score.
off = problem.with_overrides({"p53_cdkn1a.hi": 0.0})
ablated = off.evaluate(history.best_params)
```

A key names either a fittable (whatever `params` calls it) or a process field
in dotted form — `"p53_cdkn1a.hi"` and `"dp14.parameters.k"` address the same
places `FitParam` does. Both spellings reach the same field, so which list a
parameter happens to be in is not something you have to know. The call returns a
new problem and leaves the original alone; overrides compose.

An override is applied **last**, so it beats the fitted iterate and the
composite's own value alike. Every evaluation substitutes the current iterate
into each fitted field, which is why a direct `eqx.tree_at` edit of a fitted
field raises and points here: the override is the one route that survives
the substitution.

Under `fit`, an override holds its parameter fixed and the optimizer sees a zero
gradient for it.

### Loss

MSE on each arm's readout against its data. With a reference the model emits
`sign · (log2 cond − log2 ref)` per reporter, compared to the measured log2
fold-change: the two are commensurable and every reporter contributes its
O(1) fold-change regardless of the observable's absolute scale (a 1e-4 pool
and a 1e1 pool weigh equally — a plain mean or unit-norm loss lets the big
reporters dominate and makes small ones invisible). An arm with no reference
compares the summary itself, in the data's units through the reporter's
`scale`, which is what a simulator-generated trajectory or a state measured
in the model's units needs.

**Trajectory, not endpoint.** The loss fits the fold-change *time course*: it
sums one MSE term per `(arm, timepoint)`, reading each reporter at every
measured timepoint. Each condition is solved once over the full `t_span`;
the reporter summaries are query-time-aware (`summary(ts, y, query_times)`)
and read the trajectory at the requested times — grid-independently (the
running-integral windows behind `window_mean` / `window_rms` are
interpolated at the query times, not snapped to the save grid). The
timepoint axis is **vectorized**, not looped, so the traced graph is
`O(reporters)` regardless of how many timepoints there are — 2 or 200 cost
the same to compile, and the number of ODE solves never changes (one per
condition). An arm with a single timepoint is the degenerate endpoint case.
`evaluate` correspondingly returns `{arm: {timepoint: ConcordanceResult}}`.

### Priors (MAP regularization)

With few data points a fit is under-constrained and a parameter can run to an
unphysical rail. `FitParam.prior` / `prior_sigma` (log10) plus
`CalibrationProblem.prior_weight` add a log-normal MAP penalty
`Σ((log10 p − log10 prior)/σ)²` — anchoring each parameter to its
literature/derived value. Coupling-edge strengths, which have no direct
literature value, are anchored to their host-module scale (next section).

The penalty is a MAP regularizer: it sets the mode. Reporting a *width* (a
band on a prediction) needs the loss as a log density;
[roadmap.md](roadmap.md#uncertainty-options-and-costs) sets out the options
and their costs.

### Priors for coupling-edge strengths

A `HillActivationEdge` contributes `d(target)/dt += k_act · H(signal; K, n)`
with `H ∈ [0, 1]` a Hill gate, so `k_act` is the maximum rate at which the
edge can move the target pool: a phenomenological rate in the composite's
units. The literature establishes an edge's *existence and direction*
(mTORC1 → IKK: Dan et al. 2008, *Genes Dev*; Laberge et al. 2015, *Nat Cell
Biol*; ROS-activated IKKβ → NF-κB: Karin & Ben-Neriah 2000); its strength in
the composite is set by the pool it writes into.

**Anchor to the host module.** For an edge driving a pool whose initial
concentration is 0.1 and whose intrinsic turnover is of order 10⁻⁴ per native
time unit, the natural scale for `k_act` is the pool itself (~0.1): the edge
should modulate the pool on the order of its own size without dominating the
intrinsic dynamics. That is the weakly-informative prior — `FitParam(prior=
<host scale>, prior_sigma=0.5)` in log10 decades, the clamp as a hard
backstop — and with few reporters per fitted parameter it is what keeps an
edge off its clamp. A model that publishes rate constants for the same
cascade in its own units (Konrath 2023, `MODEL2307130001`, molecule counts
with IKK ≈ 10⁵) corroborates that the pathway is quantifiable, but its
constants do not transfer to a normalised pool without a scaling that would
itself be a guess.

**Placement is checked.** The gate has to sit inside its driver's realised
range, which `check_hill_gates` verifies at construction (above). The
strength prior and the placement check together make a phenomenological edge
a bounded object rather than a free knob. These are order-of-magnitude
priors; `prior_weight`, the data-versus-prior trade-off, is worth a
sensitivity check rather than one canonical value.

### Optimizer

`method="adam"` (default) or `method="lbfgs"`. **Adam (LR ~0.03 +
`reduce_on_plateau` + early stopping) is the right choice for HallSim's
expensive stiff-ODE loss** — one solve per step. L-BFGS converges in far
fewer *steps* but its line search does many solves per step, so it only wins
when the loss is cheap. Early stopping returns the *best* params seen, not the
last.

## Differentiating through a stiff multi-model composite

The composite is calibrated by gradient descent **through the ODE solve** —
the same reverse-mode autodiff that trains neural networks, applied to
mechanism parameters spread across independently-published SBML models.

### Forward — the loss is one pure function of the parameters

`CalibrationProblem.loss(θ)` composes, all in JAX:

1. **Substitute** the fit parameters `θ` into the composite's process pytree
   via `eqx.tree_at` (`_substitute`), so each lands *inside* its model as a
   traced array — the step that makes it reachable by autodiff.
2. For each `Condition`, apply the handle severities, build the flat RHS
   `f(t, y; θ)` (`Composite.build_rhs`), and solve `dy/dt = f` over the full
   `t_span` with the Scheduler (one `diffeqsolve` per timescale group, under
   operator splitting / a single `lax.scan`).
3. Read the reporter store paths off the trajectory and apply each reporter's
   query-time-aware **summary** (`window_mean` / `window_rms`, interpolating
   the running-integrals at the measured days).
4. Form the model fold-change `sign·(log2 summ_cond − log2 summ_base)` and sum
   `(model − data)²` over `(reporter × timepoint)`, plus the MAP prior penalty.

Every step — including the solve — has a VJP, so `loss: θ ↦ ℝ` is
differentiable. The loss reads the model *at the measured times* (via the
interpolating summaries), three points per arm or three hundred, and
compares each arm's readout in its own terms.

### Backward — reverse-mode through the solve

`jax.grad(loss)` propagates the chain rule back through the arithmetic, the
`log2`, and the interpolating summaries — all trivial — down to the one hard
factor, `∂y(t)/∂θ` through the solver. The solve is a sequence of solver steps
`yₙ₊₁ = step(yₙ; θ)`, each a JAX function; because `f` depends on `θ` at
*every* step, the parameter gradient **accumulates over the whole trajectory**.

Storing every intermediate state for that backward pass is `O(n_steps)`
memory — untenable for a stiff multi-day solve of thousands of steps. HallSim
uses Diffrax's **`RecursiveCheckpointAdjoint`** (the `mode="reverse"` /
`adjoint=None` path): recursive (Griewank–Walther) checkpointing keeps only
`O(log n_steps)` checkpoints and re-runs forward segments during the backward
pass to rebuild the rest — trading ~`log n` extra forward evaluations for
`log n` memory, and yielding the *exact* gradient of the numerical solution
actually taken. The continuous adjoint (an augmented ODE solved backward,
`O(1)` memory) is **not** used: on stiff p53 / NF-κB oscillators the
reconstructed backward solution drifts and corrupts the gradient.

The `lax.scan` outer solve keeps this bounded at scale — the whole multi-group
run compiles to one executable, so reverse-mode checkpoint memory no longer
grows with the macro-step count.

### The loop

`Calibrator` builds `vg = jax.jit(jax.value_and_grad(loss))` **once** and
reuses it every step; un-jitted it would re-trace the composite + solve +
adjoint each iteration (the difference between minutes and hours). Each step is
one `value_and_grad` (≈ one solve's cost) fed to the optimizer.

### What it takes on stiff systems

- **A stiff solver with a proper Newton root finder.** Stiff subsystems make
  an explicit solver's *sensitivity* (the gradient) blow up to NaN even when
  the forward trajectory is finite. The Scheduler auto-detects stiffness per
  timescale group and routes stiff groups to an A-stable implicit solver
  (`Kvaerno5` + Newton); Diffrax differentiates through the Newton root-find
  too (an implicit-function-theorem VJP). The Scheduler routes by default,
  and `CalibrationProblem` warms it on concrete parameters before
  differentiating — routing needs a concrete Jacobian, so a cold trace would
  otherwise fall back to the explicit solver (loudly) and hand back NaN
  sensitivities on exactly the stiff groups that matter.
- **End-to-end float64.** Adaptive error control at `rtol=1e-6` needs the
  state *and* the RHS in double precision; a single hardcoded `float32`
  silently caps precision and makes the implicit solver reject most steps.

### What the gradient reaches

`∂loss/∂θ` is nonzero for SBML rate constants *inside* the imported models (via
the `c`-vector substitution in `SBMLProcess.derivative`), the coupling-edge
strengths, and the driven-parameter basal `psi_basal`. Because hallmark
severity is itself a differentiable transform on parameters, `jax.grad(loss)`
w.r.t. a **severity** works too — the whole path from a gene-reporter readout,
back through the operator-split checkpointed solve, to any upstream knob is one
differentiable graph.

## The demo dataset — GSE248823

The dataset the multi-hallmark demo is calibrated and evaluated against.
Loaded by
[`demos/multi_hallmark_calibrate.py`](../demos/multi_hallmark_calibrate.py)
from `data/FibroblastsDNA_dmg_Rapamycin/`, which `simulate demo multi-hallmark
run` fills from GEO on first use (`fetch-data` does only that). The fetch
writes each file's SHA-256 to `SHA256SUMS` beside it, the loader checks
them, and the run's `config.json` records them.

**Source.** [GSE248823](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE248823):
Tighanimine et al. 2024, *Nature Metabolism* 6:323–342, "A homoeostatic
switch causing glycerol-3-phosphate and phosphoethanolamine accumulation
triggers senescence by rewiring lipid metabolism" (DOI
10.1038/s42255-023-00972-y). Platform GPL17586, Affymetrix Human
Transcriptome Array 2.0 — a microarray, so values are normalized log2
intensities and the calibration-comparable quantity is a fold-change (a
same-gene ratio cancels the probe-specific scale). WI-38 human fibroblasts;
20 arrays = 10 conditions × 2 biological replicates.

**Arms.** Timepoints are in days; the two senescence triggers run on
different clocks.

| Arm | Trigger | Timepoints | Arrays |
|-----|---------|-----------|--------|
| Etoposide (DDIS) | DNA-damage-induced senescence | D00, D07, D14 | 6 |
| Etoposide + rapamycin | + mTOR inhibitor | D07, D14 | 4 |
| RAS (OIS) | oncogene (HRAS)-induced senescence | D00, D04, D07 | 6 |
| RAS + DMOG | + DMOG (prolyl-hydroxylase inhibitor) | D04, D07 | 4 |

**Sample → column mapping.** Series-matrix column indices (0-based, after
the ID column), matching `SAMPLE_POSITION_GROUPS` in the demo; each group is
the two biological replicates for one condition.

| Columns | Sample title | Arm · timepoint |
|---------|--------------|-----------------|
| 0, 1   | `WI38_…_ETOPOSIDE_D00`            | Etoposide · D00 |
| 2, 3   | `WI38_…_ETOPOSIDE_D07`            | Etoposide · D07 |
| 4, 5   | `WI38_…_ETOPOSIDE_D14`            | Etoposide · D14 |
| 6, 7   | `WI38_…_ETOPOSIDE_RAPAMYCIN_D07`  | Etoposide+rapa · D07 |
| 8, 9   | `WI38_…_ETOPOSIDE_RAPAMYCIN_D14`  | Etoposide+rapa · D14 |
| 10, 11 | `WI38_…_RAS_D00`                  | RAS · D00 |
| 12, 13 | `WI38_…_RAS_D04`                  | RAS · D04 |
| 14, 15 | `WI38_…_RAS_D07`                  | RAS · D07 |
| 16, 17 | `WI38_…_RAS_DMOG_D04`             | RAS+DMOG · D04 *(unused)* |
| 18, 19 | `WI38_…_RAS_DMOG_D07`             | RAS+DMOG · D07 *(unused)* |

**What the demo uses.**

| Composite arm | Definition (condition vs reference) | Role |
|---------------|-------------------------------------|------|
| `DDIS_vs_ctrl` | etoposide D07, D14 **vs** etoposide D00 | **fit** (the only arm in the loss) |
| `RAPA_vs_ctrl` | etoposide+rapa D07, D14 **vs** etoposide D00 | held-out (rapamycin effect) |
| `RAS_vs_ctrl` | RAS D04, D07 **vs** RAS D00 | held-out (transfer to a different trigger) |

The RAS + DMOG arm is not used: DMOG is a metabolic perturbation outside
the composite's scope. Every arm is normalised within itself, to its own
day 0 (`normalization="baseline"`), so the model reproduces `X_t / X_0`
along each arm; the rapamycin culture's day 0 *is* etoposide D00, since
rapamycin is not added until day 2, and the drug contrast is recovered by
differencing the two within-arm curves. Replicates are averaged (mean of
log2 intensities) into each condition before the fold-change, so every
reporter contributes one measured Δ per timepoint. The reporters are the
table at the top of this page.

**Design notes.** No untreated culture is measured at D07 or D14, so each
arm's fold-change reads the trajectory from its own start rather than a
contrast against an untreated culture at the same day. Datasets that add
that contrast, for a follow-up: GSE63577 + GSE77682 (Marthandan; MRC-5 and
HFF, young vs 20 Gy at 120 h — DallePezze 2014's own cell line and dose),
GSE63577 (MRC-5 PD32 vs PD72, replicative senescence), and GSE222400 (WI-38,
doxorubicin, eight timepoints with an untreated arm; its per-sample files
are differential tables whose reference is not the one the filename
suggests, so establish the contrast structure before use). Two biological
replicates and 2–3 timepoints per arm constrain rather than fully resolve
the dynamics; the dataset was chosen for its accessibility, its topical
alignment, and its two-arm ±intervention design.

**Treatment protocol, as the series matrix records it.** DDIS was
triggered by etoposide at 20 µM for two days; cells were then washed and
given fresh medium without drug. For the rapamycin samples, rapamycin was
added to the fresh medium at 20 nM, and the growth protocol changes the
medium every two days, so rapamycin is present continuously from the
washout at day 2 to harvest at day 7 and 14. The composite matches that
shape: the Deregulated Nutrient Sensing handle is a step at day 2 held to
the end of the run (`StepSource`, mTORC1 S2448 phosphorylation at half the
published rate), not a pulse.

**Deposit.** The calibrated composite is in BioModels as MODEL2609140001
(https://identifiers.org/biomodels.db/MODEL2609140001), submitted
2026-09-14: the etoposide arm as the main file, the control and
etoposide-plus-rapamycin arms and the README as additional files. Private
until the paper is out; reviewer access on request from the entry's toolbox.
