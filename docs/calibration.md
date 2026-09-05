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

<!-- reporters:start — checked against MULTI_HALLMARK_REPORTERS by
     tests/unit/test_gene_reporters.py; edit the code, then this table. -->

| Gene | Store path | Summary | Note |
|---|---|---|---|
| `CDKN1A` (p21) | `dp14/CDKN1A` | zero-phase mean, τ=2.0 | senescence/arrest marker; DP14 transcribes it via FoxO3a × damage (no explicit p53) |
| `GLB1` (SA-β-gal) | `dp14/SA_beta_gal` | zero-phase mean, τ=2.0 | the canonical senescence marker, which DP14 models directly |
| `BNIP3` | `dp14/FoxO3a` | zero-phase mean, τ=2.0 | FoxO3 target; reads the FoxO-driven mitophagy arm downstream of nutrient sensing |
| `DDB2` | `gz06/x` | zero-phase **RMS** `√⟨x²⟩`, τ=0.75 | p53 target; GZ06's mean p53 is analytically damage-blind, so DDB2 reads pulse amplitude — see [gz06-basal-p53.md](gz06-basal-p53.md) |
| `MDM2` | `gz06/y0` | zero-phase mean, τ=0.75 | p53 target; `y0` is the paper's Mdm2 precursor, "representing, for example, Mdm2 mRNA" — the transcript, not the protein `y` |
| `FAS` (CD95/APO-1) | `k14/CD95_level` | zero-phase mean, τ=0.75 | p53 target via the FAS first-intron response element; the only transcript readout the Kallenberger arm admits — everything below the receptor is post-translational, and total Bid and total caspase-8 are conserved |

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

## Calibration API

[`hallsim.calibration`](../src/hallsim/calibration.py) provides `Calibrator`
(the low-level autodiff loop) plus a declarative layer —
`CalibrationProblem` + `Condition` + `ParameterRef` + `GeneExpressionDataset` —
for wiring any composite to any held-out gene-expression dataset:

```python
from hallsim.calibration import CalibrationProblem, Condition, ParameterRef
from hallsim.gene_reporters import GeneExpressionDataset, MULTI_HALLMARK_REPORTERS
from demos.models.multi_hallmark import build_multi_hallmark_composite

composite = build_multi_hallmark_composite()

# Self-documenting parameter discovery: walks every Process, enumerates each
# SBML constant / scalar attr, and hides hallmark-controlled knobs (they're
# set by Condition.hallmarks per arm, not learned from data).
for p in composite.calibration_targets():
    print(p.process_name, p.field, p.default, p.clamp)

ds = GeneExpressionDataset.from_series_matrix(
    series_matrix_path, platform_path, sample_position_groups=...)

problem = CalibrationProblem(
    composite=composite,
    reporters=MULTI_HALLMARK_REPORTERS,
    conditions={
        "ctrl": Condition("ctrl", {"Genomic Instability": 0.0,
                                   "Deregulated Nutrient Sensing": 0.5}),
        "DDIS": Condition("DDIS", {"Genomic Instability": 1.0,
                                   "Deregulated Nutrient Sensing": 1.0}),
        "RAPA": Condition("RAPA", {"Genomic Instability": 1.0,
                                   "Deregulated Nutrient Sensing": 0.3}),
    },
    arm_pairs={"DDIS_vs_ctrl": ("DDIS", "ctrl"),
               "RAPA_vs_ctrl": ("RAPA", "DDIS")},
    # Trajectory-native: each arm is a {day: Δlog2FC} time course (model time
    # units). A plain `ds.delta(...)` Series is the degenerate single-point
    # case, auto-normalized to {t_end: series}.
    # Every arm is normalized within itself, to its own day 0 — the rapamycin
    # culture's day 0 *is* ETOP_D00, since the drug goes in on day 2. The drug
    # contrast is recovered afterwards by differencing the two arm curves.
    data={
        "DDIS_vs_ctrl": {7.0: ds.delta("ETOP_D07", "ETOP_D00"),
                         14.0: ds.delta("ETOP_D14", "ETOP_D00")},
        "RAPA_vs_ctrl": {7.0: ds.delta("RAPA_D07", "ETOP_D00"),
                         14.0: ds.delta("RAPA_D14", "ETOP_D00")},
    },
    params={
        # No starting value: the fit begins at the composite's own value for
        # the field, so there is nowhere to declare one that could disagree.
        "CDKN1A_transcr": ParameterRef(
            "dp14", "parameters.CDKN1A_transcr_by_FoxO3a_n_DNA_damage",
            clamp=(0.001, 5.0), prior=0.085, prior_sigma=0.5),
        "alpha_y": ParameterRef(
            "gz06", "parameters.alpha_y", clamp=(0.01, 10.0),
            prior=0.8, prior_sigma=0.5),
    },
    fit_arms=["DDIS_vs_ctrl"],       # in the loss
    held_out_arms=["RAPA_vs_ctrl"],  # evaluated, not fit
)

history = problem.fit(steps=150, mode="reverse")
results = problem.evaluate(history.final_params)
```

### Principles the API enforces

- **Hallmark knobs aren't fittable by default** — a guard rail raises if you
  try to fit a pure severity dial (severity would overwrite the fit).
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
off = problem.with_overrides({"p53_cdkn1a.k_act": 0.0})
ablated = off.evaluate(params)
```

A key names either a fittable (whatever `params` calls it) or a process field
in dotted form — `"p53_cdkn1a.k_act"` and `"dp14.parameters.k"` address the same
places `ParameterRef` does. Both spellings reach the same field, so which list a
parameter happens to be in is not something you have to know. The call returns a
new problem and leaves the original alone; overrides compose.

An override is applied **last**, so it beats the fitted iterate and the
composite's own value alike. That is the whole reason this exists: every
evaluation substitutes the current iterate into each fitted field, so editing
one directly with `eqx.tree_at` is overwritten before the solve and the run
silently proceeds unablated — a live edge then measures as dead. That edit now
raises and points here.

Under `fit`, an override holds its parameter fixed and the optimizer sees a zero
gradient for it.

### Loss

MSE on **log2 fold-change**: the model emits `sign · (log2 cond − log2 base)`
per reporter, compared to the measured log2 fold-change. Because the data is
already a log ratio, the two are commensurable and every reporter contributes
its O(1) fold-change regardless of the observable's absolute scale (a 1e-4
pool and a 1e1 pool weigh equally — a plain mean or unit-norm loss lets the
big reporters dominate and makes small ones invisible).

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
unphysical rail. `ParameterRef.prior` / `prior_sigma` (log10) plus
`CalibrationProblem.prior_weight` add a log-normal MAP penalty
`Σ((log10 p − log10 prior)/σ)²` — anchoring each parameter to its
literature/derived value. Coupling-edge strengths, which have no direct
literature value, are anchored to their host-module scale; see
[coupling-edge-priors.md](coupling-edge-priors.md).

The penalty is a MAP regularizer, not a calibrated log prior — `prior_weight` is
a free multiplier and the data term is a mean, so the fit's *width* has no scale
even though its mode does. What that costs and how to fix it:
[uncertainty-quantification.md](uncertainty-quantification.md).

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
2. For each `Condition`, apply the hallmark severities, build the flat RHS
   `f(t, y; θ)` (`Composite.build_rhs`), and solve `dy/dt = f` over the full
   `t_span` with the Scheduler (one `diffeqsolve` per timescale group, under
   operator splitting / a single `lax.scan`).
3. Read the reporter store paths off the trajectory and apply each reporter's
   query-time-aware **summary** (`window_mean` / `window_rms`, interpolating
   the running-integrals at the measured days).
4. Form the model fold-change `sign·(log2 summ_cond − log2 summ_base)` and sum
   `(model − data)²` over `(reporter × timepoint)`, plus the MAP prior penalty.

Every step — including the solve — has a VJP, so `loss: θ ↦ ℝ` is
differentiable. The data is a handful of fold-change points per arm, not a
dense trajectory: the loss reads the model *at those query days* (via the
interpolating summaries) and compares fold-changes, not raw trajectories.

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
