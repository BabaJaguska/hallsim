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

| Gene | Store path | Summary | Note |
|---|---|---|---|
| `CDKN1A` (p21) | `dp14/CDKN1A` | endpoint | senescence/arrest marker; DP14 transcribes it via FoxO3a × damage (no explicit p53) |
| `DDB2` | `gz06/x2_integral` | **RMS** `√⟨x²⟩` | p53 target; GZ06's mean p53 is analytically damage-blind, so DDB2 reads pulse amplitude — see [gz06-basal-p53.md](gz06-basal-p53.md) |
| `HMOX1` | `dp14/ROS` | endpoint | Nrf2/ARE oxidative-stress reporter |
| `NFKBIA` (IκBα) | `nfkb/IkBat` | window-mean | IκBα *transcript* (an NF-κB target that rises with activity), not the protein |
| `CYCS` (cyt c) | `dp14/Mito_mass_new` | endpoint | mitochondrial biogenesis |
| `EIF4EBP1` (4E-BP1) | `dp14/mTORC1_pS2448` | endpoint | kinase-level mTOR proxy |

Oscillating species are read phase-insensitively via a
[`RunningIntegral`](../src/hallsim/models/running_integral.py) co-solved at
the oscillation's own resolution: `∫x` differenced over a trailing window
gives the mean (`window_mean`); `∫x²` gives `√⟨x²⟩` (`window_rms`, for
pulse-amplitude readouts like DDB2).

## Calibration API

[`hallsim.calibration`](../src/hallsim/calibration.py) provides `Calibrator`
(the low-level autodiff loop) plus a declarative layer —
`CalibrationProblem` + `Condition` + `ParameterRef` + `GeneExpressionDataset` —
for wiring any composite to any held-out gene-expression dataset:

```python
from hallsim.calibration import CalibrationProblem, Condition, ParameterRef
from hallsim.gene_reporters import GeneExpressionDataset, MULTI_HALLMARK_REPORTERS
from hallsim.models.multi_hallmark import build_multi_hallmark_composite

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
               "RAPA_vs_DDIS": ("RAPA", "DDIS")},
    data={"DDIS_vs_ctrl": ds.delta(...), "RAPA_vs_DDIS": ds.delta(...)},
    params={
        "CDKN1A_transcr": ParameterRef(
            "dp14", "parameters.CDKN1A_transcr_by_FoxO3a_n_DNA_damage",
            init=0.085, clamp=(0.001, 5.0), prior=0.085, prior_sigma=0.5),
        "alpha_y": ParameterRef(
            "gz06", "parameters.alpha_y", init=0.8, clamp=(0.01, 10.0),
            prior=0.8, prior_sigma=0.5),
    },
    fit_arms=["DDIS_vs_ctrl"],       # in the loss
    held_out_arms=["RAPA_vs_DDIS"],  # evaluated, not fit
)

history = problem.fit(steps=150, mode="reverse")
results = problem.evaluate(history.final_params)
```

### Principles the API enforces

- **Hallmark knobs aren't fittable by default** — a guard rail raises if you
  try to fit a pure severity dial (severity would overwrite the fit).
- **Parameter discovery is self-documenting** via `calibration_targets()`.
- **Held-out splits are mandatory** — calibrate on one arm, report concordance
  on another. Same-data fit-and-evaluate is curve-fitting, not concordance.

### Loss

MSE on **log2 fold-change**: the model emits `sign · (log2 cond − log2 base)`
per reporter, compared to the measured log2 fold-change. Because the data is
already a log ratio, the two are commensurable and every reporter contributes
its O(1) fold-change regardless of the observable's absolute scale (a 1e-4
pool and a 1e1 pool weigh equally — a plain mean or unit-norm loss lets the
big reporters dominate and makes small ones invisible).

### Priors (MAP regularization)

With few data points a fit is under-constrained and a parameter can run to an
unphysical rail. `ParameterRef.prior` / `prior_sigma` (log10) plus
`CalibrationProblem.prior_weight` add a log-normal MAP penalty
`Σ((log10 p − log10 prior)/σ)²` — anchoring each parameter to its
literature/derived value. Coupling-edge strengths, which have no direct
literature value, are anchored to their host-module scale; see
[coupling-edge-priors.md](coupling-edge-priors.md).

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
mechanism parameters spread across independently-published SBML models. Two
ingredients make this work on stiff systems:

- **A stiff solver with a proper Newton root finder.** Stiff subsystems make
  an explicit solver's *sensitivity* (the gradient) blow up even when the
  forward trajectory is fine. The Scheduler auto-detects stiffness per
  timescale group and routes stiff groups to an A-stable implicit solver
  (`Kvaerno5`); non-stiff groups stay on a cheap explicit solver.
  `CalibrationProblem` enables this routing by default.
- **End-to-end float64.** Adaptive error control at `rtol=1e-6` needs the
  state *and* the RHS in double precision; a single hardcoded `float32`
  silently caps precision and makes the implicit solver reject most steps.
