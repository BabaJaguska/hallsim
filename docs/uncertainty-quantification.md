# Uncertainty quantification: the options, what each costs, what each buys

P3.9 in [known-problems.md](known-problems.md) says outputs are lines and a lab
needs bands. This note ranks the ways to get bands against the *measured* cost
of a HallSim gradient, and says what each option buys that the one above it
does not.

It was prompted by "would Hamiltonian MCMC beat Optax for fitting". It would
not, and the reason is worth writing down: an optimizer and a sampler answer
different questions, and the sampler's question costs three orders of magnitude
more on this composite.

## Optax and HMC do not compete

`Calibrator` returns a MAP point estimate. A sampler returns a distribution.
Nothing an optimizer does is done better by a sampler — HMC spends its budget
characterising tails a point estimate discards. The only thing sampling buys is
the uncertainty, so the comparison to make is not "HMC vs Adam" but "which
uncertainty method, at what cost".

## The prerequisite for every option: the loss is not a log density

Three separate scalings sit between
[`CalibrationProblem.loss`](../src/hallsim/calibration.py) and a
negative log posterior:

- [`gaussian_nll`](../src/hallsim/calibration.py) returns `Σ w·e² / n_finite` —
  a *mean*, and without the ½.
- [`data_loss`](../src/hallsim/calibration.py) then takes the mean over arms.
- [`_prior_penalty`](../src/hallsim/calibration.py) is `prior_weight · Σ z²`,
  also without the ½, and `prior_weight` defaults to `1.0` as a free multiplier.

For a point estimate this is harmless up to one thing it does control: the
data/prior balance, which is exactly `prior_weight`'s job. For **any** method
that reports a width it is fatal — every one of them reads the curvature of the
loss *and its scale*, so bands come out wrong by an unknown, non-uniform factor.

The target is `−log p = (1/2σ²)·Σ w·e² + ½·Σ((θ−θ₀)/σ_prior)² + const`, with θ
in log10 (which is where the priors and the sensitivities already live). Two
routes to the residual scale σ:

- **Real precision weights.** `weights` already accepts per-entry precision
  (replicate precision, or a DESeq2/edgeR moderated SE). When they are real,
  `Σ w·e²` *is* the χ², σ ≡ 1, and only the `1/n`, the ½, and the arm mean are
  wrong. Weights default to unity today, so this path is unused.
- **Estimate σ̂ at the MAP** from the residuals, `σ̂² = RSS/(n − p)`. Costs
  nothing beyond the fit that already ran, and is the defensible default when
  no precision is supplied.

This change is confined to two functions and it gates everything below.

## What already exists

[`identifiability.py`](../src/hallsim/identifiability.py) builds
`∂preds/∂log10 θ` and `cov = pinv(JᵀJ)`. That is the Laplace covariance with two
pieces missing, and both are worth naming as findings rather than as work items:

1. **The noise scale is set to 1.** `std_decades` is a 1σ *in units of a
   unit-variance residual*; the real one is `σ̂ ×` the reported number. A log2
   fold-change residual around 0.3 means the current numbers overstate the
   spread by roughly 3×, so the `std_tol=1.0` threshold files parameters as
   `practical` more often than the data warrants.
2. **The prior precision is absent from the Fisher matrix.** A parameter with an
   operative prior is constrained even when no reporter moves with it. The MAP
   posterior precision is `JᵀJ/σ̂² + diag(1/σ_prior²)` — both terms already in
   log10 space, so they add with no change of variables — and it is nonsingular
   whenever every parameter carries a prior, which also removes the reliance on
   `pinv`'s `rcond` cutoff.

So bands are tens of lines from what is already here, not a project.

## Option 1 — Laplace + delta method (do this first)

`Σ_θ = (JᵀJ/σ̂² + Π_prior)⁻¹`, and the band on any predicted quantity is
`sqrt(diag(J_pred · Σ_θ · J_predᵀ))` — including arms and timepoints that were
not in the fit.

**Cost.** One Jacobian at the MAP: `(1 + n_params)` forward solves, about one
fit step. On the multi-hallmark demo that is ~41 s (7 params, 4 arms; see the
2026-08-03 diary entry). Nothing else.

**Gain.**
- P3.9 closes for the common case. Every reporter trajectory gets a band, and a
  held-out claim becomes "predicted −0.465 ± σ against measured −0.476" rather
  than a bare point that cannot be judged.
- It ranks which parameter dominates the band on a given prediction — i.e. which
  measurement to make next.
- It reuses the Jacobian the identifiability screen already builds, so verdicts
  and bands come from one object and cannot disagree with each other.

**Limits.** Local and Gaussian: it reports an ellipse where the truth is a
banana, and it is least trustworthy exactly where the Fisher condition number is
already large — which is where this composite lives. It needs Option 2 as its
standing check.

**Deliverable.** `hallsim.uncertainty` with `laplace_covariance(problem, params,
sigma)` and a `predict_with_band(...)` that the existing calibration figure
draws.

## Option 2 — Profile likelihood (Raue et al. 2009)

Fix θᵢ on a grid, re-optimize everything else, and read the interval off the χ²
profile. The boundary follows the curved valley instead of an ellipse.

**Cost.** `n_params × n_grid` restricted fits, each warm-started from its
neighbour, so ~10–30 steps rather than 150. At 7 params × 15 grid points × 20
steps × ~41 s that is roughly a day sequential. The grid is embarrassingly
parallel, and the natural shape here is one batched fit over the grid — the
framework is batch-native. Unverified: whether the equilibration Newton solve
and the concrete-only stability check survive `vmap` over the parameter axis.

**Gain over Option 1.**
- It is the systems-biology standard for this exact problem, and the lineage the
  DallePezze 2014 reduction already comes from — a result reported with profiles
  is comparable to the field's.
- It separates structural (flat to the horizon), practical (open on one side),
  and identifiable using the *nonlinear* boundary. That is the case where
  `MAX_FIT_CONDITION_NUMBER = 1e12` currently refuses the fit outright; a
  profile turns the refusal into "this combination is determined, and here is how
  far the rest can run".
- It shows a parameter pinned against a clamp rail (the `CDKN1A_transcr` floor in
  the 2026-05-28 entry) as an open profile. An ellipse cannot express that.

## Option 3 — HMC / NUTS

**The arithmetic, from measured numbers.** One reverse gradient on the
multi-hallmark demo is 25.9 s; a full fit step is 41.1 s forward / 46.1 s
reverse (2026-08-03). NUTS needs ~30–100 gradients per draw, so 1000 warmup +
1000 draws is ~10⁵ gradients ≈ **40 days per chain**. Chains `vmap`; the compute
does not. Even a 40× faster gradient leaves about a day of warmup alone. Against
a 1.7 h fit, that is a factor of several hundred.

**Blockers a faster gradient would not remove.**
- The density scale above — a prerequisite for every option, but only here does
  it decide the answer rather than the error bar on it.
- `clamps` are enforced by clipping after each Optax step. A sampler needs a
  bijection instead; the log10 reparameterization is half of one.
- `_require_stable_baseline` returns early on a tracer, so under sampling nothing
  stops a chain wandering into θ where the equilibration Newton solve lands on an
  unstable fixed point or fails to converge. That is silent garbage, not a
  rejected proposal. A sampler needs `steady_state` to return a convergence flag
  and the density to go to `−inf` there.
- Adaptive step control makes `log p` piecewise-smooth in θ; the sampler's step
  size adapts down to compensate and the gradient count goes up.

**Where it is worth doing anyway.** On a single constituent, where a gradient is
milliseconds — DP14 alone against its deposited fitting data, which
`intake.published_fit_chi2` already reproduces to 0.08% of the published χ².
That is the constituents-first rule applied to uncertainty, and it is the only
way to learn how wrong the Laplace ellipse is on a real HallSim posterior before
trusting it on a composite. blackjax is pure JAX and operates on Equinox
pytrees, so the integration is ~20 lines against `-loss` and needs no
architectural change; it is not a dependency today.

**Gain if it ever becomes affordable.** The full posterior: multimodality,
curvature honest to the nonlinearity, correlations without a Gaussian
assumption, and posterior-predictive bands that need no delta method. Also a
principled way to *report* an unidentifiable fit instead of refusing it.

## Option 4 — resampling

Multistart ensembles (the COPASI-style screen the identifiability module's
docstring already cites) and an arm- or reporter-level bootstrap answer a
different question: how far the fit moves when the *data* moves. Per unit of
insight it is the most expensive option on a composite — cost is `n_restarts`
full fits — but it is the only one short of a sampler that finds a second basin.

## Recommended order

1. **Make the loss a log density** and estimate σ̂ at the MAP. Two functions.
   Nothing below means anything without it.
2. **Laplace bands** off the existing Jacobian, plus the prior-precision term in
   `report_from_jacobian`. Closes P3.9 for the common case at the price of one
   Jacobian.
3. **Profile likelihood** for whatever step 2 calls wide, and as the standing
   check on the ellipse. Also the route to fitting above
   `MAX_FIT_CONDITION_NUMBER` rather than refusing.
4. **NUTS on one constituent**, to measure how wrong step 2 is. Not on a
   composite until a gradient costs about a second.
