# Lifting GZ06 to a population: why period spread damps the bulk

Reference note for the preprint's population run. The Geva-Zatorsky 2006
paper (BIOMD0000000157) is, at its core, about **cell-to-cell and
temporal variability**, not the existence of the oscillation. This note
records how HallSim lifts the deterministic oscillator to a heterogeneous
population, and the algebra behind a result that is easy to get wrong:
**a bulk (population-averaged) signal damps only if the oscillation
*period* varies across cells — spread in amplitude alone keeps the bulk
coherent.** The demo is `demos/gz06_population.py`.

## The model

With `alpha_x = 0` and `k = 1e-4` (so the Michaelis term `x/(x+k) ≈ 1`
whenever `x ≫ 1e-4`), the three species (p53 `x`, Mdm2 precursor `y0`,
Mdm2 `y`) obey a delayed negative-feedback loop:

```
dx/dt  = beta_x·psi          − alpha_k·y·x/(x+k)     (p53)
dy0/dt = beta_y·psi·x        − alpha_0·y0            (Mdm2 precursor)
dy/dt  = alpha_0·y0          − alpha_y·y             (Mdm2)
```

p53 makes Mdm2 precursor, which matures (`alpha_0`) into Mdm2, which
degrades p53. The maturation step is the delay that turns the negative
feedback into a limit cycle.

## Why beta_x sets amplitude and alpha_y sets the period

Near the operating point the degradation term linearises (`x/(x+k) ≈ 1`,
so p53 loss `≈ alpha_k·y`). The Jacobian of `(x, y0, y)` is then

```
      | 0          0        −alpha_k |
J  =  | beta_y·psi −alpha_0   0      |
      | 0           alpha_0  −alpha_y|
```

**`beta_x` does not appear in `J`.** It enters only as a constant forcing
term `beta_x·psi` in the `dx` equation. A constant forcing shifts the
fixed point and scales the limit-cycle amplitude, but it cannot change the
eigenvalues — so it changes neither the period nor the growth rate. That
is the exact reason `beta_x` is a pure amplitude knob.

The eigenvalues solve the characteristic cubic

```
lam³ + (alpha_0+alpha_y)·lam² + alpha_0·alpha_y·lam + alpha_k·alpha_0·beta_y·psi = 0
```

whose complex pair `a ± iω` gives period `2π/ω` and per-cell growth rate
`a`. Every coefficient contains `alpha_0, alpha_y, alpha_k, beta_y, psi` —
and **none contains `beta_x`**. At the published defaults the cubic gives
`ω = 0.928 rad/h → period 6.77 h`, matching the simulated 6.82 h.
Perturbing `alpha_y` by ±30% moves the period 6.66–6.95 h and flips the
growth rate sign; perturbing `beta_x` moves neither.

## Why a period spread damps the bulk

A bulk assay pools cells: the observed signal is `⟨x_i(t)⟩`. Write each
cell as `≈ A_i·cos(ω_i t + φ_i)`. The pooled amplitude is the coherence
(Kuramoto order parameter) of the phases. For a spread of frequencies with
std `σ_ω`, the phases fan out and coherence decays like
`exp(−½ σ_ω² t²)` — Gaussian dephasing.

- **Amplitude spread only** (`beta_x`): `σ_ω = 0`, so coherence stays ≈ 1.
  Averaging in-phase oscillators of different heights returns the
  mean-height oscillator — still fully oscillating.
- **Period spread** (`alpha_y`): `σ_ω > 0`, phases decohere, the bulk
  average damps toward its mean.

Measured over 1000 cells at 30% per-cell CV, `t_end = 120 h`:

| heterogeneity | amplitude CV | period CV | bulk coherence |
|---|---|---|---|
| `beta_x` only | ~30% | ~0% | **0.97** (no damping) |
| `alpha_y` only | ~33% | ~5% | **0.61** (damps) |
| both + random initial phase (realistic) | ~48% | ~6% | **0.09** (flat) |

## The reporter must be pooled per-cell, not on the pooled signal

DDB2 transcription is a per-cell nonlinear readout of p53 (RMS-like,
`gz06-basal-p53.md`); bulk RNA pools **mRNA**. So the correct bulk reporter
is **per-cell-RMS then average**, not RMS of the averaged p53. The two
diverge exactly when the population decoheres: identical (1.02×) while the
`beta_x` population stays coherent, but 1.24× once the realistic population
has damped. Applying the nonlinear reporter to the pooled-and-damped p53
would re-introduce the damage-blindness the RMS reporter was built to
avoid.

## References

- Geva-Zatorsky et al. 2006, *Mol Syst Biol* — GZ06, and the single-cell
  finding that amplitude is noisy (~70% CV) while period is robust
  (~20% CV); noise attributed to low-frequency fluctuations in protein
  production rates.
- `docs/gz06-basal-p53.md` — the RMS reporter and psi/basal mapping.
