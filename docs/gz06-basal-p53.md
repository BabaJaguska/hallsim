# Reading p53 damage from the Geva-Zatorsky 2006 oscillator

Reference note for the preprint: how the composite reads the p53-target
reporter (DDB2) off the Geva-Zatorsky 2006 (GZ06) p53–Mdm2 oscillator
(BIOMD0000000157), why the obvious readout (the mean) is wrong, and the
two model properties that force the design.

## GZ06 has no basal p53 — psi=0 collapses it

GZ06 models p53 (`x`) as production minus Mdm2-dependent degradation:

```
dx/dt = beta_x · psi  −  alpha_k · y · x/(x + k)     (alpha_x = 0)
        └ production      └ Mdm2-dependent degradation
```

Production is entirely proportional to the damage signal `psi`, and the
Mdm2-independent degradation `alpha_x` is zero. So at `psi = 0` there is no
p53 source and p53 decays to exactly zero. Geva-Zatorsky built the model
for damage-*induced* pulses, so an unstressed baseline was out of scope —
but a composite reading a p53-target reporter in a control arm needs one.
Basal p53 is low but nonzero in real cells, sustained by ~50 spontaneous
double-strand breaks per cell cycle (a *basal damage signal*, not a
damage-independent synthesis term) — Wang et al., *Modeling the Basal
Dynamics of P53 System*, PLoS ONE 2011
([PMC3218058](https://pmc.ncbi.nlm.nih.gov/articles/PMC3218058/)). So the
composite seeds `psi` at a nonzero basal level (see the mapping below).

## The deeper finding: the mean p53 is analytically damage-blind

Fixing the psi=0 collapse is not enough. Solve GZ06 for its steady state
(dx/dt = dy/dt = dy0/dt = 0). From the Mdm2 equations, `y = beta_y·x·psi /
alpha_y`; substituting into dx/dt = 0:

```
beta_x · psi = alpha_k · (beta_y·x·psi / alpha_y) · x/(x+k)
  ⟹  beta_x = alpha_k · beta_y · x² / (alpha_y · (x + k))
```

**psi cancels.** The steady-state (and hence long-window mean) p53 level is
fixed by the rate constants alone, independent of the damage signal — the
Mdm2 negative feedback buffers the mean against damage. Confirmed by
simulation: mean p53 = 0.385 at psi=0.05 and 0.394 at psi=1.0 (a 1.02×
fold across the whole range).

What psi *does* control is the **pulsing**. GZ06 undergoes a Hopf
bifurcation near psi ≈ 0.6: below it p53 sits at a stable point (amplitude
≈ 0), above it p53 oscillates (amplitude, std ≈ 0.33 at psi=1.0). This is
the textbook p53 behaviour — DNA damage is encoded in p53 **pulse
dynamics**, not in the mean level (Lahav et al. 2004, *Nat Genet*
[p53–Mdm2 pulses in single cells]; Purvis et al. 2012, *Science* [p53
dynamics control cell fate]).

Consequence: a window-**mean** readout of p53 (the natural
phase-insensitive summary for bulk transcriptomics) is exactly the summary
that *cannot* see damage in this model. The one summary you'd reach for is
the one that's blind.

## The reporter: RMS, not mean

DDB2 is read as the **trailing-window RMS** of p53, `√⟨x²⟩`, computed via a
`RunningIntegral(power=2)` that accumulates `∫x²` at the oscillation's own
resolution (`hallsim.gene_reporters.window_rms`). RMS:

- **rises with pulse amplitude** (so it reads the damage-encoding pulsing,
  unlike the mean), and
- **keeps the mean as a floor** — `√⟨x²⟩ = √(mean² + variance)` — so a
  quiescent baseline gives a finite fold-change instead of the divergence a
  bare amplitude/variance would produce below the Hopf point.

By Parseval's theorem the variance is the total non-DC spectral power, so
RMS is the differentiable form of "how much is p53 pulsing" (the
non-differentiable alternatives — zero-crossing count, peak-FFT frequency —
can't be used in a gradient-based fit).

Simulation vs data (etoposide DDIS vs control):

| p53 readout | model DDIS-vs-ctrl log2FC | vs measured DDB2 (+0.29) |
|---|---|---|
| mean | ~0.02 | far under (buffering) |
| amplitude (std) | ~10+ | far over (divergence) |
| **RMS √⟨x²⟩** | **+0.43** | close ✓ |

Biologically RMS is defensible: a p53 target integrates pulsatile p53
weighted toward peaks (⟨x²⟩ weights high-p53 excursions), which is how
pulse dynamics drive target-gene programs (Purvis 2012).

## psi_basal and its prior

The Genomic Instability hallmark interpolates `psi` from a basal level
(control) to the full-dose reference at DDIS (`hallsim.hallmarks`, Genomic
Instability → `gz06.parameters.psi`):

```
psi(severity) = psi_basal · (1 − severity)  +  GZ06_PSI_FULL · severity
```

- `severity = 0` (control): `psi = psi_basal`.
- `severity = 1` (DDIS): `psi = GZ06_PSI_FULL` (= 1.0, GZ06's calibrated
  full-irradiation reference).

**Prior on `psi_basal`:** a nonzero basal damage level from spontaneous
DSBs (PMC3218058); the exact value is only weakly informative because the
mean is buffered, and it becomes identifiable only through the RMS reporter
(it sets where control sits relative to the Hopf point, hence the pulse
amplitude). Seeded at 0.3 (`GZ06_PSI_BASAL_DEFAULT`), clamped (0.02, 0.95);
`GZ06_PSI_DEFAULT = 1.0` is retained for standalone screening at full dose.
GZ06 itself is unmodified — the baseline enters only through the hallmark
mapping.

## Honest limitations

- GZ06's Hopf is **sharp** (quiescent below psi≈0.6, full pulsing above),
  so the model predicts a near-switch p53 damage response, whereas the
  measured DDB2 change is gradual and modest (1.2×). Likely contributors:
  GZ06 is parameterised for *acute* pulsing while the data is D14 *chronic*
  senescence; and DDB2 mRNA integrates/saturates over pulses. This is a
  model-interrogation finding, recorded not patched.
- WT p53 *protein* induction on topo-II damage is a few-fold (doxorubicin
  ~4×; [PMC3702437](https://pmc.ncbi.nlm.nih.gov/articles/PMC3702437/);
  [Hdmx/p53, PNAS](https://www.pnas.org/doi/10.1073/pnas.0701497104)) — a
  *peak/pulse* quantity, not the population mean, consistent with the mean
  being buffered here.

## References

- Geva-Zatorsky et al. 2006, *Mol Syst Biol* — GZ06 p53–Mdm2 model
  (BIOMD0000000157).
- Lahav et al. 2004, *Nat Genet* — p53–Mdm2 pulses in individual cells.
- Purvis et al. 2012, *Science* — p53 dynamics control cell fate.
- Wang et al. 2011, PLoS ONE (PMC3218058) — basal p53 from spontaneous
  DSBs.
- p53 protein fold-induction: PMC3702437; PNAS 10.1073/pnas.0701497104.
