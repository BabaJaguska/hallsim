# HallSim Roadmap

Planned and queued work, grouped by area. See
[crossgen-suggestions.md](crossgen-suggestions.md) for the cross-domain
analysis behind the Scheduler items.

## Scheduler & Multi-Scale

* [ ] Combine Strang splitting + interpolated coupling (currently mutually exclusive)
* [ ] Event-bearing and adaptive_dt composites under batched `y0` (currently
  rejected at `Scheduler.run` entry — both rely on Python-side branching that
  doesn't compose with `vmap`)
* [ ] Waveform relaxation (Gauss-Seidel iteration at sync points, from FSI/PLL analogy)
* [ ] Anderson acceleration for waveform relaxation convergence
* [ ] Mori-Zwanzig memory kernel for fast→slow coupling (captures history effects)
* [ ] Coupling residual spectral monitoring (early-warning diagnostic)
* [ ] IFT-based adjoint at sync boundaries (for gradient-based optimization)
* [ ] IMEX (implicit-explicit) solver for stiff multi-scale systems

## Models & Validation

* [ ] **Lipid-metabolism extension** — Tighanimine et al. 2024 (*Nat Metab*, the paper behind GSE248823) identified a G3P/PEtn homeostatic switch as *causal* for senescence (p53 → glycerol kinase activation drives G3P↑; PCYT2 post-translational inactivation drives PEtn↑; lipid droplet biogenesis is the downstream effect). Adding a `LipidMetabolism` Process (states: G3P, PEtn; inputs: `p53_activity`, a PCYT2-PTM proxy; outputs: a senescence-amplifying signal that feeds back into the SASP axis) would let HallSim test their causal claim *in silico* — and the GSE248824 SuperSeries includes the paired metabolomics needed to validate it. HallSim recapitulates the G3P/PEtn → senescence amplification loop and predicts G3PP/ETNPPL overexpression as senomorphic.
* [ ] **Trajectory-level validation** — GSE248823 has 3 timepoints per arm (DDIS: D00/D07/D14, OIS: D00/D04/D07). Current concordance uses two-endpoint deltas; matching predicted vs. measured pathway-score *trajectories* (rate of change, time-constant ordering across pathways) would be a substantially stronger validation than scalar deltas.
* [ ] Validate against scRNA-seq (Tabula Muris Senis, Ma 2020 caloric restriction) — pseudobulk ssGSEA
* [ ] PINNs: physics-informed loss for NeuralODE training

### Stochastic DISCRETE / Gillespie support

Several aging mechanisms are intrinsically stochastic at the single-cell
scale and not well-described by the ODE mean-field:

* **Telomere shortening** — discrete length loss (≈50–200 bp) per
  division; aggregate length depends on division-history sampling
* **Somatic mutation accumulation** — Poisson process per genome per
  cell-cycle; rate is the Genomic Instability hallmark
* **Senescence entry** — threshold-on-stochastic-state transition
  (DDR signal accumulates by jumps; entry fires once threshold crossed)

The framework already has the right abstraction (`ProcessKind.DISCRETE`
with `update(t, state) -> delta` and `ProcessKind.EVENT` with
`condition`/`handler`). What's missing is:

* PRNG plumbing — pass a `jax.random.PRNGKey` into the Scheduler and
  thread split keys to each stochastic Process
* A `StochasticDiscrete` example Process (telomere-shortening or
  per-genome mutation Poisson) demonstrating the contract
* Population-level statistics via batched y0 with per-cell PRNG keys
  (the existing batched-IC machinery already gives the cell axis;
  we just need the key axis alongside it)

### Multi-cell / inter-cell communication

Batched y0 currently gives **N independent cells** — every batch element
runs in isolation. Tissue-level aging biology (niche signaling, paracrine
SASP, contact inhibition) requires cells that *exchange state*. Two
plausible architectures:

* **Mean-field paracrine.** Each cell reads a population aggregate of a
  secreted factor (e.g. SASP-IL6 = mean of all senescent cells'
  secretion). Implementation: a `PopulationAggregate` Process that
  reduces along the batch axis and writes a shared store path read by
  every cell. Works inside one `Scheduler.run` call; gradients flow.
* **Spatial / graph-coupled.** Cells live on a graph (epithelium
  topology, niche geometry); communication is along edges. Reaction-
  diffusion or graph-Laplacian coupling. Heavier — needs a spatial
  state representation orthogonal to the per-cell trailing axis.

Concrete first-cut deliverable: a `PopulationAggregate` Process and a
SASP-propagation demo where the senescence fraction in a population
modulates each individual cell's p53 baseline. Demonstrates that
HallSim's composability extends to inter-cell coupling without leaving
the JAX-native execution model. Designed as a natural follow-up.

### Other queued items

* [ ] LLM agent-assisted model composition
* [ ] FBA / genome-scale metabolism via `jaxopt`-based LP — couples
  ERiQ signaling state to BiGG-scale flux distributions with gradients
* [ ] 3D spatial diffusion & ECM modelling

## SBML Import

* [ ] **Translate SBML events into `ProcessKind.EVENT`** — generic event translator,
  so models with discontinuous state resets (Proctor 2008 BIOMD0000000188 and
  ~10–20% of curated BioModels) become importable. Diffrax 0.5+ already supports
  events natively; HallSim already has `ProcessKind.EVENT`. The missing piece is
  parsing SBML event MathML (trigger expressions, assignments, delays, persistence)
  and emitting the corresponding `condition` / `handler` methods. Most useful
  long-term enhancement to the SBML pipeline.
