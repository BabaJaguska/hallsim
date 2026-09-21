# HallSim Roadmap

Planned and queued work, grouped by area. See
the CrossGen suggestions (working notes, not in the repository) for the cross-domain
analysis behind the Scheduler items.

## Scheduler & Multi-Scale

* [ ] Combine Strang splitting + interpolated coupling (currently mutually exclusive)
* [ ] Compiled hybrid Strang splitting for DISCRETE/EVENT processes — define
  jump timing within the half-steps before implementing the static JAX path
* [ ] Compiled hybrid interpolated coupling — define dense-waveform replay
  and event ordering for continuous/discontinuous coupling
* [ ] Compiled hybrid waveform relaxation — replay threshold crossings and
  handlers deterministically across fixed Gauss-Seidel sweeps
* [ ] Event-bearing composites under batched `y0` with a public event-buffer
  contract
* [ ] Anderson acceleration for waveform relaxation convergence
* [ ] Mori-Zwanzig memory kernel for fast→slow coupling (captures history effects)
* [ ] Coupling residual spectral monitoring (early-warning diagnostic)
* [ ] IFT-based adjoint at sync boundaries (for gradient-based optimization)
* [ ] IMEX (implicit-explicit) solver for stiff multi-scale systems

## Calibration & Uncertainty

In order — each step is the prerequisite for the next; the rationale and
costs are in [Uncertainty: options and costs](#uncertainty-options-and-costs)
below.

* [ ] **Make the loss a proper log density.** `gaussian_nll` means over entries
  and drops the ½, `data_loss` means again over arms, `prior_weight` is a free
  multiplier. The MAP is unaffected; this is the prerequisite for reporting
  widths. Needs a residual σ — real precision weights where the data has
  them, else σ̂ from the MAP residuals, which `identifiability.residual_scale`
  already computes for the identifiability report.
* [ ] **Laplace / delta-method bands** — `(JᵀJ/σ̂² + Π_prior)⁻¹` off the Jacobian
  `identifiability.py` already builds, and `sqrt(diag(J Σ Jᵀ))` on any
  prediction. Closes P3.9 for the common case at the cost of one Jacobian
  (~41 s on the multi-hallmark demo, one fit step). Also adds the same module's
  missing prior-precision term.
* [ ] **Profile likelihood** (Raue 2009) — the nonlinear check on the ellipse,
  and the way to report a fit above `MAX_FIT_CONDITION_NUMBER` instead of
  refusing it. Batched over the profile grid; to check whether the
  equilibration Newton solve survives `vmap` over the parameter axis.
* [ ] **NUTS on a single constituent** (blackjax; DP14 against its deposited fit)
  to measure how wrong the Laplace ellipse is on a real posterior. Not on a
  composite — one gradient is 26 s there, so a chain is ~40 days.

### Uncertainty: options and costs

Fitted outputs are point estimates; a lab needs bands. An optimizer and a
sampler answer different questions — `Calibrator` returns a MAP point, a
sampler a distribution — so the choice is which uncertainty method, at what
cost, measured against one HallSim gradient.

**Prerequisite.** The loss is not yet a log density: `gaussian_nll` is a
mean without the ½, `data_loss` means again over arms, and `prior_weight` is
a free multiplier. The MAP is unaffected, but every method that reports a
width reads the curvature *and its scale*, so the scale has to be real —
precision weights where the data has them (`weights` already accepts them),
else `σ̂² = RSS/(n − p)` at the MAP, which `identifiability.residual_scale`
already computes. Two functions.

**Laplace + delta method, first.** `Σ_θ = (JᵀJ/σ̂² + Π_prior)⁻¹` off the
Jacobian `identifiability.py` already builds, and `sqrt(diag(J_pred Σ_θ
J_predᵀ))` on any prediction, held-out arms included. One Jacobian at the
MAP, about one fit step (~41 s on the multi-hallmark demo). It also ranks
which parameter dominates a band, i.e. which measurement to make next.
Local and Gaussian, so least reliable where the Fisher condition number is
large; the profile is its standing check.

**Profile likelihood** (Raue et al. 2009): fix θᵢ on a grid, re-optimise
the rest, read the interval off the χ² profile. About a day sequential at 7
params × 15 grid points, embarrassingly parallel and batch-shaped. It
separates structural, practical and identifiable with the nonlinear
boundary, and it turns a fit above `MAX_FIT_CONDITION_NUMBER` from a refusal
into "this combination is determined, and here is how far the rest can run".

**NUTS.** One reverse gradient on the composite is ~26 s; at 30–100
gradients per draw, 1000 warmup + 1000 draws is ~40 days per chain. Worth
doing on a single constituent instead (DP14 against its deposited fitting
data, milliseconds per gradient) to measure how far the Laplace ellipse
departs from a real posterior; blackjax is pure JAX and the integration is
~20 lines. A sampler also needs clamps as a bijection rather than a clip,
and `steady_state` to report non-convergence so the density goes to `−inf`
there.

**Resampling** (multistart ensembles, an arm- or reporter-level bootstrap)
asks how far the fit moves when the data moves; the most expensive per unit
of insight on a composite, and the only one short of a sampler that finds a
second basin.

### One calibration path for mechanism and learned components

Mechanistic and neural components are fitted jointly, through one
objective. `CalibrationProblem` fits named scalars in log space with priors,
held-out arms, best-iterate tracking, checkpoints and the Fisher gate, and
the same path carries what a learned block needs: an objective over observed
states, derivative matching without a solve, and multiple shooting with
curriculum and continuity. A reporter is a store path plus a summary plus a
data key, and a trajectory observation is the same thing with the identity
summary and the path as its key. The rest are options.

Done: `Arm(condition, reference)`, reference `"t0"` | a condition | `None`,
the last comparing values in the data's units through `Readout.scale`
(P3.24). `Condition(start=, window=)`: a condition from its own state over
its own span, a batched start running one member per initial state through
the Scheduler's batch axis, a timepoint's data then a frame with one row per
member; `trajectory_readouts(*paths)` reads store paths as reporters, so an
observed trajectory is one condition, one arm with no reference and the
paths as data columns. `shooting_conditions(ts, ys, paths, segments=,
match=)` cuts a trajectory set into such conditions, consecutive windows
sharing their boundary sample (the continuity term, inside the same loss),
`match` keeping a prefix of each window for a curriculum stage and a subset
of the arms to `data_loss` being the active windows. `Collocation(ts, ys,
paths, condition=, weight=)`: the composite's field at the observed states
against their central-difference slopes, each path scaled by its slope's
spread, no solve; `collocation_loss` alone is a pretraining stage, and
`loss` adds it at `weight`. `FitBlock(process_name, frozen=)`: a
learned block's trainable leaves as one flat fittable, linear space, no
prior, no clamp, outside the identifiability report (`scalar_fittables` is what
that report and the log transform cover), `fit` switching to reverse mode
when one is present. Minibatches are a PRNG key the Calibrator threads
into the loss (`minibatch_seed`): `Collocation.batch` samples,
`member_batch` members of a batched condition; iterates are ranked on the
whole objective, or `eval_loss_fn`, every `eval_every` steps.
`fit_neuralode_derivative` and `fit_neuralode_shooting` are wrappers over
all of this (2026-09-20); the derivative stage runs in 68 s against 1,012 s
before. Open: the shooting stage's ranking evaluations (25 per stage) are
the remaining tunable cost, and a second shooting seed would tighten its
held-out number.

Where a learned component belongs, in the order an agent meets it: the
coupling between published blocks, which is where the demo's four
hand-anchored edges are; the readout from mechanistic state to the
transcriptome; a module with no published rate laws, the secretory
response downstream of NF-κB; and a model whose code is not an ODE, or
whose code is gone and only trajectories survive. A model that exists as
readable code — `discover`'s `source:matlab` candidates — is translated,
not surrogated: write it as SBML, check it against the original with the
conformance run, and it enters through the importer with its parameters,
provenance and gradients intact. The hybrid demo shows the mechanics a
learned block needs, gradients through the mechanistic and learned parts
together and a held-out score on the block; the cases above are where that
block earns its place. Three timepoints of bulk arrays cannot constrain a
learned block, and the Fisher gate says so.

jaxkineticmodel (PLOS Comput Biol 2025), checked from source: its package
fits scalars in log2 space by masked MSE on states at the data's times,
single shooting, last iterate returned, serial Latin-hypercube multi-start,
no networks. Its hybrid example lives in a rebuttal script outside the
package: an `eqx.nn.MLP` added to the whole vector field, kinetic
parameters frozen, trajectory MSE with a negativity penalty, Adam with the
global norm clipped at 1, trained on noisy trajectories simulated from the
full model. The same two-loop split this section removes.

## Models & Validation

* [ ] **`place_dose` — pick a stimulus level by measured contrast.** The
  framework places a Hill gate from operating levels (`place_hill_gate`) and a
  clamp rate from a measured flux (`place_clamp_rate`), but has nothing for the
  commonest placement of all: *at what dose does this readout discriminate?* A
  deposit's own stimulus is usually chosen to saturate — Kallenberger's
  `CD95L = 16.6` moves commitment only 1.18× for a 3.5× receptor change, where
  `CD95L = 2.0` moves it 4.25× — so composing at the deposited value silently
  puts the model where it cannot respond. The rule is one line (argmax over the
  dose axis of the readout difference between two levels of the contrasting
  parameter) and it needs a simulation rather than arithmetic, so it belongs
  next to `place_clamp_rate`, which already measures through the composite.
  The numbers above come from a hand-run probe.
* [ ] **Lipid-metabolism extension** — Tighanimine et al. 2024 (*Nat Metab*, the paper behind GSE248823) identified a G3P/PEtn homeostatic switch as *causal* for senescence (p53 → glycerol kinase activation drives G3P↑; PCYT2 post-translational inactivation drives PEtn↑; lipid droplet biogenesis is the downstream effect). Adding a `LipidMetabolism` Process (states: G3P, PEtn; inputs: `p53_activity`, a PCYT2-PTM proxy; outputs: a senescence-amplifying signal that feeds back into the SASP axis) would let HallSim test their causal claim *in silico* — and the GSE248824 SuperSeries includes the paired metabolomics needed to validate it. HallSim recapitulates the G3P/PEtn → senescence amplification loop and predicts G3PP/ETNPPL overexpression as senomorphic.
* [ ] **Trajectory-level validation** — GSE248823 has 3 timepoints per arm (DDIS: D00/D07/D14, OIS: D00/D04/D07). Concordance reads two-endpoint deltas; matching predicted vs. measured pathway-score *trajectories* (rate of change, time-constant ordering across pathways) would add the dynamics to the score.
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

Landed: the Gillespie lane (`hallsim.stochastic`) runs an imported
reaction network at reaction level inside a composite, with
`Scheduler(batch_mode=...)` choosing host threads for a stochastic batch
(`simulate demo proctor2007-ssa`, `multi-hallmark-ssa`). Queued on top of
it:

* PRNG plumbing for hand-written stochastic `DISCRETE` processes — a
  `jax.random.PRNGKey` into the Scheduler, split keys per process
* A `StochasticDiscrete` example Process (telomere shortening or a
  per-genome mutation Poisson) demonstrating the contract
* Population-level statistics via batched y0 with per-cell PRNG keys

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
  signaling state to BiGG-scale flux distributions with gradients. Queued
  without a sponsor: the first application that asked for it (a CRISPRi
  perturbation-extent problem, 2026-09-06) turned out to need only
  essentiality annotation, since a genome-scale network routes around single
  deletions.
* [ ] 3D spatial diffusion & ECM modelling

## Model-adjacent formats

**A repository is not a format, and SED-ML is not a model format.** SBML, XPP
`.ode` and CellML all describe a *model*; SED-ML describes a *simulation
experiment over* a model — which model, which time span, which parameter
changes per task, which outputs. Keep the distinction explicit in anything
user-facing: HallSim imports SBML and XPP, discovers CellML and COMBINE
archives without importing them, and would *execute* SED-ML rather than
import it.

### SED-ML: run a deposit's own verification

* [ ] **Read the SED-ML that curated deposits already ship, and run it.**
  Curated BioModels entries carry a `.sedml` alongside the model (and a COPASI
  `.cps` plus MATLAB/Octave exports); `discovery.download_biomodel_files`
  fetches them as of 2026-09-04. Where the SED-ML encodes the curator's actual
  reproduction run, executing it answers "does our import of this deposit
  behave like the reference implementation?" without anyone hand-writing a
  probe.

  **Not every shipped SED-ML is that, and the difference must be detected.**
  Kallenberger 2014 (BIOMD0000000524) ships autogenerated boilerplate —
  `outputEndTime="10"` against a 240-minute published figure, a generic
  KISAO:0000694 algorithm, and a curve for every parameter including `cell`.
  Run blindly it would compare the wrong window and pass. A SED-ML whose span
  does not cover the model's own dynamics, or whose outputs are undifferentiated
  from its parameters, is boilerplate and should be reported as absent rather
  than executed.

  **Why this is on the roadmap.** Every candidate screened on 2026-09-04
  came down to the same question — does the deposit reproduce its paper —
  and each time the check was built by hand. `intake.published_fit_chi2`
  covers the minority of papers that deposit fitting data; SED-ML covers the
  majority that deposit a curated simulation instead.

  Scope is the subset curated deposits actually use, not SED-ML L1V4 in full:
  `<uniformTimeCourse>` (start, end, steps), `<task>` and repeated tasks,
  `<changeAttribute>` for per-task parameter changes, `<dataGenerator>` and
  `<plot2D>`/`<report>` for the outputs to compare. Map those onto
  `Scheduler.run` and a comparison against the deposit's own exports.

  Two things fall out of it. It gives `intake` an automatic reproduction gate,
  which is the check the model-selection work most needed. And it demonstrates
  a genuinely different axis than SBML/XPP import — the framework consuming an
  *experiment description*, not another model dialect. A composite carries
  its members' events by default, so a SED-ML task runs the model as
  deposited.

### CellML

* [ ] **Importer queued.** `discovery.search_physiome` finds CellML models
  and returns a pointer; an importer follows the first model that needs one.

## Model discovery: which repositories are worth adding

`SOURCES` now reaches BioModels, **JWS Online**, ModelDB, BioSimulations and
Physiome. JWS was added 2026-09-06: **676 unique curated kinetic models**
served as SBML, reachable by `jws:<slug>` in `process_from_sbml`. Its listing
endpoint repeats a slug once per model version, so it returns 826 rows; 30 of
those models (`beuke*`) answer 500 on the detail endpoint and index with slug
only. 646 carry species and reaction names, 520 a PubMed ID.

The criterion for the rest is **whether the maths fits an ODE composite**, not
whether an importer exists — writing importers is the framework's job. What a
source *holds* is the second criterion, and it has to be measured rather than
assumed: the catalogue check below moved SBML qual off the top of this list.

| Source | Formalism | Verdict |
|---|---|---|
| **BioModels' own SBML-qual branch** | Boolean / logical, SBML qual | **Add first — it needs no new source.** 72 logical/Boolean deposits are already in BioModels, 65 of them SBML-qual, 61 in the `MODEL` branch that `curated_only=True` hides. Reaching them is a flag, not an adapter. The importer is the work: a logical model is a discrete-time update rule, which is `ProcessKind.DISCRETE` and its `update(t, state)`, so the formalism exists and this gives the DISCRETE path the standing workload P1.16 says it lacks. |
| **Cell Collective** | Boolean / logical, SBML qual | **Blocked, not merely unwritten.** The API authenticates with `X-AUTH-TOKEN`; every anonymous endpoint 404s, and all five registered sources are anonymous, so this needs a credential story before it needs a parser. Its own client `ccapi` still points at the dead `ginsim.org` for model data. |
| **GINsim** | Boolean / logical, SBML qual | **Small.** `ginsim.org` has lapsed to a parked domain; the live repository is `ginsim.github.io/models/`, **50 models**. Worth harvesting once a qual importer exists, but it is not a catalogue on the scale the previous entry assumed. |
| **NeuroML-DB** | conductance-based ODEs | **Add.** 1,500+ published models; Hodgkin-Huxley is an ODE system, so this needs a parser, not a new solver. |
| **DDMoRe / Open Systems Pharmacology** | compartmental PK/PD ODEs | **Add.** Linear compartment models are among the easiest to import. Check DDMoRe's service status first. |
| **FAIRDOMHub / SEEK** | SBML inside COMBINE/OMEX | Worth it once OMEX unpacking exists; DOI-linked investigations. |
| **Zenodo / Figshare / Dryad** | arbitrary | Catches the common case of a model existing only as a paper supplement. No format guarantee, so this is a fetch-and-triage path rather than a search adapter. |
| **BiGG / VMH / ModelSEED / KBase** | constraint-based (FBA) | **Do not add as a search source.** A genome-scale reconstruction is stoichiometry with no rate laws, solved by linear programming over a steady-state null space rather than integrated. Importing one yields a composite with no dynamics. If constraint-based models matter, the question is whether `steady_state` grows an LP path — an architecture decision, not an importer. |
| **CoMSES / NetLogo** | agent-based, stochastic | No shared state vector and no derivative. Wrong formalism. |
| **CellML Model Repository** | CellML | Already covered: it runs on the Physiome infrastructure already registered. A format view, not an independent source. |

Ordering: **SBML qual first**, then NeuroML, then PK/PD, on the strength of
the existing formalism; the catalogue check below shows it does not fill the
missing response programs.

### What the logical-model catalogues actually contain

Measured against the six response programs in `VCC/docs/model_candidates.md`,
across BioModels' qual branch (72 deposits) and GINsim (50 models):

| program | logical models |
|---|---|
| growth arrest / cell cycle | 8 in BioModels, ~8 in GINsim |
| NF-κB | 2 |
| p53-DDR | 0 in BioModels' qual branch; 2 in GINsim |
| UPR | 0 |
| ISR-ATF4 | 0 |
| nucleolar stress / ribosome biogenesis | 0 |

So qual adds depth to the two programs that are *already* best covered by
kinetic models, and closes neither gap. The gap is not a formalism gap — those
programs are absent from BioModels in every formalism, curated or not, and a
targeted search over ISR and nucleolar terms returns nothing usable. They need
a model built, not found.

JWS holds `jws:goodman`, a PKR/eIF2α model (species `PKRp`, `eIF2ap`,
`P58a`, influenza `NS1`): the ISR sensing arm, though not the ATF4
translational-control arm.


## SBML Import

* [x] **SBML events translate into `ProcessKind.EVENT`** (`hallsim.sbml_events`):
  triggers, assignments to species and to parameters (Yao 2008's serum steps
  run as published), persistence; a composite carries its members' events by
  default.
* [ ] **Delayed events and event priorities** — the two constructs the
  translator still declines; the census counts how many deposits carry them.
