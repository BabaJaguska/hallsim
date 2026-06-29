# HallSim: A Differentiable, Composable Multi-Scale Simulator for Aging Biology
[![Basic CI/CD Workflow](https://github.com/BabaJaguska/HallSim/actions/workflows/basic_CI_linux.yaml/badge.svg)](https://github.com/BabaJaguska/HallSim/actions/workflows/basic_CI_linux.yaml)

**HallSim composes independently-published systems-biology models into one multi-scale dynamical system — and calibrates the whole thing by gradient descent through the ODE solve.** Built on JAX/Equinox/Diffrax, with a focus on aging biology, where no single model captures the crosstalk between hallmarks.

- **End-to-end differentiable.** The entire composite — multiple stiff (SBML) models, operator-split across timescales — is a single differentiable function. Mechanism parameters spread across separate publications are fit with the same reverse-mode autodiff that trains neural networks, *through* the stiff ODE solve. GPU friendly. Demo with held-out validation. See [Differentiable calibration of stiff multi-model composites](#differentiable-calibration-of-stiff-multi-model-composites).
- **Agent-friendly by construction.** A model is discovered and imported from BioModels in two calls (`search_for_model` → `process_from_sbml`), wired by a plain `{process: {port: path}}` topology dict, and its fittable parameters self-document via `Composite.calibration_targets()`. Can accept custom models and neuralODE models as well. Typed ports carry units and ontology annotations, and `analyze_composability` proposes how to merge overlapping models. The framework is meant for an LLM agent to assemble and calibrate a digital twin without bespoke glue code.

## Background & Motivation

Aging is a complex, network-level phenomenon. Its hallmarks — such as mitochondrial dysfunction, genomic instability, and altered intercellular communication — do not act in isolation. Instead, they form dense webs of feedback and crosstalk.

Traditional approaches to modeling aging have been reductionist. Inspired by conceptual frameworks like Cohen et al. (2022) [1], HallSim embraces complex systems theory to explore emergent properties of aging arising from loss of resilience across multiple sub-systems. Existing tools and repositories standardize and simulate individual dynamical models well, but generally lack support for composable model libraries — reusable modules with high-level interfaces that assemble into heterogeneous multi-scale systems. HallSim closes that gap JAX-native, so composition buys differentiability, JIT, and batched populations for free.

---

## Project Goals

* A composable, differentiable, multi-scale simulator for aging biology — bring your own modules (hand-written, SBML-imported, or learned via `NeuralODE`)
* High-level severity handles for the 12 hallmarks of aging [2] (4 mapped today — see [Hallmark Handles](#hallmark-handles) — each new one a single handle away)
* Calibrate interventions and emergent phenotypes against real experimental data, with held-out validation (ssGSEA pathway scores, gene-reporter concordance)
* Make multi-model composition tractable for AI agents building digital twins
* Serve as an educational in-silico testbed for perturbations (rapamycin, caloric restriction, etc.)

---

## Architecture

![HallSim Architecture](docs/assets/hallsim_architecture.png)

HallSim is built on a **composable architecture** using JAX/Equinox/Diffrax. Design borrows composition semantics from Vivarium [3] and scheduling concepts from Ptolemy II [5], implemented natively on JAX for GPU acceleration and differentiability.

**Scope:** HallSim deliberately covers a narrower set of modeling formalisms than Vivarium-collective in exchange for end-to-end differentiability, JIT, and native batched populations. See [docs/formalism-coverage.md](docs/formalism-coverage.md) for the full coverage table and the trade-off rationale.

**Core concepts:**

| Concept       | Description |
|---------------|-------------|
| **Process**   | Equinox module (`eqx.Module`) — declares typed ports and a `kind` (CONTINUOUS / DISCRETE / EVENT). Parameters are JAX arrays: differentiable, JIT-compilable, vmappable. |
| **Port**      | Named connection point with a role (`INPUT` / `EVOLVED` / `EXCLUSIVE` / `LATCHED`), default value, units, description, and ontology annotation. |
| **Topology**  | Static wiring map `{proc_name: {port_name: store_path}}`. Defined at composition time, not inside processes. |
| **Composite** | Bundles processes + topology. `build_rhs()` / `build_group_rhs()` return JAX-compatible ODE right-hand sides. Auto-groups continuous processes by timescale. |
| **Scheduler** | The unified runner — handles every composite shape. Multi-rate orchestrator: groups continuous processes by timescale (Lie splitting), dispatches discrete processes at intervals, fires events at sync points. Supports `coupling_mode="interpolated"` for dense-output coupling between groups. For single-group continuous composites with no events, takes a fast path that issues one `diffrax.diffeqsolve` over the whole `t_span` (no per-macro-step overhead). The state pipeline is shape-polymorphic — pass a `(n_vars,)` y0 tensor for a single run or a `(batch, n_vars)` tensor for a JAX-native population study, no extra `vmap` required. See [design doc](docs/design-multiscale-scheduler.md). |
| **Store**     | Flat `dict[str, jnp.ndarray]` with path-like keys (e.g., `"cytoplasm/ROS"`). A valid JAX PyTree. |

**Process kinds:**

- `CONTINUOUS` (default) — computes `derivative(t, state) -> dy/dt`, solved by Diffrax ODE integrator.
- `DISCRETE` — computes `update(t, state) -> delta`, called every `dt_step` seconds by the Scheduler.
- `EVENT` — declares `condition(t, state) -> bool` and `handler(t, state) -> delta`, fires on False-to-True crossing.

**Port roles:**

- `INPUT` — read-only, process uses this value but doesn't write a derivative.
- `EVOLVED` — additive: multiple processes can contribute derivatives to the same store path (summed).
- `EXCLUSIVE` — sole ownership: only one process may write to this store path (validated at composition time).
- `LATCHED` — written by discrete/event processes, read as constant by continuous processes within a macro step.

**Validation layer** (on by default, runs at composition time):

| Subsystem          | What it checks |
|--------------------|----------------|
| **UnitChecker**    | pint-based dimensional analysis across shared store paths |
| **SemanticChecker**| Ontology ID comparison (ChEBI, GO, SBO, UniProt) for species disambiguation |
| **GraphAnalyzer**  | Feedback cycle detection, fan-in analysis, coupling density, unfed-INPUT detection |
| **CouplingAuditor**| Heuristic duplicate-reaction detection via description overlap |

Validation runs warnings-by-default but raises on hard conflicts (incompatible units, ontology mismatches at shared paths). Use `semantic_validation={"strict": True}` to promote warnings to errors; disable with `semantic_validation=False`. Per-subsystem opt-out via `semantic_validation={"check_units": False, ...}`. Numerical-range mismatches at coupled store paths are a calibration problem, not a validation problem — use `hallsim.calibration.Calibrator` to fit the rate constant(s) that bring the two sides into a compatible regime.

**Composing composites.** `Composite` accepts other `Composite` instances inside its `processes` dict — they are flattened with namespace prefixes (`outer.sub_proc` for processes, `outer/path` for store paths). Bare `Process` values get their ports auto-prefixed under the outer key too. A `rewire={old_path: new_path}` kwarg on the constructor aliases overlapping biology onto canonical paths. `analyze_composability(**composites)` (in `hallsim.validation`) surfaces candidate overlaps before merging by matching on ontology IDs first, then name and substring heuristics.

```python
from hallsim import Composite, analyze_composability
from hallsim.sbml_import import process_from_sbml

a = process_from_sbml(582, name="dp14")      # DallePezze2014 — 23 species
b = process_from_sbml(157, name="gz06")      # Geva-Zatorsky 2006 — p53 oscillator
report = analyze_composability(dp14=a, gz06=b)
print(report)                                 # candidate overlaps + suggested rewire
merged = Composite(
    processes={"dp14": a, "gz06": b},
    rewire=report.suggested_rewire,
)
```

### Example composites and SBML import

HallSim is a *framework* for composing biological models, not a model library. The expectation is that you bring your own Processes — either hand-written, imported from curated SBML on BioModels, or learned via `NeuralODE`. A small set of example composites ships under [`src/hallsim/models/`](src/hallsim/models/) to demonstrate the framework's patterns:

- A **DP14-anchored multi-hallmark composite** ([`multi_hallmark.py`](src/hallsim/models/multi_hallmark.py)) — three independent BioModels SBML imports (DallePezze 2014 + Geva-Zatorsky 2006 + Ihekwaba 2004) plus one cross-publication mechanistic edge: `MtorNFkBActivator` wires DP14's mTORC1 into the Ihekwaba NF-κB module (mTOR → IKK, activating). DP14↔GZ06 are coupled instead through the Genomic Instability hallmark as a shared experimental knob — each responds to the same severity at its own calibrated scale. Spans Cellular Senescence, Deregulated Nutrient Sensing, Genomic Instability, and Inflammaging in one substrate. The current validation composite.

The SBML import path ([`sbml_import.py`](src/hallsim/sbml_import.py)) is one of the framework's main entry points: it auto-generates a Process from any BioModels entry via `sbmltoodejax`, auto-populates every SBML constant into `SBMLProcess.parameters` so the full mechanism surface is immediately discoverable via `Composite.calibration_targets()`, includes a libsbml-driven `<functionDefinition>` inliner that unlocks the large majority of curated models that would otherwise hit "Custom functions are not handled" upstream, pre-flight checks that reject `<event>` blocks and unsupported MathML operators with actionable error messages, and MIRIAM annotation extraction that populates `Port.ontology` from species CVTerms. Bundled SBML files live under [`models/<author><year>/`](models/) for offline use; arbitrary BioModels IDs download to `~/.cache/hallsim/biomodels/` on first import.

The BioModels REST API is live end-to-end: `process_from_sbml(582)` fetches and imports a model by ID, and the curated repository can be searched by keyword via `sbmltoodejax.biomodels_api.search_for_model("genotoxic stress NFkB")` — so models are discoverable, not just retrievable by known ID. Discover-then-import in two calls makes the catalog directly available to an agent composing a new system:

```python
from sbmltoodejax.biomodels_api import search_for_model
from hallsim.sbml_import import process_from_sbml

hits = search_for_model("genotoxic stress NFkB")        # -> [{'id': 'MODEL2307130001', ...}]
proc = process_from_sbml(hits[0]["id"], name="dna_nfkb") # fetch + auto-generate a Process
```

The hand-written Processes that ship — [`saturating_removal.py`](src/hallsim/models/saturating_removal.py) (Uri Alon damage model), [`kick_event.py`](src/hallsim/models/kick_event.py) (one-shot perturbation EVENT pattern), the [ERiQ](src/hallsim/models/eriq.py) decomposition, and [`neuralode.py`](src/hallsim/models/neuralode.py) — exist either as reference implementations of common patterns or to support the example composites above.

#### Merge or couple? — when two models share a node

Two models both have an "NF-κB" — same entity (merge) or distinct (couple)? It's an identity question, not a biology one: run `analyze_composability(a=.., b=..)` ([`validation.py`](src/hallsim/validation.py)). A shared ontology ID ⇒ same entity ⇒ merge (point both at one store path; `EVOLVED` sums them; `report.suggested_rewire` → `Composite(rewire=..)`). No ontology match ⇒ take the conservative choice — don't merge, add one documented coupling edge — and let a held-out [`gene_reporters`](src/hallsim/gene_reporters.py) split score it. Example: Konrath 2023 vs the Ihekwaba NF-κB module reports **no overlap** (Konrath ends at IKK, has no NF-κB species) ⇒ couple Konrath's IKK → `nfkb/IKK`, don't merge.

### Hallmark Handles

Each hallmark of aging is represented as a 0-1 severity handle that modulates parameters across one or more processes. Hallmark severity is differentiable end-to-end: `jax.grad` through the whole pipeline works.

```python
from hallsim import Composite
from hallsim.hallmarks import apply_hallmarks
from hallsim.models.multi_hallmark import build_multi_hallmark_composite

base = build_multi_hallmark_composite()

# Rapamycin = downward severity shift on Deregulated Nutrient Sensing.
# The hallmark mapping targets DP14's mTORC1_S2448 phosphorylation rate
# via the parameters dict; severity=1.0 → SBML default (full
# dysregulation), severity=0.0 → 30% of default (rapamycin-rescued).
treated_procs = apply_hallmarks(
    base.processes, {"Deregulated Nutrient Sensing": 0.3}
)
treated = Composite(
    processes=treated_procs,
    semantic_validation={"check_semantics": False},
)
# Run treated and control composites, compare gene-reporter outputs.
```

Hallmark transforms are **multiplicative of the current calibrated
base value**: a transform receives `(severity, base)` and returns
`base * f(severity)`. This lets `Calibrator` substitute mechanism
parameters via the `parameters` field and then apply hallmarks at the
experimental severity profile without the hallmark clobbering the
calibrated values — both `severity` and `base` are differentiable
through `apply_hallmarks`.

**Pharmacological interventions belong on the hallmark layer they perturb**, not as separate Processes. Rapamycin targets mTORC1 → maps to Deregulated Nutrient Sensing. The same hallmark applies to ERiQ-based composites (targets ERiQ's `GLYCOL_SA`) and to DP14-based composites (targets DP14's `mTORC1_S2448_phos_by_AA_n_Akt_pS473`) without changing how the user calls `apply_hallmarks`.

**Genomic Instability** in the multi-hallmark composite drives both DP14's `DNA_damaged_by_irradiation` rate constant and GZ06's `psi` damage-signal parameter, each at its own model's calibrated scale. Same severity, no internal topology coupling between the SBML models — the hallmark is the shared knob, so cross-model coupling is mediated at the experimental-condition level rather than via state-into-constant patching (a foot-gun the framework deliberately doesn't expose). The composite spans Cellular Senescence, Deregulated Nutrient Sensing, Genomic Instability, and Inflammaging in one substrate.

**Population studies via batched `y0`.** The Scheduler's state pipeline
is shape-polymorphic — a batched `y0` tensor of shape `(batch, n_vars)`
flows through every group's Diffrax solve as a single batched
computation, no `jax.vmap` over `Scheduler.run` required:

```python
keys = comp.store_keys()
y0 = comp.initial_state_vec()                           # (n_vars,)
y0 = jnp.broadcast_to(y0, (1024, len(keys)))            # (1024, n_vars)
# Vary the DP14 starting damage across cells:
y0 = y0.at[..., keys.index("dp14/DNA_damage")].set(
    jnp.linspace(0.0, 10.0, 1024)
)
result = Scheduler().run(comp, t_span=(0.0, 50.0), macro_dt=5.0, y0=y0)
result.ys.shape                          # (n_time, 1024, n_vars)
result.get("dp14/CDKN1A").shape          # (n_time, 1024)
```

On a GPU this is near-flat in `batch` — kernel launch dominates over
per-cell compute. On CPU, scaling is sub-linear because Python overhead
amortizes across the batch.

### Data validation (gene reporters, calibration, held-out splits)

HallSim mechanistic states are validated against transcriptomic data via **single-gene reporters** in [`hallsim.gene_reporters`](src/hallsim/gene_reporters.py): one canonical reporter gene per mechanistic store path, with literature-anchored sign expectations and a per-reporter trajectory summary (defaults to the endpoint; oscillator-driven reporters like DDB2 use a cycle-averaged readout). The multi-hallmark composite uses these reporters:

| Gene | Store path | Note |
|---|---|---|
| `CDKN1A` (p21) | `dp14/CDKN1A` | senescence/arrest marker — DP14 transcribes it via FoxO3a × damage (DP14 has no explicit p53 species) |
| `DDB2` | `gz06/x` | direct p53 transcriptional target — maps to GZ06's p53 protein (cycle-averaged) |
| `HMOX1` | `dp14/ROS` | Nrf2/ARE oxidative-stress reporter |
| `NFKBIA` (IκBα) | `nfkb/IkBa` | NF-κB autoregulation |
| `CYCS` (cytochrome c) | `dp14/Mito_mass_new` | mitochondrial biogenesis |
| `EIF4EBP1` (4E-BP1) | `dp14/mTORC1_pS2448` | kinase-level mTOR proxy |

**Calibration is a first-class framework primitive.** [`hallsim.calibration`](src/hallsim/calibration.py) provides `Calibrator` (the low-level forward/reverse-mode autodiff loop) plus a high-level declarative API — `CalibrationProblem` + `Condition` + `ParameterRef` + `CalibratableParam` + `GeneExpressionDataset` — for wiring any composite to any held-out gene-expression dataset:

```python
from hallsim.calibration import CalibrationProblem, Condition, ParameterRef
from hallsim.gene_reporters import GeneExpressionDataset, MULTI_HALLMARK_REPORTERS
from hallsim.models.multi_hallmark import build_multi_hallmark_composite

composite = build_multi_hallmark_composite()

# Discover what's fittable. composite.calibration_targets() walks every
# Process, enumerates each SBML constant or hand-rolled scalar attr,
# and subtracts every (process_name, field) pair targeted by any
# hallmark in HALLMARK_REGISTRY. The next agent / human sees only
# mechanism candidates — hallmark-controlled knobs (irradiation dose,
# mTOR phos rate, etc.) are hidden because they're set by
# Condition.hallmarks per experimental arm, not learned from data.
for p in composite.calibration_targets():
    print(f"  {p.process_name}.{p.field}  default={p.default:.4g}  clamp={p.clamp}")

ds = GeneExpressionDataset.from_series_matrix(
    series_matrix_path, platform_path, sample_position_groups=...,
)

problem = CalibrationProblem(
    composite=composite,
    reporters=MULTI_HALLMARK_REPORTERS,
    conditions={
        "ctrl": Condition("ctrl", {"Genomic Instability": 0.0, "Deregulated Nutrient Sensing": 0.5}),
        "DDIS": Condition("DDIS", {"Genomic Instability": 1.0, "Deregulated Nutrient Sensing": 1.0}),
        "RAPA": Condition("RAPA", {"Genomic Instability": 1.0, "Deregulated Nutrient Sensing": 0.3}),
    },
    arm_pairs={"DDIS_vs_ctrl": ("DDIS", "ctrl"), "RAPA_vs_DDIS": ("RAPA", "DDIS")},
    data={"DDIS_vs_ctrl": ds.delta(...), "RAPA_vs_DDIS": ds.delta(...)},
    params={
        # Mechanism rate constants — biological properties that vary
        # across cell states. NOT hallmark targets (the guard rail
        # would raise if you tried to fit one).
        "CDKN1A_transcr": ParameterRef("dp14", "parameters.CDKN1A_transcr_by_FoxO3a_n_DNA_damage", init=0.085, clamp=(0.001, 5.0)),
        "alpha_y":        ParameterRef("gz06", "parameters.alpha_y",      init=0.8,   clamp=(0.01, 10.0)),
    },
    fit_arms=["DDIS_vs_ctrl"],         # in the loss
    held_out_arms=["RAPA_vs_DDIS"],    # evaluated but not fit
)

history = problem.fit(steps=20, learning_rate=0.05)
results = problem.evaluate(history.final_params)
problem.save_outputs("outputs/run/", history)   # graph.png, trajectories_*.png, summary.json
```

The calibration API enforces three principles — hallmark knobs aren't fittable by default, parameter discovery is self-documenting (`Composite.calibration_targets()`), and held-out splits are mandatory. Details in [docs/development.md](docs/development.md#calibration-principles).

The full runnable validation against GSE248823 (etoposide DDIS ± rapamycin) is in [`demos/multi_hallmark_calibrate.py`](demos/multi_hallmark_calibrate.py).

### Differentiable calibration of stiff multi-model composites

The composite is calibrated by gradient descent **through the ODE solve** — the
same reverse-mode autodiff that trains neural networks, applied to mechanism
parameters spread across independently-published SBML models. Two ingredients
make this work on stiff biochemical systems:

- **A stiff solver with a proper Newton root finder.** Stiff subsystems make an
  explicit solver's *sensitivity* (the gradient) blow up even when the forward
  trajectory is fine. The Scheduler auto-detects stiffness per timescale group
  (`hallsim.stiffness.analyze_groups`) and routes stiff groups to an A-stable
  implicit solver (`Kvaerno5` with a Newton root finder); non-stiff groups stay
  on a cheap explicit solver.
- **End-to-end float64.** Adaptive error control at `rtol=1e-6` requires the
  state *and* the right-hand side to be double precision; a single hardcoded
  `float32` anywhere silently caps precision and makes an implicit solver reject
  most of its steps.

**Validation in progress.** On GSE248823, fitting the mechanism parameters
on one arm (DDIS vs control) improves that arm's concordance, but the **held-out**
arm (rapamycin vs DDIS) does **not** improve — the current composite's models are
*co-simulated under shared hallmark dials but not yet mechanistically coupled
tightly enough* for a rapamycin (mTOR) perturbation to propagate to the readouts.
The contribution here atm is making stiff, composed,
multi-model systems **differentiably calibratable at all**, plus a held-out test
that diagnoses where the *model* (not the method) falls short — which motivates
adding curated cross-model coupling (e.g. a damage/p53 → NF-κB crosstalk model)
or a learned coupling edge.

---

## Getting Started

### Install

```bash
make install        # or make install-dev for development
```

### Run demos

```bash
simulate compose              # ROS production + antioxidant defense ODE
simulate compose-kick         # Same system + mid-run perturbation
simulate multiscale           # Continuous + discrete + event scheduling
simulate validate-demo        # Validation layer catching unit/semantic issues
simulate validate-demo --strict # Strict mode: warnings become errors
simulate info                 # Architecture overview

# Multi-timescale coupling comparison (frozen vs interpolated)
.venv_hallsim/bin/python demos/multiscale_coupling_demo.py
```

### Run tests

```bash
make test
# or directly:
.venv_hallsim/bin/python -m pytest tests/ -v
```

### Python API

```python
import jax.numpy as jnp
from hallsim.process import Process, Port, PortRole
from hallsim.composite import Composite
from hallsim.scheduler import Scheduler

# 1. Define processes
class Decay(Process):
    rate: float = 0.1

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=1.0, units="uM")}

    def derivative(self, t, state):
        return {"x": -self.rate * state["x"]}

class Growth(Process):
    rate: float = 0.05

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=1.0, units="uM")}

    def derivative(self, t, state):
        return {"x": self.rate * state["x"]}

# 2. Wire via topology
composite = Composite(
    processes={"decay": Decay(), "growth": Growth()},
    topology={"decay": {"x": "pool/x"}, "growth": {"x": "pool/x"}},
    semantic_validation=True,  # optional: run unit/semantic checks
)

# 3. Solve
result = Scheduler().run(
    composite, t_span=(0.0, 100.0), macro_dt=1.0, save_dt=1.0
)
print(result.ts.shape, result.get("pool/x").shape)
```

### Multi-timescale with Scheduler

```python
from hallsim.process import ProcessKind
from hallsim.scheduler import Scheduler

# Discrete process: fires every dt_step seconds
class CellDivision(Process):
    kind: ProcessKind = ProcessKind.DISCRETE
    dt_step: float = 86400.0  # once per day

    def ports_schema(self):
        return {
            "count": Port(role=PortRole.LATCHED, default=1.0, units="cells"),
            "damage": Port(role=PortRole.INPUT, default=0.0),
        }

    def update(self, t, state):
        can_divide = state["damage"] < 0.8
        return {"count": jnp.where(can_divide, state["count"], 0.0)}

# Scheduler handles multi-rate orchestration
scheduler = Scheduler()
result = scheduler.run(composite, t_span=(0.0, 86400.0), macro_dt=3600.0)
print(result.events)  # log of fired events
```

### Splitting schemes and coupling modes

The Scheduler supports multiple strategies for how groups communicate
during operator splitting. Both the **splitting scheme** (how groups
are ordered) and the **coupling mode** (what state information groups
exchange) are configurable:

```python
# Lie splitting (default): groups solved sequentially, one pass. O(dt) error.
scheduler = Scheduler(splitting="lie", coupling_mode="frozen")

# Strang splitting: symmetric half-steps cancel leading-order error. O(dt²).
scheduler = Scheduler(splitting="strang")

# Interpolated coupling: groups query a dense interpolant of the previous
# group's trajectory instead of a frozen snapshot.
scheduler = Scheduler(coupling_mode="interpolated")

# Adaptive macro_dt: PLL-inspired — shrinks step when coupling residual
# is large, grows when it's been small for several consecutive steps.
scheduler = Scheduler(adaptive_dt=True)

# Combine as needed:
scheduler = Scheduler(splitting="strang", adaptive_dt=True)
```

See `demos/multiscale_coupling_demo.py` for a comparison. On a coupled
oscillator/integrator system with macro_dt=2.0:
- Lie (frozen): baseline
- Lie (interpolated): ~2.4x error reduction on slow coupling variable
- Strang: ~2.3x error reduction on slow coupling variable

Parameters are JAX arrays, so you can `jax.grad` through an entire
simulation. `build_rhs()` returns a flat `(rhs_fn, keys)` pair — flat-vector
state is what JAX/Diffrax compile fastest, and `flatten`/`unflatten` convert
at the boundary. See [Differentiable calibration of stiff multi-model
composites](#differentiable-calibration-of-stiff-multi-model-composites) for
the real, multi-model version.

---

## Dev Instructions

To add a model, subclass `Process` in `src/hallsim/models/` and wire it
via topology.

---

## Roadmap

Planned work spans the Scheduler (waveform relaxation, IMEX, Mori-Zwanzig
coupling), models & validation (lipid-metabolism extension, trajectory-level
validation, stochastic/Gillespie support, multi-cell communication), and
SBML import (event translation). Full list in
[docs/roadmap.md](docs/roadmap.md).

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

The Scheduler's splitting schemes (Strang splitting, interpolated dense-output coupling, adaptive macro_dt) and the multi-scale roadmap were informed by cross-domain analogies generated using [CrossGen](https://github.com/mohamedAtoui/CrossGen), a structure-mapping tool that finds solutions in unrelated fields. Three CrossGen sessions surfaced techniques from gyrokinetic plasma simulation, partitioned fluid-structure interaction, Mori-Zwanzig projection, waveform relaxation, and federated Kalman filtering — many of which turned out to be established methods in those communities for the exact same multi-scale coupling problem. See [docs/crossgen-suggestions.md](docs/crossgen-suggestions.md) for the full analysis.

## References

1. Cohen, Alan A., et al. "A complex systems approach to aging biology." Nature aging 2.7 (2022): 580-591.
2. Lopez-Otin, Carlos, et al. "Hallmarks of aging: An expanding universe." Cell 186.2 (2023): 243-278.
3. Agmon, Eran, et al. "Vivarium: an interface and engine for integrative multiscale modeling in computational biology." Bioinformatics 38.7 (2022): 1972-1979.
4. Alfego, D., & Kriete, A. (2017). Simulation of cellular energy restriction in quiescence (ERiQ)—a theoretical model for aging. Biology, 6(4), 44.
5. Ptolemaeus, C. (Ed.). "System Design, Modeling, and Simulation using Ptolemy II." Ptolemy.org, 2014. (Heterogeneous models of computation, Director abstraction.)
