# HallSim architecture

How the framework fits together: the core objects, how processes are wired
and run, the validation layer, how models compose, and how SBML models are
imported. For the calibration/validation side see
[calibration.md](calibration.md); for the multi-rate scheduler internals see
[design-multiscale-scheduler.md](design-multiscale-scheduler.md).

![HallSim Architecture](assets/hallsim_architecture.png)

Built on JAX / Equinox / Diffrax. Composition semantics borrow from Vivarium
and scheduling concepts from Ptolemy II, implemented natively on JAX for GPU
acceleration and differentiability. HallSim deliberately covers a narrower
set of modeling formalisms than Vivarium-collective in exchange for
end-to-end differentiability, JIT, and native batched populations — see
[formalism-coverage.md](formalism-coverage.md).

## Core concepts

| Concept | Description |
|---|---|
| **Process** | `eqx.Module` — declares typed ports and a `kind` (CONTINUOUS / DISCRETE / EVENT). Parameters are JAX arrays: differentiable, JIT-compilable, vmappable. |
| **Port** | Named connection point with a role, default value, units, description, and ontology annotation. |
| **Topology** | Static wiring map `{proc_name: {port_name: store_path}}`, defined at composition time — not inside processes. |
| **Composite** | Bundles processes + topology. `build_rhs()` returns a JAX-compatible flat ODE right-hand side over `store_keys()` order. Auto-groups continuous processes by timescale. |
| **Scheduler** | The unified runner for every composite shape — multi-rate orchestration (timescale groups, discrete dispatch, event firing), single-group fast path, shape-polymorphic state (single or batched `y0`). See [scheduler design](design-multiscale-scheduler.md). |
| **Store** | Flat `dict[str, jnp.ndarray]` with path-like keys (`"cytoplasm/ROS"`). A valid JAX PyTree. |

**Flat-state order.** `store_keys()` is natural-sorted — digit runs compare
numerically, so `net/node2` precedes `net/node10` and a generator's own
numbering survives into the state vector. Align any externally-built,
node-indexed array (a weight matrix, a mask, an observation vector) through
`store_index()`, which maps store path → column. Re-deriving the order instead
misaligns silently: with generated port names there is no error, only wrong
numbers.

**Process kinds:**
- `CONTINUOUS` (default) — `derivative(t, state) -> dy/dt`, solved by Diffrax.
- `DISCRETE` — `update(t, state) -> delta`, called every `dt_step` seconds.
- `EVENT` — `condition(t, state) -> bool` + `handler(t, state) -> delta`, fires on a False→True crossing.

**Port roles:**
- `INPUT` — read-only; the process uses the value but writes no derivative.
- `EVOLVED` — additive; multiple processes' derivatives to the same path are summed. A pure source (contribution independent of the path's own value — e.g. a cross-model edge or a running integral) sets `reads_value=False` so the graph analyzer doesn't infer a spurious feedback cycle.
- `EXCLUSIVE` — sole owner; a second writer raises at composition time.
- `LATCHED` — written by discrete/event processes, read as a constant by continuous processes within a macro step.
- `ASSIGNED` — algebraic, not integrated: the process computes the value each step in `assign(t, state)` and is the path's sole owner. Use it for anything defined by a formula rather than a rate (a ratio, a gate, a quasi-steady-state manifold). `Composite` sorts assignments into dependency order automatically, and `SchedulerResult.get` returns them materialised alongside the integrated states.

**Timescales decide grouping.** `Composite.auto_groups` clusters CONTINUOUS
processes by `proc.timescale`, and `timescale=None` is not "don't care" — it
puts the process in its own `"default"` group, which is then *operator-split*
from everything else. A hand-written coupling edge left at the default is
therefore split away from the SBML module it writes into (imports always carry
`timescale=native_time_seconds`), paying splitting error at every macro-step
boundary with nothing in the output to show for it. Give an edge the same
`timescale` as the module it drives; `models/multi_hallmark.py` is the worked
example.

## Validation layer

On by default, runs at composition time (~ms), warnings-by-default but raises
on hard conflicts (incompatible units, ontology mismatch at a shared path):

| Subsystem | Checks |
|---|---|
| **UnitChecker** | pint dimensional analysis across shared store paths |
| **SemanticChecker** | ontology-ID comparison (ChEBI, GO, SBO, UniProt) for species disambiguation |
| **GraphAnalyzer** | feedback-cycle detection, fan-in, coupling density, unfed-INPUT detection |
| **CouplingAuditor** | heuristic duplicate-reaction detection via description overlap |

Promote warnings to errors with `semantic_validation={"strict": True}`;
disable with `semantic_validation=False`; opt out per subsystem with
`semantic_validation={"check_units": False, ...}`. Numerical-range mismatches
at coupled paths are a *calibration* problem, not a validation one — fit the
rate constant(s) with [`Calibrator`](calibration.md).

## Composing composites

`Composite` accepts other `Composite` instances inside its `processes` dict;
they flatten with namespace prefixes (`outer.sub_proc` for processes,
`outer/path` for store paths). A `rewire={old_path: new_path}` kwarg aliases
overlapping biology onto canonical paths.

```python
from hallsim import Composite, analyze_composability
from hallsim.composite import single_process_composite
from hallsim.sbml_import import process_from_sbml

a = process_from_sbml(582, name="dp14").reconciled_to(86400.0)  # DallePezze 2014, days
b = process_from_sbml(157, name="gz06").reconciled_to(86400.0)  # Geva-Zatorsky 2006, hours -> days
report = analyze_composability(dp14=single_process_composite(a),
                               gz06=single_process_composite(b))  # overlaps + rewire
merged = Composite(processes={"dp14": a, "gz06": b},
                   rewire=report.suggested_rewire)
```

### Merge or couple? — when two models share a node

Two models both have an "NF-κB" — same entity (merge) or distinct (couple)?
It's an identity question, not a biology one: run
`analyze_composability(a=.., b=..)`. A shared ontology ID ⇒ same entity ⇒
**merge** (point both at one store path; `EVOLVED` sums them;
`report.suggested_rewire` → `Composite(rewire=..)`). No ontology match ⇒ take
the conservative choice — **don't merge; add one documented coupling edge** —
and let a held-out [gene-reporter](calibration.md) split score it. Before
adding a model, `hallsim.diagnostics.recommend_coupling_source` checks whether
it even exposes a usable coupling source (a bounded, consumed state) rather
than a dead sink or an unbounded accumulator.

### Driving a model from outside — pick by what the target *is*

An imported model exposes three kinds of target, and each has its own
primitive. Reaching for the wrong one is how a hand-authored one-off gets
written — and a *parameter* change where an *input* change belongs is how an
arm stops being a condition a culture could be in:

| Target | Primitive |
| --- | --- |
| a constant (SBML parameter) | `ImportedODEProcess.with_param_input` — read a store path as the parameter's value each step |
| a species another model owns | `SBMLProcess.with_species_input` — the port keeps its name and ontology and becomes INPUT; wire it to the owner's pool through a level edge carrying the conversion factor (SBML comp's replaced element). Unwired, it holds the published initial value |
| a boundary input (`boundaryCondition` species), dosed then withdrawn | `models.forcing.drive_pulse` — a `PulseSource` on `[t_start, t_end)`, or `t_end=None` to sustain |
| a boundary input held at one level, then another | `models.forcing.drive_step` — a `StepSource`, `before` until `t_step` then `after`; `before == after` is a constant drive |
| a species the model **integrates** | `models.clamp_edge.clamp_species` — a `ClampEdge` holding it at a setpoint |

The third is the one with no obvious workaround: an integrated species is
consumed by the model's own rate laws, so a pulse into it drains away and the
composite can only show an acute response. A `ClampEdge` adds
`k_clamp·(setpoint − target)`, which competes with that consumption rather
than overriding it — additive `EVOLVED` semantics leave no way to preempt
another writer. So the hold is proportional: with net removal flux `v` at the
setpoint, the clamped level settles at `setpoint − v/k_clamp`. Measure `v`
with `measure_unclamped_flux` and pick the rate with `place_clamp_rate`
(which also flags a clamp stiff enough to split off into its own
`auto_groups` group) instead of guessing. `simulate demo clamp` plots all of it.

## SBML import

[`sbml_import.py`](../src/hallsim/sbml_import.py) auto-generates a Process
from an SBML file — from BioModels or a paper supplement — compiling its math
through `hallsim.sbml_core` (libsbml → sympy → JAX), and:

**A repository is not a format.** SBML comes from BioModels, from
BioSimulations' COMBINE archives, and from paper supplements; XPP `.ode` comes
from ModelDB and from supplements — both have importers
(`process_from_sbml`, `process_from_xpp`). CellML/Physiome serves CellML and
ModelDB also serves NEURON, neither of which has an importer, so
`hallsim.discovery` returns those as pointers rather than imports. SED-ML,
which curated deposits ship alongside the model, is a different kind of
artefact again — it describes a *simulation experiment over* a model, not the
model — see [roadmap.md](roadmap.md).

The importer:

- auto-populates every SBML constant into `SBMLProcess.parameters`, so the
  full mechanism surface is discoverable via `Composite.calibration_targets()`;
- inlines `<functionDefinition>` blocks (via libsbml), unlocking the majority
  of curated models that would otherwise hit "Custom functions are not
  handled" upstream;
- translates `<event>` blocks (see `hallsim.sbml_events`), including those
  whose assignment target is a parameter rather than a species — the target is
  promoted onto the owning process via `with_param_input` so the assignment
  reaches the rate laws. A nonzero delay or a priority is still refused, and a
  zero delay is not a delay (COPASI writes `<delay>0</delay>` on every export).
  **A `Composite` expands a member process's events automatically**; discarding
  them is `proc.without_events()`, written at the call site so the discard is
  visible where it is decided. `intake.triage_sbml` rejects two trigger
  pathologies before import: complementary triggers sharing a boundary, which
  make the outcome depend on round-off at the crossing, and equalities against
  time, which make the scheduler's `macro_dt` decide whether the event fires at
  all;
- extracts MIRIAM annotations into `Port.ontology` from species CVTerms;
- freezes inert sinks (species reactions write and nothing reads) at their
  initial value so a degradation counter cannot grow without bound. A
  terminal product looks the same to that test: `proc.frozen_species()`
  names them, a composite lifts the freeze for any the wiring reads, and
  reading a still-frozen path from a result warns and names
  `proc.with_unfrozen(...)`;
- records provenance, `proc.provenance()`: the source asked for, the file
  read and its SHA-256, the native clock and its reconciliation, the frozen
  species, and every parameter changed from the deposit. A calibration
  run's `config.json` and `Composite.to_sbml()` carry it.

Discover-then-import is two calls — the catalog is directly usable by an agent:

```python
from hallsim.discovery import search_for_model
from hallsim.sbml_import import process_from_sbml

# Every registered repository at once: BioModels, JWS Online, ModelDB,
# BioSimulations and Europe PMC supplements; sources=[...] narrows it.
hits = search_for_model("genotoxic stress NFkB")   # -> [ModelCandidate, ...]
proc = process_from_sbml(hits[0].id, name="dna_nfkb")   # fetch + generate
```

A BioModels ID downloads to `~/.cache/hallsim/biomodels/` on first import.
The demos' own models ship under
[`demos/models/sbml/<author><year>/`](../demos/models/sbml/) and are loaded
by path.

### On-disk caches

Three, all under `~/.cache/hallsim/`, all safe to delete:

| path | holds | keyed on |
|---|---|---|
| `biomodels/` | SBML downloaded by ID | the BioModels ID |
| `converted/` | function-expanded and event-stripped SBML | the source file's size and mtime |
| `jax/` | XLA's compiled executables, reused across processes | JAX's own hash of the computation |

Writes are atomic, so concurrent processes never read a partial file. Set
`HALLSIM_COMPILATION_CACHE_DIR` to relocate the compile cache, or to `off` to
disable it — which is what you want when timing a cold compile.

## Perturbation handles

A handle is a named severity that moves parameters across one or more
processes, differentiable end-to-end (`hallsim.handles`). Transforms are
**multiplicative of the current calibrated base value** — a transform gets
`(severity, base)` and returns `base * f(severity)` — so `Calibrator` can fit
mechanism parameters and then apply severities without the handle
clobbering the fit. A registry is a plain `{name: Handle}` dict and every
call names the one it uses. The demos ship the hallmarks of aging as one,
`demos.models.hallmarks.HALLMARK_REGISTRY`, mapped onto the demo models; a
drug, a gene dosage or any other perturbation is another `Handle` in a
registry of its own, applied the same way. An experimental arm is a choice
of severities over one composite.

A registry is written, not resolved: the hallmarks are also described once
without naming any model, as intents in ontology terms
(`hallsim.hallmarks.HALLMARK_INTENTS`: the species by UniProt, ChEBI or GO
id, its role in the reactions to scale, the gain), and
`hallsim.handles.suggest_registry(intents, composite)` proposes named
mappings for any composite from what its members annotate. `simulate
handles <id-or-path>` prints that table for a model, including where
nothing is annotated the way an intent asks. The proposal is reviewed and
kept as a file, and the file is what gets applied.

```python
from hallsim.handles import with_handles
from demos.models.hallmarks import HALLMARK_REGISTRY
from demos.models.multi_hallmark import build_multi_hallmark_composite

base = build_multi_hallmark_composite()
# Rapamycin = downward shift on Deregulated Nutrient Sensing (targets DP14's
# mTORC1 phosphorylation rate): +1 is full dysregulation, -1 is rapamycin.
treated = with_handles(base, {"Deregulated Nutrient Sensing": -1.0},
                       registry=HALLMARK_REGISTRY)
```

`with_handles` keeps the topology; `apply_handles(processes, {...})` is
the same transform on a bare process dict.

**Pharmacological interventions belong on the handle layer they perturb**,
not as separate Processes. **Cross-model coupling is mediated at the
experimental-condition level where possible**: e.g. Genomic Instability drives
both DP14's `DNA_damaged_by_irradiation` and GZ06's `psi` at each model's own
scale — same severity knob, no state-into-constant patching between the SBML
models (a foot-gun the framework deliberately doesn't expose).

## Example composites

HallSim is a *framework*, not a model library — bring your own Processes
(hand-written, SBML-imported, or a `NeuralODE`).

Reusable primitives — no domain content, and the part that is under test —
ship under [`src/hallsim/models/`](../src/hallsim/models/):
[`saturating_removal.py`](../src/hallsim/models/saturating_removal.py) (Uri
Alon damage motif), [`hill_edge.py`](../src/hallsim/models/hill_edge.py),
[`clamp_edge.py`](../src/hallsim/models/clamp_edge.py),
[`kick_event.py`](../src/hallsim/models/kick_event.py) (one-shot EVENT
perturbation), [`forcing.py`](../src/hallsim/models/forcing.py),
[`running_integral.py`](../src/hallsim/models/running_integral.py),
[`bistable_latch.py`](../src/hallsim/models/bistable_latch.py) and
[`neuralode.py`](../src/hallsim/models/neuralode.py).

Specific biology lives in [`demos/models/`](../demos/models/) and is **not**
part of the package — [`multi_hallmark.py`](../demos/models/multi_hallmark.py),
[`eriq.py`](../demos/models/eriq.py),
[`mitochondrial_aging.py`](../demos/models/mitochondrial_aging.py),
[`stem_cell_niche.py`](../demos/models/stem_cell_niche.py).

**These are exercises of the framework's mechanics, not results.** They exist to
stress the paths this document describes — several time bases reconciled onto
one clock, cross-publication coupling edges, multi-group solves, held-out arms,
and end-to-end gradients through all of it. No claim about the framework rests
on a demo's concordance score, and none should be built on one as biology.
Which models a demo currently composes, and why, belongs to that demo's own
docstring and to `docs/known-problems.md` — not here, where it goes stale
silently.

## Population studies via batched `y0`

The Scheduler's state pipeline is shape-polymorphic — a `(batch, n_vars)` y0
flows through every group's Diffrax solve as one batched computation, no
`jax.vmap` over `Scheduler.run`:

```python
import jax.numpy as jnp
from hallsim.scheduler import Scheduler
from demos.models.multi_hallmark import build_multi_hallmark_composite

comp = build_multi_hallmark_composite()
y0 = comp.initial_state_vec()                        # (n_vars,)
y0 = jnp.broadcast_to(y0, (64, y0.shape[0]))         # (64, n_vars)
y0 = y0.at[..., comp.store_index()["dp14/DNA_damage"]].set(
    jnp.linspace(0.0, 10.0, 64))
result = Scheduler().run(comp, t_span=(0.0, 50.0), macro_dt=5.0, y0=y0)
result.get("dp14/CDKN1A").shape                      # (n_time, 64)
```

Near-flat in `batch` on GPU (kernel launch dominates) is the design intent
and is unmeasured. On CPU it is measured and it is not what the design
intends: on the 79-state multi-hallmark composite over 14 days, per-member
cost is 329 ms at 64 members, 442 ms at 256 and 707 ms at 1024, at which
point one batched run is slower than 1024 sequential ones. One vmapped
solver loop has one trip count, so every member steps as many times as the
slowest, and a 10% jitter on the initial condition spreads adaptive step
counts 1.6–3.6× between members. For launched populations whose dynamics
tolerate a validated fixed step, `Scheduler(fixed_dt=...)` provides lockstep
integration; it replaces error control, so compare its readouts with an
adaptive reference first. Otherwise, run populations in chunks of about 64.
Every process kind rides the
batch axis: a discrete `update` and an event `condition`/`handler` see
`(batch,)` per port, a delta may be a scalar for every member or one per
member, and an event fires for exactly the members whose condition just
turned True — `EventRecord.members` is that mask.

Batched `y0` broadcasts *initial conditions*. Varying a **parameter** across a
batch — a hallmark severity sweep, for instance — changes the process pytree
rather than the state vector, so it is not a `y0` batch; build one composite
per arm.

## Supporting modules

Small modules a model author needs early, each importable on its own:

| Module | What it gives you |
|---|---|
| `hallsim.kinetics` | `hill_gate`, `hill_inhibition` and friends — the saturating forms every coupling edge needs. Reach for these instead of hand-rolling `x**n / (K**n + x**n)`. |
| `hallsim.io` | `outdir` and `make_run_dir` — the output convention every demo follows (timestamped run directory plus a `latest` symlink). |
| `hallsim.bifurcation` | `equilibrium`, `spectrum`, `codim1_scan` — continuation and stability analysis around a fixed point. `codim1_scan` finds both codimension-1 crossings, Hopf (oscillation onset) and fold (bistability, an invasion threshold), each with its normal-form coefficient. Pass `laws=` for any model with a conserved moiety, or the Newton is singular at every state. Continuation is plain Newton, so a branch is followed only until it folds — tracing both arms of a hysteresis loop needs a multi-seed sweep. |
| `hallsim.stiffness` | `analyze_groups(composite, *, y0, groups, t0, dt)` — per-group spectral abscissa, Jacobian condition number and state-scale spread, with the solver verdict. Keyword-only. |
| `hallsim.structure` | What the declared symbolic forms (`reaction_channels`, `assignment_rules`, `rate_rules`) imply for a whole composite: `composite_stoichiometry` / `composite_moieties` (exact `N` and its integer moieties over store paths), `jacobian_pattern` + `compressed_jacobian` (the Jacobian in as many forward passes as its sparsity has colours; dense only on an undeclared process's own block), `check_pattern` (the pattern against the composite's derivative), `symbolic_field` (the field as sympy, over path and `<process>.<field>` parameter symbols). `steady_state` and `identifiability.structural_redundancy` are built on it. |
| `hallsim.diagnostics` | `screen_process` / `screen_composite` (the constituents-first pre-flight), `screen_sensitivity`, and `recommend_coupling_source`. |
