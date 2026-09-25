# HallSim architecture

How the framework fits together: the core objects, how processes are wired
and run, the validation layer, how models compose, and how SBML models are
imported, plus the [Scheduler](#scheduler)'s macro-step semantics,
[formalism coverage](#formalism-coverage) and the [key files](#key-files).
For the calibration side see [calibration.md](calibration.md).

![HallSim Architecture](assets/hallsim_architecture.png)

Built on JAX / Equinox / Diffrax. Composition semantics borrow from Vivarium
and scheduling concepts from Ptolemy II, implemented natively on JAX for GPU
acceleration and differentiability. HallSim deliberately covers a narrower
set of modeling formalisms than Vivarium-collective in exchange for
end-to-end differentiability, JIT, and native batched populations — see
[Formalism coverage](#formalism-coverage).

## Core concepts

| Concept | Description |
|---|---|
| **Process** | `eqx.Module` — declares typed ports and a `kind` (CONTINUOUS / DISCRETE / EVENT). Parameters are JAX arrays: differentiable, JIT-compilable, vmappable. |
| **Port** | Named connection point with a role, default value, units, description, and ontology annotation. |
| **Topology** | Static wiring map `{proc_name: {port_name: store_path}}`, defined at composition time — not inside processes. |
| **Composite** | Bundles processes + topology. `build_rhs()` returns a JAX-compatible flat ODE right-hand side over `store_keys()` order. Auto-groups continuous processes by timescale. |
| **Scheduler** | The unified runner for every composite shape — multi-rate orchestration (timescale groups, discrete dispatch, event firing), single-group fast path, shape-polymorphic state (single or batched `y0`). See [Scheduler](#scheduler). |
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

## Scheduler

The Scheduler is the one runner. It borrows Vivarium's composition
semantics (Process, Port, Topology, Store; Agmon et al. 2022) and Ptolemy
II's idea of heterogeneous models of computation under one orchestrator,
and implements both natively on JAX:

| Capability | Vivarium | Ptolemy II | HallSim |
|------------|----------|------------|---------|
| GPU-accelerated continuous solves | No (CPython) | No (Java) | Yes (JAX/Diffrax) |
| Differentiability through ODE solves | No | No | Yes (`jax.grad`) |
| Population parallelism | multiprocessing | Threads | batched `y0` |
| Composition-time validation | Basic | No | 4-subsystem semantic layer |
| Heterogeneous process types | Yes (Engine) | Yes (Directors) | Yes (Scheduler) |

**The macro step.** `Scheduler.run(composite, t_span, macro_dt)` advances in
communication intervals of `macro_dt`. Within one:

1. each continuous timescale group is solved by Diffrax over the interval,
   groups in sequence under Lie splitting (`splitting="strang"` for the
   symmetric second-order variant), the other groups' states supplied
   frozen or interpolated (`coupling_mode`);
2. discrete processes whose `dt_step` is due fire `update`, and their
   deltas are added;
3. event conditions are checked at the sync point, and a handler fires
   once on a False→True crossing.

The whole loop compiles to one `lax.scan`, so reverse-mode gradients flow
through every macro step and across timescale groups; only event handlers
are non-differentiable, which matches the biology they represent. A
composite with one continuous group and no discrete, event or Strang
machinery takes a fast path: one `diffeqsolve` over the whole span.

**Choosing `macro_dt`.** It must not exceed the smallest discrete
`dt_step`; it should be small enough that LATCHED values do not go stale in
ways that matter, and large enough that orchestration is negligible against
the solve. `min(dt_step) / 2` is the usual starting point, and
[benchmarks.md](benchmarks.md) §7 measures the splitting error against the
macro step on a feedback loop and on the multi-hallmark composite.

**Stiffness routing.** Each group is analysed at `y0`
(`hallsim.stiffness.analyze_groups`) and routed to an implicit solver
(`Kvaerno5` with a chord root finder) or an explicit one (`Tsit5`); a group
may be pinned. A wrong pin costs 14× on a stiff group and 22× on a non-stiff
one ([benchmarks.md](benchmarks.md) §9). Two cases the verdict at `y0`
cannot see are handled at run time: a Jacobian that is not finite at the
initial state (a square root or a fractional power of a state that starts
at zero) routes the group explicit, since Newton could not use that
Jacobian either; and a failed solve climbs a ladder. A group whose explicit
solve fails — a model that is not stiff at `y0` and turns stiff along its
trajectory — is re-run on the implicit solver; a stiff group runs the
fifth-order implicit solver under a step budget (`LADDER_STEP_BUDGET`) and,
if it runs out, is re-run on `Kvaerno3` with diffrax's per-step chord, which
is what a very stiff system at a loose tolerance needs. The verdict a group
ends on is kept for the composite, so the next run takes it directly.

**Validation for the multi-timescale contract**, run at composition time
alongside the semantic layer:

| Check | Severity |
|-------|----------|
| Continuous process writes to a LATCHED port | error |
| Discrete/event process writes to an EVOLVED port | error |
| `macro_dt` larger than the smallest discrete `dt_step` | error |
| LATCHED port has no discrete/event writer | warning |
| Timescale ratio within a group above 100× | warning |
| Discrete `dt_step` not aligned with `macro_dt` | warning |

**A multi-timescale cell**, in outline: fast ROS kinetics (`timescale=1.0`)
and slow epigenetic drift (`timescale=86400*30`) auto-group apart and are
solved at their own step sizes; a `CellDivision` discrete process with
`dt_step=86400.0` fires once a day; a `SenescenceEntry` event process latches
`cell/senescent` when its `p53` input crosses a threshold. One
`Scheduler().run(composite, t_span=(0, 86400*365), macro_dt=3600.0)` runs the
year, syncing the store every hour. `simulate demo multiscale` is the
runnable version on toy processes.

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
  reaches the rate laws. Delayed events and event priorities are not translated
  yet; a zero delay is not a delay (COPASI writes `<delay>0</delay>` on every
  export).
  **A `Composite` expands a member process's events automatically**; discarding
  them is `proc.without_events()`, written at the call site so the discard is
  visible where it is decided. `intake.triage_sbml` rejects two trigger
  pathologies before import: complementary triggers sharing a boundary, which
  make the outcome depend on round-off at the crossing, and equalities against
  time, which make the scheduler's `macro_dt` decide whether the event fires at
  all;
- carries a compartment whose size a rate rule or an assignment rule sets:
  values are amounts throughout, so a moving volume changes what a rate law
  reads and never the stoichiometric balance, and a rate rule on a
  concentration in such a compartment gains the dilution term. Checked
  against libRoadRunner on Schaber 2012 (rate rule) and Zi 2011 (assignment
  rule) to better than one part in ten million;
- gives a species or parameter with no initial value the zero that
  libRoadRunner and COPASI give it, with a warning naming it, and ignores
  the value attribute on a rule's target, as SBML specifies;
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

When nothing is deposited, `simulate discover <topic>` (`hallsim.web_discovery`)
searches Europe PMC for model papers, reads them and their linked PDFs for
repository links, and classifies the repositories cited: `importable:<format>`,
`source:<language>`, `organisation`, or `linked-unverified` for a pointer not
inspected. `--alias` and `--mechanism` add queries verbatim; `--url` seeds a
paper or repository; `--web` adds Brave Web Search from `BRAVE_SEARCH_API_KEY`,
and any `provider` with `search(query, *, limit, timeout)` plugs in the same
way. PDF text needs the `search` extra. A label describes filenames, not the
model: screen anything selected with `simulate screen`.

**The supply, measured.** `simulate census run` puts every SBML deposit in
BioModels — curated and uncurated, or one branch with `--branch` — through
the same gate as `simulate screen`, in parallel with a per-deposit timeout,
streaming one row per deposit; `simulate census report` writes the funnel
(listed → kinetic → imports → solves → clock → annotated → at rest → clean)
per branch, a salvage class per deposit
(`hallsim.census.SALVAGE`: as-is, cheap-fix, needs-review, importer-work,
wrong-formalism, deposit-defect, framework-defect, timeout) with the
concrete action beside it, the per-deposit failure table carrying each
paper's own account of what it models, two figures, and a write-up with a
preprint paragraph. It is the numerical gate only: nothing in it reads the
paper, so its counts are upper bounds on the usable supply. Every row
carries the HallSim version and commit it was screened under and when.
When the run is the whole census — every listed deposit screened, every
row stamped — the report also copies the table and the counts to
`results/census/`, which is tracked, so a diff between two commits names
the deposits a change lifted or broke; a probe or a partial run leaves
that table alone.

**The data side.** `simulate census-data run` enumerates the repositories
the same way: every GEO series for the organisms, every PRIDE,
MetaboLights, Metabolomics Workbench and ArrayExpress entry, the BioImage
Archive, and each screened model's own paper through Europe PMC (its
flags, the datasets it links, its supplement). Every dataset meets four
nested gates: timed (three or more timepoints, read from the sample
titles, a declared time factor or the description), measured (quantities
that can be named: a transcriptome, a proteome, listed metabolite ids),
matched (a screened model carries one of them, by ontology: a shared
ChEBI id, a protein in a proteome, a transcription factor a transcriptome
reads through its regulon) and loadable (a reader exists for its tables).
Arms, a named control and the perturbation labels are recorded, not gated:
an unperturbed time course is data. `census-data report` writes the funnel
per route and modality, the direct pair list and the paper census;
`census-data rescreen` re-judges stored rows under the current gates and
model set without asking the repositories again.

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
processes, differentiable end-to-end (`hallsim.handles`). Each mapping is
`floor + slope * severity`, read against the parameter's kind: a **rate**
is scaled relative to its current value, so `Calibrator` can fit mechanism
parameters and then apply severities without the handle clobbering the
fit; an **input level** (`calibratable(..., level=True)`, as a forcing
source's amplitude is) rests at 0 and is set by severity, with the
magnitude on the source as its `dose`. Severity 0 is neutral for both, and
a composite with no handle applied is at neutral. A registry is a plain `{name: Handle}` dict and every
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
the same operation on a bare process dict.

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

These demos exercise the paths this document describes — several time bases
reconciled onto one clock, cross-publication coupling edges, multi-group
solves, held-out arms, and end-to-end gradients through all of it. Which
models a demo composes, and why, is in that demo's docstring.

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

On GPU the batch is close to flat: on a Tesla T4, 256 DallePezze 2014 cells
cost 4.6× one cell, and 256 members of the 79-state multi-hallmark composite
11.7× one. On CPU one vmapped solver loop has one trip count, so every member
steps as many times as the slowest, and the efficient unit is a chunk of
about 64 members (329 ms per member on that composite over 14 days, against
528 ms solved one at a time); [benchmarks.md](benchmarks.md) §6 has the
numbers and the worker-process pattern for larger populations. For
populations whose dynamics tolerate a validated fixed step,
`Scheduler(fixed_dt=...)` provides lockstep integration; it replaces error
control, so compare its readouts with an adaptive reference first.
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
| `hallsim.attenuation` | `trace_path(composite, control, reporter, ...)` — follows a handle or a parameter to a reporter through the wiring, runs the composite at two settings of it, and reports the relative change at every store path on the route, naming the node where it collapses and the reactions that carry that step. The diagnosis behind a flat reporter or a structural verdict; the identifiability report points here. |
| `hallsim.view` | `page_for(composite, registry, t_end=...)` + `serve(page)` (`simulate view module:name`): a Dash page for any composite with a levers tab (one slider per handle that reaches it, a trajectory row per process, a population band for reaction-level members), a wiring tab (processes opened into their reactions and states, a `trace_path` route coloured by relative change) and, when `--run` names one, a fit tab (a saved calibration run's history, parameters and concordance). The hallmark-lever demo is one `Page` over it. Needs the `app` extra. `bake(page, dir, step=0.25)` (`--bake DIR` on `simulate view` and on the lever demo) writes the levers as a static site instead: every slider setting on a grid, solved once, read back by plain HTML and JS from any static host. |

## Formalism coverage

HallSim covers a chosen subset of the modelling formalisms a "multiscale
model" slide lists, and the choice is the design: the composition contract
(`derivative` / `update` / `condition` + `handler`, all over JAX arrays) is
narrower than Vivarium's any-Python-callable `update(t, dt)`, and in
exchange the whole composite is differentiable, JIT-compilable, GPU-runnable
and natively batched. For aging biology — signaling networks, metabolic
ODEs, oscillators, stem-cell niches — that is the right trade; for
genome-scale FBA or molecular dynamics it is not, and those compose at the
boundary as INPUT ports fed from their own tools.

Of Milner's bigraphs, which the Vivarium papers cite, HallSim keeps the link
graph (`topology = {process: {port: store_path}}`) and drops the place graph:
state is one flat `dict[str, jnp.ndarray]` with `/`-separated keys, not a
nested hierarchy. Homogeneous populations are a batched `y0`; paracrine
coupling is a process that reduces along the batch axis and writes a shared
store path every cell reads; a composite reused under two names nests inside
another and flattens with a prefix.

| Formalism | Status | Notes |
|---|---|---|
| Differential equations | ✅ | native CONTINUOUS Process via Diffrax |
| Michaelis-Menten, Monod-Wyman-Changeux, linear degradation | ✅ | ODE forms of the same machinery |
| Neural network | ✅ | `NeuralODE` Process with its training path (`hallsim.models.neuralode`) |
| SBML — deterministic ODE | ✅ | `process_from_sbml`, native (`hallsim.sbml_core`); `scripts/conformance.py` compares an import with libRoadRunner and COPASI |
| SBML — events | 🟢 | `hallsim.sbml_events`, including assignments to parameters; delayed events and priorities are roadmap |
| Gillespie / stochastic | 🟢 | `hallsim.stochastic` runs an imported reaction network at reaction level inside a composite, with a threaded batch lane; PRNG plumbing for hand-written stochastic DISCRETE processes is roadmap |
| Boolean network | 🟡 | a DISCRETE update over logical ops; no model written yet |
| Rule-based (BNGL / Kappa) | 🟡 | a CONTINUOUS Process emitting an expanded ODE system; not built |
| Agent-based / multi-cellularity | 🟡 | batched `y0` gives N independent cells; inter-cell communication needs a `PopulationAggregate` — [roadmap.md](roadmap.md) |
| Constraint-based (FBA / BiGG) | ❌ | needs an LP solver; doable via `jaxopt` and queued behind an application |
| Molecular dynamics, Brownian dynamics, physics engines | ❌ | wrong scale or no spatial state; OpenMM / GROMACS |
| Graphical / Bayesian networks | ❌ | a different paradigm; wrap behind a Process if needed |

✅ native or reduces to something native, with working models; 🟢 built
and exercised; 🟡 the abstractions support it and a canonical example is
mostly Process authoring; ❌ outside the design envelope, composed at the
boundary.

## Development

- `pyproject.toml` is the single source of dependencies.
- A reusable primitive (a coupling edge, a clamp, an event) is a `Process`
  subclass in `src/hallsim/models/`; a specific biology lives in
  `demos/models/` with a `build_<name>_composite()` factory.
- Models aggregate additively in the ODE RHS via EVOLVED ports. A
  multiplicative effect reads the other variable through a separate store
  path and an INPUT port.
- Handle-targeted parameters are not fittable by default (a handle's
  severity is the experimental condition, set per arm);
  `Process.calibratable_params()` is the self-documenting discovery API;
  held-out splits are mandatory — see [calibration.md](calibration.md).
- `make test-docs` runs every Python block in the README, this page and
  calibration.md as written, top to bottom per page; minutes on CPU and
  needs the network, so run it before a release.
- The version is the git tag (setuptools-scm): `0.2.0` at tag `v0.2.0`,
  `0.2.1.devN+g<hash>` N commits past it; `simulate --version` prints it.
  A release is `git tag -a vX.Y.Z -m "..."` and a push of the tag: the
  release workflow runs the suite, builds the wheel and the source
  distribution from the tag and attaches them to a GitHub Release.
- The public lever page is the baked site, served at demo.hallsim.org as
  static files on Cloudflare (`wrangler.jsonc`): `simulate demo
  hallmark-levers --bake outputs/site` (about an hour at the default grid
  and 16 cells), then `npx wrangler deploy`.

### Key files

```
src/hallsim/
  process.py           — Process base class (Port, PortRole, ProcessKind), read_param / write_param
  store.py             — flat store: build, extract, route, validate
  composite.py         — Composite: topology wiring, auto-grouping, build_rhs, calibration_targets()
  scheduler.py         — the one runner: multi-rate orchestration, batched lanes, single-group fast path
  root_finders.py      — the Chord root finder the Scheduler installs into implicit solvers
  stiffness.py         — per-group spectral verdict and solver routing
  structure.py         — stoichiometry, moieties, Jacobian sparsity, symbolic field
  steady_state.py      — Newton to a fixed point of a composite
  bifurcation.py       — equilibrium, spectrum, codim-1 continuation
  validation.py        — unit / semantic / graph / coupling checks, analyze_composability
  diagnostics.py       — screen_process / screen_composite, coupling-source verdicts
  intake.py            — triage_sbml, published_fit_chi2
  census.py            — the corpus census: every deposit through the gate, and the report
  discovery.py         — search_for_model across BioModels, JWS, ModelDB, BioSimulations, Physiome, Europe PMC
  literature.py        — Europe PMC full text, model pointers, what a cited repository holds
  datasets.py          — search_for_dataset (GEO, Zenodo, PRIDE, MetaboLights, Metabolomics Workbench, ArrayExpress, BioImage Archive), parse_design, a paper's own data, coverage of a composite
  dataset_census.py    — the data census: every deposited time course through timed → measured → matched → loadable
  rejections.py        — the record of deposits screened out, and why
  sbml_core.py, sbml_math.py, sbml_events.py — libsbml -> sympy -> JAX
  sbml_import.py, cps_import.py, xpp_import.py — SBML / COPASI / XPPAUT importers
  sbml_export.py       — Composite.to_sbml
  imported.py          — ImportedODEProcess: time reconciliation, parameter and species inputs
  handles.py           — Handle, ParameterMapping, apply_handles, with_handles; suggest_registry
  hallmarks.py         — HALLMARK_INTENTS: the twelve hallmarks in ontology terms, naming no model
  gene_reporters.py    — GeneReporter, MULTI_HALLMARK_REPORTERS, GeneExpressionDataset, GEO fetch
  calibration.py       — Calibrator, CalibrationProblem, Condition, FitParam
  identifiability.py   — structural redundancy, fittable-set screen
  stochastic.py        — Gillespie SSA lane
  attenuation.py       — trace_path: follow a control to a reporter through the wiring
  view/                — the Dash page: levers, wiring, fit tabs
  plotting.py          — figures for a run
  cli.py               — the `simulate` command group
  models/              — reusable primitives: hill_edge, gain_edge, clamp_edge, kick_event,
                         forcing, running_integral, bistable_latch, gated_removal,
                         saturating_removal, observer, neuralode
demos/models/          — specific biology: multi_hallmark (DP14 + GZ06 + Proctor 2007), eriq,
                         stem_cell_niche, hallmarks (HALLMARK_REGISTRY on those models),
                         and the vendored SBML under sbml/
```
