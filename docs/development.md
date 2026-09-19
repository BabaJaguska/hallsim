# Development

- `pyproject.toml` is the single source of dependencies.
- A reusable primitive (a coupling edge, a clamp, an event) is a `Process` subclass in `src/hallsim/models/`; a specific biology lives in `demos/models/` with a `build_<name>_composite()` factory.
- Models are aggregated on an additive basis in the ODE RHS via EVOLVED ports. If your model's effect is supposed to be multiplicative, use a separate store path and an INPUT port to read the other variable.

See [CLAUDE.md](../CLAUDE.md) for the load-bearing architecture invariants and the "where to add things" guide.

`make test-docs` runs every Python block in the README, [architecture.md](architecture.md) and [calibration.md](calibration.md) as written, top to bottom per page. Minutes on CPU and needs the network; run it before a release.

## Key files

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
  discovery.py         — search_for_model across BioModels, JWS, ModelDB, BioSimulations, Physiome, Europe PMC
  literature.py        — Europe PMC full text, model pointers, what a cited repository holds
  datasets.py          — search_for_dataset (GEO), platform-table check against the loader
  rejections.py        — the record of deposits screened out, and why
  sbml_core.py, sbml_math.py, sbml_events.py — libsbml -> sympy -> JAX
  sbml_import.py, cps_import.py, xpp_import.py — SBML / COPASI / XPPAUT importers
  sbml_export.py       — Composite.to_sbml
  imported.py          — ImportedODEProcess: time reconciliation, parameter and species inputs
  handles.py           — Handle, ParameterMapping, apply_handles, with_handles; targets_for,
                         suggest_mappings, Intent, suggest_registry (proposals from annotations)
  hallmarks.py         — HALLMARK_INTENTS: the twelve hallmarks in ontology terms, naming no model
  gene_reporters.py    — GeneReporter, MULTI_HALLMARK_REPORTERS, GeneExpressionDataset, GEO fetch
  calibration.py       — Calibrator, CalibrationProblem, Condition, ParameterRef
  identifiability.py   — structural redundancy, fittable-set screen
  stochastic.py        — Gillespie SSA lane
  plotting.py          — figures for a run
  cli.py               — the `simulate` command group
  models/              — reusable primitives: hill_edge, gain_edge, clamp_edge, kick_event,
                         forcing, running_integral, bistable_latch, gated_removal,
                         saturating_removal, observer, neuralode
demos/models/          — specific biology: multi_hallmark (DP14 + GZ06 + Proctor 2007), eriq,
                         stem_cell_niche, hallmarks (HALLMARK_REGISTRY on those models),
                         and the vendored SBML under sbml/
```

## Calibration principles

`hallsim.calibration` wires any composite to any held-out gene-expression
dataset. Three principles are enforced by the framework:

1. **Handle-targeted parameters are not fittable by default.** A handle's severity is the experimental condition (DDIS severity, rapamycin), set per arm, not inferred from data. `Composite.calibration_targets()` subtracts its targets from discovery; `CalibrationProblem.__init__` raises if you pass one as a `ParameterRef`, naming the handle that controls it.
2. **`Process.calibratable_params()` is the self-documenting discovery API.** Each Process declares its own fittable scalars; `Composite.calibration_targets()` aggregates with namespaced names. `SBMLProcess` auto-returns every SBML constant with a published default and a two-OOM clamp — no per-composite hand-curated list anywhere.
3. **Held-out splits are mandatory.** Calibrate on one arm, evaluate on a held-out arm via `problem.evaluate(...)`. Same-data calibrate-and-evaluate is curve-fit, not concordance.
