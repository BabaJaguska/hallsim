# Development

- `pyproject.toml` is the single source of dependencies.
- To add a model, define a new `Process` subclass in `src/hallsim/models/`.
- Models are aggregated on an additive basis in the ODE RHS via EVOLVED ports. If your model's effect is supposed to be multiplicative, use a separate store path and an INPUT port to read the other variable.

See [CLAUDE.md](../CLAUDE.md) for the load-bearing architecture invariants and the "where to add things" guide.

## Key files

```
src/hallsim/
  process.py           — Process base class (Port, PortRole, ProcessKind, calibratable_params)
  store.py             — Store utilities (build, extract, route, validate)
  composite.py         — Composite: topology wiring, auto-grouping, calibration_targets() discovery
  scheduler.py         — The runner: multi-rate orchestration + single-group fast path
  validation.py        — Semantic validation subsystems + analyze_composability
  hallmarks.py         — HallmarkHandle, ParameterMapping (multiplicative transforms), HALLMARK_REGISTRY
  gene_reporters.py    — GeneReporter, MULTI_HALLMARK_REPORTERS, GeneExpressionDataset, cycle_average + last_value summaries
  calibration.py       — Calibrator (forward/reverse autodiff) + CalibrationProblem + Condition + ParameterRef + CalibratableParam (high-level framework)
  sbml_import.py       — process_from_sbml (auto-populates `parameters` from every SBML constant; libsbml functionDefinition inliner; MIRIAM ontology extraction; pre-flight checks)
  plotting.py          — plot_composite_run, plot_runs_comparison, draw_composite_graph, save_run_results
  cli.py               — CLI entry points (simulate command group)
  models/
    multi_hallmark.py     — DP14 + GZ06 + Ihekwaba multi-publication composite (current validation substrate)
    stem_cell_niche.py    — Niche deterioration + Sivakumar 2011 crosstalk
    eriq.py               — ERiQ Energy Restriction in Quiescence (3 Processes)
    saturating_removal.py — Uri Alon damage model
    kick_event.py         — One-shot perturbation EVENT Process
    neuralode.py          — NeuralODE Process + training infrastructure
```

## Calibration principles

`hallsim.calibration` wires any composite to any held-out gene-expression
dataset. Three principles are enforced by the framework:

1. **Hallmark-targeted parameters are not fittable by default.** Hallmarks represent experimental conditions (DDIS severity, rapamycin treatment) — knobs the experimenter set per arm, not biology to be inferred from data. `Composite.calibration_targets()` subtracts them from discovery; `CalibrationProblem.__init__` raises if you pass one as a `ParameterRef`, naming the hallmark that controls it. Escape hatch: `allow_hallmark_override=True`.
2. **`Process.calibratable_params()` is the self-documenting discovery API.** Each Process declares its own fittable scalars; `Composite.calibration_targets()` aggregates with namespaced names. `SBMLProcess` auto-returns every SBML constant with a published default and a two-OOM clamp — no per-composite hand-curated list anywhere.
3. **Held-out splits are mandatory.** Calibrate on one arm, evaluate on a held-out arm via `problem.evaluate(...)`. Same-data calibrate-and-evaluate is curve-fit, not concordance.
