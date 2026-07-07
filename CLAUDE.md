# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Project venv is `.venv_hallsim/` (note the non-standard name). Use `.venv_hallsim/bin/python` for ad-hoc runs.
- Python >=3.9. Core stack: JAX + Equinox + Diffrax + Optax. Validation uses `pint` (units) and `networkx` (graph analysis). SBML import uses `sbmltoodejax`.
- `pyproject.toml` is the single source of truth for dependencies. No lockfiles — `make install` / `make install-dev` editable-install directly from pyproject.

## Common commands

```bash
# Install
make install        # editable install of runtime deps from pyproject.toml
make install-dev    # dev deps + editable install

# Tests
make test
.venv_hallsim/bin/python -m pytest tests/test_eriq_composable.py -v
.venv_hallsim/bin/python -m pytest tests/test_multiscale.py -v -k scheduler

# Lint / format (line length 79, black-managed; flake8 ignores E501,E402,W504,W503,E226,E203)
make format
make lint

# Run demos via the `simulate` entry point (registered in pyproject.toml -> hallsim.cli:simulate)
simulate compose | compose-kick | multiscale | validate-demo | info
simulate validate-demo --strict        # warnings -> errors

# Demos that aren't CLI commands run as scripts:
.venv_hallsim/bin/python demos/multiscale_coupling_demo.py
```

## Architecture invariants

The README has the full overview. The points below are the load-bearing rules — break them and composition fails silently or at validation time.

**Store has path-like keys, no hierarchy.** Dict shape at API boundaries is `dict[str, jnp.ndarray]` with keys like `"cytoplasm/ROS"`. The `/` is convention, not nesting — there is no nested dict.

**Port roles encode write semantics (validated at composition time in `Composite.__init__`):**
- `EVOLVED` — additive. Multiple processes writing the same store path get summed in the RHS.
- `EXCLUSIVE` — sole owner. A second writer to the same path raises in `validate_topology`.
- `INPUT` — read-only. Process reads but does not contribute a derivative.
- `LATCHED` — written only by DISCRETE/EVENT processes; CONTINUOUS processes read it as a constant within a macro step. Writing LATCHED from a CONTINUOUS process is a contract violation.

**Topology lives outside processes.** Wiring is `{proc_name: {port_name: store_path}}` passed to `Composite`. Don't put store paths inside Process classes.

**`composite.build_rhs(proc_names=None) -> (rhs_fn, keys)`.** Flat by design — operates on a 1-D `jnp.ndarray` in `sorted(store_paths())` order so JAX traces a single array under JIT/grad. Pair with `composite.flatten()` / `composite.unflatten()` to convert at API boundaries. `proc_names=None` (default) uses every CONTINUOUS process; passing an explicit list builds a partial RHS for operator splitting (the Scheduler does this per group). State flows as a flat vector throughout; dict shape only appears at API boundaries and inside the small per-process port-name dict that `Process.derivative` consumes.

**Process kinds drive scheduler dispatch:**
- `CONTINUOUS` (default) — implements `derivative(t, state)`, solved by Diffrax.
- `DISCRETE` — implements `update(t, state)`, fired every `dt_step` seconds.
- `EVENT` — implements `condition(t, state)` and `handler(t, state)`; handler fires on False→True crossing at sync points.

The Scheduler is the only runner. There is no separate `Simulator` class. For single-group continuous composites with no events / discrete / adaptive_dt / Strang / interpolated, the Scheduler takes a fast path that issues one `dfx.diffeqsolve` over the whole `t_span` (no per-macro-step overhead).

**Timescale auto-grouping.** `Composite.auto_groups(max_ratio=100.0)` clusters CONTINUOUS processes by `proc.timescale`; processes within 100x of each other share a Diffrax solve. Set `timescale` on processes with very different rates (signaling vs damage accumulation) so the Scheduler doesn't force a stiff integrator on all of them.

**Splitting / coupling are independent Scheduler knobs:**
- `splitting`: `"lie"` (default, O(dt)) vs `"strang"` (symmetric half-step, O(dt²)).
- `coupling_mode`: `"frozen"` (default snapshot at sync) vs `"interpolated"` (dense-output query).
- `adaptive_dt=True`: PLL-inspired step control on coupling residual.
- Strang + interpolated are mutually exclusive and rejected in `Scheduler.__init__` — Strang's reverse pass would need each prior group's interpolant over the second half-step, but that interpolant has not been produced yet (the group runs after, not before). Fixable via predictor-corrector or Picard iteration; not implemented.

**Hallmarks are immutable parameter modifiers.** `apply_hallmarks(processes, {hallmark: severity})` returns a *new* dict of processes; it doesn't mutate. Severity is differentiable end-to-end — you can `jax.grad` through hallmark severity to a downstream loss.

**Validation layer is on by default and warning-by-default.** `Composite(...)` runs UnitChecker / SemanticChecker / GraphAnalyzer / CouplingAuditor (~ms for realistic composites). Unit/ontology *conflicts* raise; everything else (cycles, fan-in, unfed inputs, unit-scale mismatches) emits Python warnings. Disable with `semantic_validation=False`; promote all warnings to errors with `semantic_validation={"strict": True}`. Topology validation (`validate=True`, default) is always errors.

## Where to add things

- **New model**: `src/hallsim/models/<name>.py`. Subclass `Process`, declare `ports_schema()`, implement `derivative` (or `update` / `condition`+`handler`). Provide a `build_<name>_composite()` factory that wires processes via topology — the established pattern in `eriq.py` and `stem_cell_niche.py`.
- **Multiplicative coupling**: model effects are summed via EVOLVED ports. For multiplicative coupling, route the modulating variable through a separate store path and read it via an `INPUT` port.
- **SBML model**: import via `hallsim.sbml_import.process_from_sbml(...)` — wraps `sbmltoodejax`. Pre-imported BioModels live under `models/sivakumar2011/`.
- **Hallmark mapping**: `hallsim.hallmarks.apply_hallmarks` is the public surface. The Stem Cell Exhaustion → Sivakumar2011 path via `StemCellNiche` is the reference example.
- **Tests**: `tests/test_composition.py` (Process/Port/Topology contracts), `test_multiscale.py` (Scheduler/splitting/coupling), `test_eriq_composable.py` (full-model patterns), `test_validation.py` (semantic layer), `test_models.py` (per-model regression).

## Notes for editing

- Process subclasses are `eqx.Module` — fields are JAX-traced. New parameters must be declared as class attributes with type annotations or they won't survive `eqx`/JIT.
- Keep duplicated algebraic intermediates inside Processes (the ERiQ pattern). Sharing them via the store crosses module boundaries and breaks independent testability — the README is explicit about preferring duplication of cheap arithmetic.
- `NeuralODE` ships with training infrastructure, not just a dynamics module — see `hallsim.models.neuralode` for the optimizer/loss harness.
- The CLI is registered as `simulate = "hallsim.cli:simulate"` (a click group). Adding a demo means adding a `@simulate.command()` in `cli.py`.


## User preferences
- The repo is all about modularity, composability and jax-based speedups and backprop. This is an absolute must. If something makes things slow, or uncomposable, or monolithic, or non-differentiable, we are not going that route. 
- Avoid making wrappers that just call another function, effectively just renaming the function, what's the point of that
- We want the framework to be extremely easy for AI agents to use in order to compose many models in order to make digital twins eventually. Keep this in mind. 
- We built a Scheduler for a reason. This is the default handler of multi-timescale processes, and honestly it should be generalizable enough to be THE default handler of everything.
- There are ZERO users of this at the moment. Backwards compatbility is absolutely not a factor in any design decisions.
- End-to-end differentiability is a must. An issue in end-to-end differentiability is never to be circumvented, always solved.
- No sunk-cost reasoning. If a curated / better-validated / more-composable component supersedes existing code, replace it. The fact that we built something earlier is irrelevant to whether it stays — the question is whether it earns its place on biological / architectural grounds today. Drop without ceremony.
- **No locally-cheaper over architecturally-right.** When the design / diary / project plan already names a tool (`Calibrator`, `Scheduler`, etc.) that fits a coupling / fitting / scheduling problem, use that tool — even when a quick adapter, bridge, or shim is several times less work this session. Cheap adapters around the wrong primitive compound across sessions. The DamagePsiBridge + RangeChecker saga is the canonical example: both were built to work around the absence of cross-model calibration that the project had already planned for. They added free parameters, stiffness, and validation noise that compounded for three weeks until they were removed. Watch for the warning sign: if you find yourself writing "this is a starting calibration, the real value should be learned by Calibrator later" — wire Calibrator now, not later.
  - **This applies to throwaway and diagnostic code too, not just library scripts.** When probing, debugging, or one-off-measuring, drive it through the real API (`Scheduler.run`, `analyze_groups`, `screen_composite`, …) rather than re-deriving the mechanism inline — e.g. don't hand-roll `dfx.diffeqsolve` to inspect a solve when `Scheduler.run` is what the production path uses. Ad-hoc code that bypasses the API tests a different thing than what ships, and when it reveals a gap (no result code surfaced, no diagnostic exposed) the fix is to make the API expose it, which improves the framework for the next agent. If a quick check can't be expressed through the API, that's a signal the API is missing something — add it there.

## Documentation style
- **Module / function docstrings are forward-facing.** Describe what the
  thing IS and how to USE it, not what it replaced or what mistakes led
  here. No "previously we did X; this is better because Y" narratives in
  docstrings, even when the discovery context feels informative. The
  discovery history lives in `docs/diary.md` and Claude's memory; the
  code documentation is for readers who don't care how we got there.
- Brief "why this approach" rationale is fine (1–2 sentences) when it
  meaningfully clarifies design intent; skip the contrast with
  abandoned alternatives.
- Same rule for module-top docstrings: open with what the module does,
  not the methodological saga of how we chose this design.
- When suggesting model composite enhancements, prefer models that have downloadable implementations, e.g. on BioModels or elsewhere

## Validation methodology
- **Validation against transcriptomic data uses
  `hallsim.gene_reporters`** — one mechanistic observable ↔ one
  canonical reporter gene, with literature-anchored sign expectations.
  No tunable interpretation layer between mechanism and observation.
- **Calibration is at the mechanism level, not the readout layer.**
  Tuning composite parameters (`alpha`, `k_sasp`, `MDAMAGE_SA`, …)
  against gene-level Δ_data via `jax.grad` + `optax` is fine and
  demonstrates HallSim's end-to-end differentiability; tuning the
  reporter mapping itself is not.
- **Held-out splits are mandatory** for any reported concordance number
  — calibrate on one condition / arm / dataset, evaluate on a held-out
  one. Same-data calibrate-and-evaluate is curve-fit, not concordance.
- **Constituents-first rule (always, no exceptions): before composing — and before debugging a composite — verify that every constituent both *runs* and *tunes* on its own.** "Runs" = simulates to a successful solver result, bounded and tolerance-insensitive (`screen_process` / `screen_composite`). "Tunes" = a forward-mode gradient of a summary w.r.t. one of its parameters is finite (a one-process `CalibrationProblem`, or `jax.jvp` through `Scheduler.run`). A composite can only be as healthy as its parts; when a composite misbehaves (NaN, max_steps, divergence), re-run this check first — if each part is individually fine, the bug is in the *composition* (coupling edge, shared tolerance/atol vector, timescale grouping, reconciliation), not the models. This single step localizes most composite failures in one shot.
- Seek to understand why composites fail to converge, rather than brute forcing different solvers
- Don't pull models from memory/training. Search BioModels API or other objective sources for what you need instead.
- Readme is other user facing. Do not write stuff that is just useful to developers there, such as "This is out of scope for preprint". Cringe. Don't embarrass me.
- Preserve JIT-ability wherver possible. Choose data types and design patterns accordingly.

## Model intake protocol — READ BEFORE TRUSTING A COMPOSITE

A composite is only as trustworthy as its parts, and the most dangerous
failures are silent: a model that *looks* like it ran but produced
numerical garbage. Before importing a new model, composing it, or
believing any composite output, screen each constituent **on its own**.

**The three failure modes to screen for (every new model, every time):**

1. **Exploding** — unbounded growth / NaN / Inf. Usually *numerical, not
   biological*: an explicit solver at loose tolerance pumps energy into
   an oscillator until it diverges ("numerical anti-damping"). A
   published, curated oscillator that "blows up on its own" is almost
   always the solver, not the model.
2. **Vanishing** — every state collapses to ~0; the subsystem lost its
   dynamics and silently contributes nothing.
3. **Tolerance-sensitive** — the load-bearing check. Run the model at a
   loose and a tight solver tolerance. **If the trajectory changes
   materially, the result is solver-dependent and is not yet a result.**
   A bounded, non-vanishing, tolerance-*insensitive* trajectory is one
   you can trust. (This single check would have caught the canonical
   example below.)

**How to screen — use the tooling, don't eyeball:**
- `hallsim.diagnostics.screen_process(proc, t_end)` /
  `screen_composite(comp, t_end)` — returns a pass/flag `ScreenReport`
  per process; `assert all(r.ok for r in reports)` in a test.
- `demos/subsystem_diagnostics.py` — the visual version: plots each
  subsystem solo on its native clock.

**Solver tolerance.** Scheduler default is `rtol=1e-6, atol=1e-9` —
chosen because oscillatory biology (p53–Mdm2, NF-κB, cell cycle, MAPK)
needs accuracy-limited stepping. Do **not** loosen it for a speed-up
without screening every oscillator in the system first. Canonical
example: Geva-Zatorsky 2006 p53 (BIOMD0000000157) is a clean bounded
oscillator that *diverges to ~300× its amplitude and goes negative* at
`rtol=1e-4`, and is bounded from `rtol=1e-5` down. Nothing in the model
changed — only the tolerance.

**Time units.** Composed SBML models often declare different native time
units (DallePezze 2014 = days, Geva-Zatorsky 2006 = hours, an
unannotated model = seconds). `SBMLProcess` extracts `native_time_seconds`
at import; `reconciled_to(canonical_seconds)` puts a model on the
composite's shared clock via chain-rule rescaling. If you compose SBML
models without reconciling, they run at different real-world speeds on a
shared `t` and the result is meaningless — screen for it.

## misc
- Use the venv if there's an associated venv my dude.
- Use framework defaults unless justified
- before composing an sbml model, test it with sbmltoodejax and with our framework. then proceed to compose.
- code should be self-explanatory; minimize comments. Comments must be lean!
- VECTORIZE what vectorized can be.
- Logging, not printing.