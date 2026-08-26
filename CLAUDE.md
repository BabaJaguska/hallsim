# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Environment

- Project venv is `.venv_hallsim/` (non-standard name). Use `.venv_hallsim/bin/python` for ad-hoc runs.
- Python >=3.11. Core stack: JAX + Equinox + Diffrax + Optax. Validation uses `pint` (units) and `networkx` (graph analysis). SBML import uses `sbmltoodejax`.
- `pyproject.toml` is the single source of truth for dependencies. No lockfiles — `make install` / `make install-dev` editable-install directly from pyproject.
- `cd` into the project before running scripts.
- Importing `hallsim` enables x64 and points XLA's persistent compilation cache at `~/.cache/hallsim/jax`. **Set `HALLSIM_COMPILATION_CACHE_DIR=off` before timing anything compile-related**, or the second measurement is a cache hit and the comparison is meaningless.

## Common commands

```bash
make install        # editable install of runtime deps
make install-dev    # dev deps + editable install

make test           # CI subset: not slow, not network, not demo
make test-all       # everything except network
.venv_hallsim/bin/python -m pytest tests/unit/test_multiscale.py -v -k scheduler

make format         # black, line length 79
make lint           # flake8, ignores E501,E402,W504,W503,E226,E203
make check          # format --check + lint (what CI runs)

# Demos via the `simulate` entry point (hallsim.cli:simulate)
simulate compose | compose-kick | multiscale | validate-demo | info
simulate multi-hallmark run | calibrate | sweep
simulate stiffness
```

## Architecture invariants

The README has the overview. These are the load-bearing rules — break them and composition fails silently or at validation time.

**Store has path-like keys, no hierarchy.** Dict shape at API boundaries is `dict[str, jnp.ndarray]` with keys like `"cytoplasm/ROS"`. The `/` is convention, not nesting.

**Port roles encode write semantics** (validated in `Composite.__init__`):
- `EVOLVED` — additive. Multiple processes writing the same path get summed in the RHS.
- `EXCLUSIVE` — sole owner. A second writer raises in `validate_topology`.
- `INPUT` — read-only. No derivative contribution.
- `LATCHED` — written only by DISCRETE/EVENT processes; CONTINUOUS processes read it as constant within a macro step.
- `ASSIGNED` — algebraic output computed each step via `assign`, not integrated. Sole owner of its path.

**Topology lives outside processes.** Wiring is `{proc_name: {port_name: store_path}}` passed to `Composite`. Never put store paths inside Process classes.

**`composite.build_rhs(proc_names=None) -> (rhs_fn, keys)`.** Flat by design — operates on a 1-D array in `sorted(store_paths())` order so JAX traces a single array under JIT/grad. `proc_names=None` uses every CONTINUOUS process; an explicit list builds a partial RHS for operator splitting. State flows as a flat vector throughout; dict shape appears only at API boundaries and inside the per-process port view.

**Process kinds drive scheduler dispatch:**
- `CONTINUOUS` (default) — `derivative(t, state)`, solved by Diffrax.
- `DISCRETE` — `update(t, state)`, fired every `dt_step`.
- `EVENT` — `condition(t, state)` + `handler(t, state)`, fires on False→True at sync points.

The Scheduler is the only runner; there is no separate `Simulator`. Single-group continuous composites with no events/discrete/adaptive_dt/Strang take a fast path: one `dfx.diffeqsolve` over the whole `t_span`.

**Timescale auto-grouping.** `Composite.auto_groups(max_ratio=100.0)` clusters CONTINUOUS processes by `proc.timescale`. Set `timescale` on processes with very different rates so the Scheduler doesn't force a stiff integrator on all of them.

**Splitting / coupling are independent Scheduler knobs:**
- `splitting`: `"lie"` (default, O(dt)) vs `"strang"` (symmetric half-step, O(dt²)).
- `coupling_mode`: `"auto"` (default), `"frozen"`, or `"interpolated"`.
- `adaptive_dt=True`: PLL-inspired step control on coupling residual.
- Strang + interpolated are rejected in `Scheduler.__init__` — Strang's reverse pass needs an interpolant that has not been produced yet.

**Hallmarks are immutable parameter modifiers.** `apply_hallmarks(processes, {hallmark: severity})` returns a *new* dict. Severity is differentiable end-to-end.

**Validation layer is on by default, warning-by-default.** `Composite(...)` runs UnitChecker / SemanticChecker / GraphAnalyzer / CouplingAuditor. Unit/ontology *conflicts* raise; everything else warns. Disable with `semantic_validation=False`; promote to errors with `semantic_validation={"strict": True}`.

## JAX invariants

These decide whether the framework is fast. Violating one is a performance bug, not a trade-off.

**Structure is static; values are traced.** A Process field that is a name, index map, or port default must be `eqx.field(static=True)`. A field that is a fitted parameter must be a traced array — `Process.__check_init__` coerces floats to arrays so a value change is data, not a recompile. Get this backwards and either `ports_schema()` breaks under trace, or every parameter value recompiles.

**Tracing is not compilation.** The XLA compile cache is keyed *on the trace*, so an unjitted function reuses the compiled executable but must re-trace to look it up. "0 recompiles" is necessary, not sufficient — a re-traced `lax.scan` of N `diffeqsolve` bodies costs hundreds of ms per call with zero recompiles. Measure trace cost, not just `jax.log_compiles`.

**`_FlatRHS` is an `eqx.Module`, not a closure.** A closure inside `ODETerm` is a static leaf: a fresh one per `build_rhs` hashes differently and every solve misses the cache.

**Keep loop-invariant work out of the RHS.** Anything that doesn't depend on `t` or `y` belongs in `build_rhs`, not in `derivative`.

**Preserve JIT-ability everywhere.** Choose data types and design patterns accordingly.

## Where to add things

- **New model**: `src/hallsim/models/<name>.py`. Subclass `Process`, declare `ports_schema()`, implement `derivative` (or `update` / `condition`+`handler`). Provide a `build_<name>_composite()` factory — the pattern in `eriq.py` and `stem_cell_niche.py`.
- **Multiplicative coupling**: effects are summed via EVOLVED ports. For multiplicative coupling, route the modulating variable through a separate store path and read it via an `INPUT` port.
- **SBML model**: `hallsim.sbml_import.process_from_sbml(...)`. Pre-imported BioModels live under `src/hallsim/models/sbml/<author><year>/`.
- **Hallmark mapping**: `hallsim.hallmarks.apply_hallmarks`.
- **Tests**: `tests/unit/test_composition.py` (Process/Port/Topology contracts), `test_multiscale.py` (Scheduler), `test_validation.py` (semantic layer), `test_models.py` (per-model regression), `test_performance.py` (JAX invariants above).

## Notes for editing

- Process subclasses are `eqx.Module` — fields are JAX-traced. New parameters must be declared as class attributes with type annotations or they won't survive `eqx`/JIT.
- Keep duplicated algebraic intermediates inside Processes (the ERiQ pattern). Sharing them via the store crosses module boundaries and breaks independent testability.
- `NeuralODE` ships with training infrastructure, not just a dynamics module — see `hallsim.models.neuralode`.
- Adding a demo means adding a `@simulate.command()` in `cli.py`.

## Design principles

- The repo is about modularity, composability, JAX-based speedups and backprop. If something makes things slow, uncomposable, monolithic, or non-differentiable, we don't go that route.
- End-to-end differentiability is a must. A break in it is solved, never circumvented.
- **Performance is a first-class citizen.** JAX was the choice *because* of it. Correct-but-slow is not done: measure it, and if something got slower that is a bug. Compile and dispatch overhead count.
- Avoid wrappers that just rename another function.
- The framework must be easy for AI agents to use to compose many models into digital twins.
- The Scheduler is THE default handler of everything. Calling `dfx.diffeqsolve` directly and bypassing it is a mistake — if tempted, ask what would make the Scheduler appealing enough to use, then implement that.
- ZERO users today. Backwards compatibility is not a factor in any design decision.
- **No sunk-cost reasoning.** If a better-validated or more-composable component supersedes existing code, replace it. Drop without ceremony.
- **No locally-cheaper over architecturally-right.** When the design already names a tool (`Calibrator`, `Scheduler`) that fits the problem, use it — even when a quick adapter is less work. Cheap adapters around the wrong primitive compound across sessions. Warning sign: writing "this is a starting calibration, the real value should be learned by Calibrator later" — wire Calibrator now.
  - This applies to throwaway and diagnostic code too. Drive probes through the real API (`Scheduler.run`, `analyze_groups`, `screen_composite`) rather than re-deriving the mechanism inline. If a quick check can't be expressed through the API, that's a signal the API is missing something — add it there.
- **Am I patching, or fixing?** A patch makes the symptom go away and leaves the cause; a fix removes the cause and usually *deletes* code. If the change is a special case, a workaround, or a flag routing around a bug — stop, find what made it necessary, remove that instead.
- Separation of concerns: no monoliths doing several jobs in one place.
- Vectorize. Python loops are looked down upon, even three-row ones. We are batch-native — don't skip the batch dimension.
- Logging, not printing.
- Parametrize figures and outputs; assume things change, hard-code nothing.
- Use framework defaults unless justified.
- If COPASI / Tellurium / AMICI solved a problem, borrow it. Don't reinvent.

## Code and documentation style

- **Minimal comments.** Comment only the genuinely non-obvious. The code should read for itself; variable names carry the explanation.
- **Docstrings are forward-facing.** Describe what the thing IS and how to USE it — not what it replaced or what mistakes led here. No "previously we did X" narratives. A 1–2 sentence "why this approach" is fine when it clarifies design intent.
- **Am I writing diary, or code?** Rationale, measurements, dead ends, and "we used to do X" belong in `docs/diary.md`. Code says what the thing IS plus the one line that stops someone re-breaking it.
- **Never leak the local environment into public-facing text.** No venv names (`.venv_hallsim/bin/python`), no absolute paths, no machine-specific dirs — not in the README, demo docstrings (they surface in `--help`), docs/, or error messages. Public invocations use `simulate <command>` or plain `python demos/x.py`. The local venv belongs in this file and nowhere else.
- The README is user-facing. No developer-only notes there.
- **We have a click CLI — use it.** A capability worth showing a user gets a `simulate` command, and that is what the README quotes. Don't document `python demos/x.py` when the CLI covers it.

## Communication style

- "Honest" is a banned word. Prefacing a bad result with "an honest finding" does not rescue it — if it's bad, it's bad. Don't hunt for synonyms ("truthful") either.
- "Hand waving" (any form) is banned too: it gestures at rigour instead of having it. Don't announce that you are *not* hand waving, and don't accuse a result of it — either measure the thing and report the number, or say plainly what you did not check.
- When debugging, we debug together. Show the output graphs, tables, and figures the decisions rest on — not a scratchpad.
- Claude memory is not how to make agents behave a certain way. THE FRAMEWORK ITSELF is. Whatever can be built unambiguously into the framework should be. This file is for you, not a manual for everyone.

## Validation methodology

- **Two readout layers, two rules.** `hallsim.gene_reporters` is the validation instrument — one observable ↔ one canonical gene, literature-anchored sign, no tunable parameters. Its value is that it cannot be fit. A *fitted* readout head (TF activity → regulon → transcriptome) is how the framework reaches transcriptome scale; it is a model, needs priors, identifiability screening, and a held-out split over **perturbations**. Report the two scores separately.
- **Calibration targets mechanism parameters and readout-head gains, never the canonical reporter mapping.**
- **Held-out splits are mandatory** for any reported concordance number. Same-data calibrate-and-evaluate is curve-fit, not concordance. Test on held-out data, not training data.
- **Constituents-first rule (always, no exceptions): before composing — and before debugging a composite — verify every constituent both *runs* and *tunes* on its own.** "Runs" = a successful solver result, bounded and tolerance-insensitive (`screen_process` / `screen_composite`). "Tunes" = a forward-mode gradient of a summary w.r.t. one parameter is finite. A composite can only be as healthy as its parts; when one misbehaves, re-run this check first — if each part is fine, the bug is in the *composition* (coupling edge, shared tolerance, timescale grouping, reconciliation).
- Seek to understand why composites fail to converge rather than brute-forcing solvers.
- Don't pull models from memory/training. Get them from a source that can be cited and re-downloaded: BioModels, CellML/Physiome, JWS Online, ModelDB (XPP), or a paper's supplement. Record where it came from.
- Before composing an SBML model, test it with sbmltoodejax and with our framework.
- Prefer models with a downloadable implementation over ones that exist only as printed equations — but transcribing equations from a paper is in scope, not a fallback.
- Remember to equilibrate — but check whether the source model envisions it. Read the original literature.

## Model intake protocol — read before trusting a composite

A composite is only as trustworthy as its parts, and the most dangerous failures are silent: a model that *looks* like it ran but produced numerical garbage. Screen each constituent **on its own** before importing, composing, or believing any output.

**The three failure modes:**

1. **Exploding** — unbounded growth / NaN / Inf. Usually *numerical, not biological*: an explicit solver at loose tolerance pumps energy into an oscillator until it diverges ("numerical anti-damping"). A published, curated oscillator that "blows up on its own" is almost always the solver.
2. **Vanishing** — every state collapses to ~0; the subsystem silently contributes nothing.
3. **Tolerance-sensitive** — the load-bearing check. Run at a loose and a tight tolerance. **If the trajectory changes materially, the result is solver-dependent and is not yet a result.**
4. **Not at rest** — the published initial condition is a fitted experimental starting point, not a steady state, so the run is mostly relaxation. `ScreenReport.rest_tau` is how long the fastest state takes to change by 100% of itself; below the save interval, the saved trajectory never contains the declared IC. Composition makes it worse — that transient is injected into every process downstream through the coupling edges. Advisory, because a stimulus at t=0 is *supposed* to move the state: the flag tells you to decide whether to equilibrate, not that the model is wrong. Canonical case: DallePezze 2014 reports τ = 13 s on `Mitophagy` against a 14-day horizon, and its IC is provably not a rest state of the model at all.

**How to screen — use the tooling, don't eyeball:**
- `hallsim.diagnostics.screen_process(proc, t_end)` / `screen_composite(comp, t_end)` — returns a pass/flag `ScreenReport`; `assert all(r.ok for r in reports)` in a test.
- `demos/subsystem_diagnostics.py` — the visual version.

**Numerical screening is not scientific review.** A model can pass every check above and still be wrong: parameters invented rather than measured, a citation that does not say what it is cited for, a sign error, units that only appear consistent. Peer review is not a guarantee — assume nothing about an imported model because it was published.

`hallsim.intake.triage_sbml` / `triage_batch` is the cheap gate that runs first — SBML metadata plus one solve, no reviewer. It rejects what will not import, has unsupported constructs, or fails the numerical screen, and flags a missing time unit, thin ontology coverage, or an IC that is not a rest state (`rest_residual` = ‖f(y₀)‖/‖y₀‖). Only `verdict.escalate` models are worth a reviewer's time.

What survives triage then gets the review panel in `.claude/agents/`, run on the *individual* model before it joins any composite:
- **bench-scientist** — pulls the cited papers, checks they say what they are cited for, classifies every parameter measured / fitted / invented.
- **mathematician** — well-posedness, dimensional consistency, stiffness, identifiability.
- **physicist** — orders of magnitude, thermodynamic consistency, timescale separation.

Reports land in `docs/review-<model>-{wetlab,maths,physics}.md` (gitignored raw evidence); findings that matter are lifted into `docs/known-problems.md`. The panel definitions are tracked; their output is not. Skip the panel for a quick probe, never for a model whose numbers will be reported.

**Solver tolerance.** Scheduler default is `rtol=1e-6, atol=1e-9`, because oscillatory biology (p53–Mdm2, NF-κB, cell cycle, MAPK) needs accuracy-limited stepping. Do **not** loosen it for a speed-up without screening every oscillator first. Canonical case: Geva-Zatorsky 2006 p53 (BIOMD0000000157) is a clean bounded oscillator that *diverges to ~300× its amplitude and goes negative* at `rtol=1e-4`, and is bounded from `rtol=1e-5` down. Nothing changed but the tolerance.

**Time units.** Composed SBML models often declare different native time units (DallePezze 2014 = days, Geva-Zatorsky 2006 = hours, an unannotated model = seconds). `SBMLProcess` extracts `native_time_seconds` at import; `reconciled_to(canonical_seconds)` puts a model on the composite's clock via chain-rule rescaling. Compose without reconciling and the models run at different real-world speeds on a shared `t` — the result is meaningless. Screen for it.
