# Known problems

Defects found by review, ordered by priority. Distinct from
[roadmap.md](roadmap.md), which is planned work — everything here is something
that is wrong now. Each entry carries the evidence that established it, so
nothing has to be re-argued.

Evidence sources: the mitochondrial stress test (2026-08-18) and the flagship
review (2026-08-19), both by the review panel in `.claude/agents/`. Findings
confirmed by two independent reviewers are marked ✓✓.

## The standing criterion

**A user must never have to edit the framework to finish a task.** Every time
one does, that is a P0 regardless of how small the edit was, because the edit
does not survive their next pull and the next user repeats it. Instances so far:

| what forced it | status |
|---|---|
| `screen_sensitivity` would not take a `registry=`, so a model could not ship its own hallmark mappings | fixed |
| `_same_default` broke the trace, so a calibration loss could not run | fixed, and guarded by `tests/unit/test_trace_safety.py` |
| no public parameter setter (P0.9) | open |

The guard for the second one is the pattern to repeat: reachability under trace
is a whole-call-graph property, so the test traces the public entry points and
lets the failure surface rather than trying to reason about it.


---

## P0 — produces a wrong answer with no warning

The framework returns a plausible number and nothing indicates it is wrong.

- [~] **P0.1 — Gradients through a cold stiffness cache are finite garbage.**
  *Correctness fixed 2026-08-19; one consistency gap open.* A traced state is
  now analysed at the composite's concrete initial state (no fallback needed —
  the cache signature already declares the verdict state-independent), and a
  traced composite degrades to the **implicit** solver rather than the explicit
  one. Flagship: max|g| 270591 cold = 270591 warmed, against 6.19×10²³⁶ before.
  **Still open:** `test_batched_matches_solo` — a batched run (traced →
  implicit, `Kvaerno5`) and a solo run (eager → measured, `Tsit5`) pick
  different solvers, so batched ≠ solo when the cache is cold; worst
  disagreement 1.14×10⁻⁴ against the test's `atol=1e-10`. Marked
  `xfail(strict=True)` on 2026-08-23 so the suite is green and the fix cannot
  land silently — it will XPASS and force the marker's removal. Closing it means
  resolving the verdict once at construction and carrying it as a static field,
  which also removes the 115 ms a fresh Scheduler pays and makes cross-process
  persistence a small extension.
  See the diary entry of 2026-08-19 (evening). Original report:
  `jax.jvp` through `Scheduler.run` returns −6.19×10²³⁶ for 22 of 52 states, all
  finite, no raise; `warm_up` first gives −4.9×10⁻³. Stiffness routing cannot
  read a Jacobian under trace, so it falls back to the explicit solver for every
  group and an explicit solver on a stiff system yields nonsense sensitivities.
  The framework prints a warning and continues.
  *Fix:* resolve the stiffness verdict at construction (structure plus a
  representative state are both known before anyone differentiates), or warm on
  first eager call, or raise on a cold trace. Not: expect users to remember
  `warm_up`.

- [ ] **P0.2 — Strang splitting attains no order and is sign-wrong at the
  shipped `macro_dt`.** Refining 14→0.219: Lie converges at O(dt¹·⁰) as
  advertised; Strang's error is 34× larger at the demo step, orders decaying
  1.06 → 0.68 → 0.19 → 0.09. NFKBIA day 14: **+3.27** (Strang) vs −0.647 (Lie)
  vs −0.535 (converged). Lie at `macro_dt=3.5` still carries 17% error.
  `design-multiscale-scheduler.md:445` has said "Measure first" since day one
  and no order test exists.
  *Fix:* withdraw `splitting="strang"` until an order test passes; add that
  test; set `macro_dt ≤ 0.875` for the flagship meanwhile.

- [ ] **P0.3 — The fold-change reference is an acausal filter of the whole
  trajectory.** ✓✓ Summaries are forward–backward EMAs, so the value at index 0
  averages the future. `dp14/CDKN1A` raw y(0)=10, reference used **4176**
  (×418). p21 reports falling when the model has it rising; DDB2 reports −0.089
  where a matched-time rescore gives +0.37 against a measured +0.29.
  *Fix:* default `normalization="paired"` (already implemented —
  `calibration.py:1266`, uses the control arm at the same query times), and make
  summaries causal. Use one normalization rule everywhere.
  **Third instance, 2026-08-24, independent model and dataset.** An outside
  fibroblast composite calibrated on GSE248823 reports loss 105.8 → 9.86 while
  every held-out correlation is *negative*: Spearman −0.061 / −0.339 / −0.043 /
  −0.075, sign agreement 53–67%. A training loss falling 10× while the model
  ranks genes backwards is the diagnostic signature of this defect — the loss is
  reducible by matching the filter artefact. Nobody hunting a modelling problem
  would look at the normalizer, which is why the default has to change rather
  than be documented.

- [ ] **P0.4 — `dose_window=None` silently deletes a hallmark dial.**
  Documented as "sustained drive". `drive_pulse` is skipped, the pulse process
  never exists, and `HallmarkHandle.apply` skips mappings whose target is
  absent. Sweeping severity 0→50 returns the identical attractor to 4 s.f.
  *Fix:* raise when every mapping of an applied hallmark misses its target.

- [x] **P0.5 — `_substitute` overwrites `eqx.tree_at` edits on fitted
  `ParameterRef`s.** *Fixed 2026-08-23.* `CalibrationProblem` snapshots each
  fitted field at construction and `_substitute` raises on an edit it would
  overwrite, pointing at `with_overrides` — the single route for changing any
  parameter, fitted or not, by fittable name or by `<process>.<field>` address.
  Overrides are applied last, so they outrank both the fitted iterate and the
  composite's own value; no caller has to know which list a parameter is in.
  Verified on the flagship: zeroing `mtor_nfkb.k_act` — one of the three edges
  the review ablated — raises when edited in the pytree, and via `with_overrides`
  moves the control arm 0.924 relative against the review's 2.7×10⁻¹³. Editing a
  field nobody fits is untouched and still reaches the solver.
  **Still open:** the mitochondrial panel's "loop gain 3×10⁻⁴" claim was
  possibly produced this way and needs re-measuring.
  Original report: any ablation done by editing the pytree is silently undone,
  so an edge appears dead when it is live. This produced a wrong finding in
  review (three edges measured at 2.7×10⁻¹³; true gains 1.85, ≤0.185, 0.0077).

- [x] **P0.6 — Group execution order came from timescale, so cross-group edges
  ran backwards and interpolated coupling was unreachable.** *Fixed
  2026-08-25.* `auto_groups` still clusters by timescale; `_order_by_coupling`
  then topologically sorts the groups so one runs after whatever drives it,
  keeping timescale order on a cycle. The flagship's dp14/gz06 group now
  precedes nfkb, `_effective_coupling` returns `interpolated`, and NF-κB reads
  an interpolant of its driver instead of a staircase.

  Measured on `nfkb/IkBat` against a `macro_dt=0.109` reference:

  | `macro_dt` | frozen | interpolated |
  |---|---|---|
  | 3.5 (shipped) | 20.9% | **1.7%** |
  | 1.75 | 15.2% | 1.7% |
  | 0.875 | 12.6% | 3.4% |

  Correct ordering at the shipped step beats frozen at a 4× smaller step, at no
  cost. An outside reviewer independently measured ~20% at `macro_dt=3.5`,
  matching the frozen column.
  **Open:** the interpolated column is not monotone (1.7 → 1.7 → 3.4); a
  smaller macro step should not be worse, so either the fixed
  `coupling_interp_points=16` interacts with step size or the reference carries
  error. Not yet understood.
  Original report: `timescale = native_time_seconds`; `auto_groups` sorted by
  it, putting NF-κB first; `_effective_coupling` finds no earlier-writes /
  later-reads pair and returns `frozen`. NF-κB integrated 3.5 days against a
  4-point staircase of its own driver.

- [ ] **P0.13 — `timescale` is a declared unit, not a rate.** Split from P0.6,
  whose execution-order half is fixed. An SBML import sets `timescale =
  native_time_seconds` — the model's declared time unit, not how fast it
  moves — and `auto_groups` clusters on it, so two models with the same
  dynamics but different declared units land in different groups and one with
  the same unit but different speeds lands in the same one.
  *Fix:* cluster on a measured rate. `analyze_groups` already computes a
  spectral abscissa.

- [ ] **P0.7 — Prior σ is documented as log10 and passed linear.** Two priors
  are inoperative: `etoposide_potency` (σ=9000) and `psi_K` (σ=200), max penalty
  1.3×10⁻⁸ across the whole clamp box.
  **Second instance, 2026-08-24.** The outside fibroblast model reports
  `etoposide_potency = 9237.104086` after fitting — still the uncorrected 593×
  exposure mismatch it started from, because nothing penalised it. Its own
  earlier note said the inflated magnitudes were "to be corrected in
  calibration"; they were not, and could not be.

- [ ] **P0.8 — A contested initial value is resolved by a rule the user cannot
  see.** Was insertion order (and changed under any pytree round-trip — `jit`,
  `vmap`, `tree_at`); now writer-outranks-reader, which is defensible but still
  invisible from the topology.
  *Fix:* raise when two ports disagree about a path's initial value, and let the
  topology name the winner explicitly.

---

- [ ] **P0.9 — The supported parameter route exists only on
  `CalibrationProblem`, and is undiscoverable from anywhere else.**
  `with_overrides` (`calibration.py:1067`) is the correct answer and P0.5's fix
  routes users to it — but it is a method on the *calibration problem*. A user
  doing bifurcation analysis, an ablation sweep, or any bare `Scheduler.run`
  never touches a `CalibrationProblem`, finds nothing on `Process` or
  `Composite`, and invents a name: `replace_param`, `set_param`, then falls
  back to `eqx.tree_at`. Observed in an agent session after the P0.5 fix landed,
  which is the point — the fix guards the footgun for calibration callers and
  leaves everyone else where they were.
  *Fix:* a `Process.with_param(name, value)` / `Composite.with_params({...})`
  that `with_overrides` itself delegates to, so there is one implementation and
  it is reachable from the object the user already has. **Landed 2026-08-24**
  (`hallsim.process.write_param`); the entry stays open until `with_overrides`
  is confirmed delegating on every path.
  **Second instance, 2026-08-24, cost a whole analysis.** The outside model's
  sensitivity study covers "only SBML-based parameters; hand-written process
  parameters (SASP, Passos) could not be analyzed due to API differences" — so
  it excluded precisely the five processes that were the model's contribution.

- [ ] **P0.12 — Nothing enforces the tolerance prohibition the docs state.**
  `CLAUDE.md:160` says Geva-Zatorsky 2006 "diverges to ~300× its amplitude and
  goes negative at `rtol=1e-4`, and is bounded from `rtol=1e-5` down". A user
  may still pass `rtol=1e-4` to a composite containing that oscillator and gets
  no warning. Worse, the workaround is *attractive*: relaxing tolerance is the
  obvious response to a stage-convergence failure (see P0.11), so the prohibited
  setting is the one a stuck user reaches for.
  **Observed 2026-08-24.** The outside fibroblast model ran at `rtol=1e-4` with
  GZ06 in the composite and reported, as a *biological* finding, that "the GZ06
  p53 oscillator produces very large amplitude values that may need rescaling".
  That is the documented anti-damping, mis-read as biology, and it contaminates
  the dose-response, bifurcation and population figures the oscillator feeds.
  *Fix:* refuse — or warn loudly and record on the result — when `rtol` is
  looser than the screened bound for a composite whose constituents include a
  model flagged oscillatory. `screen_process` already measures tolerance
  sensitivity; the Scheduler should consult that verdict rather than leaving it
  in a prose document.

- [ ] **P0.10 — The default `macro_dt = 5.0` carries material splitting error,
  unwarned.** Lie splitting was measured at 17% error on the flagship at
  `macro_dt = 3.5`; the shipped default is larger still, and nothing reports it.
  `CalibrationProblem` defaults to it (`calibration.py:787`).
  *Fix:* report the splitting-error estimate at construction, or default to a
  step the order test justifies.

- [ ] **P0.11 — One parameter feeding two consumers makes every diagnosis of it
  ambiguous.** `atol` was the error controller's tolerance *and* the implicit
  stage's Newton tolerance, so "relax `atol`" fixed a convergence failure while
  silently degrading trajectory accuracy for every oscillator — and made the
  prohibited workaround the only one that appeared to work. Fixed for `atol`
  (`DEFAULT_NEWTON_ATOL`); the class is not audited. `rtol` still feeds both.
  *Fix:* audit every solver parameter for double duty and split each one.

## P1 — cannot tell whether a result is trustworthy

The check that would catch a mistake does not exist, does not run, or fails open.

- [ ] **P1.1 — No `check_gradient`.** End-to-end differentiability is the
  framework's headline claim and ships with no validator. (Review verified the
  adjoint by hand: forward ≡ reverse, finite-difference confirmed. Users cannot.)
- [ ] **P1.2 — `screen_process` never varies `atol`** (`diagnostics.py:384`
  deliberately does not), though tolerance sensitivity is the intake protocol's
  own load-bearing check.
- [ ] **P1.3 — Identifiability is reported but not enforced.** Fisher condition
  number 3.5×10³⁰²; exactly one parameter estimable; σ up to 7128 decades; three
  pairs at |corr| > 0.998. Fitting proceeds regardless.
- [ ] **P1.4 — A held-out arm can be bit-identical to a fit arm** with nothing
  detecting it. ✓✓ `RAS_vs_ctrl` max|Δ| = 0.000e+00.
- [ ] **P1.5 — A fittable coefficient on a `slope` is unlearnable from arms
  where that hallmark's severity is 0.** ✓✓ `∂loss/∂gain = base × severity`.
  The mTOR suppression gain has exactly zero gradient on every fit arm and is
  the only parameter distinguishing the held-out arm.
  *Fix:* report zero-gradient fittables at problem construction.
  **Second instance, 2026-08-24, and it invalidated a headline claim.** The
  outside fibroblast model reports `passos_k_gadd45 = 0.100000` and
  `passos_k_ros = 0.100000` — both exactly their initial values to six decimal
  places — and `dns_mtor_gain = 0.703092` from an init of 0.7. Its key finding
  states that "the Passos feedback loop … creates self-reinforcing senescence",
  but the two parameters governing that loop were never fitted. A report cannot
  be expected to notice three unmoved parameters among sixteen; the problem
  construction can, in one pass, for free.
- [ ] **P1.6 — No null-model baseline is reported.** The flagship scores 19/36
  signs (52.8%); "every reporter rises" scores 30/36 (83.3%).
- [ ] **P1.7 — No parameter provenance.** Nothing distinguishes measured from
  fitted from invented, so a benchmark can be scored against a parameter fitted
  to it — which happened in both models reviewed.
- [ ] **P1.8 — Operating-range violations warn and continue.** DP14 runs 416×
  outside its own range with the framework's 593× exposure warning printed and
  ignored.
- [ ] **P1.9 — Conservation laws are still inferred numerically for any
  composite containing a hand-written process.** The exact stoichiometric path
  needs every process to declare `stoichiometry()`; one undeclared edge disables
  it.

- [ ] **P1.10 — `equilibrate=True` is ill-posed for a composite containing an
  autonomous oscillator, and fails.** `steady_state`'s Newton finds the
  *unstable* fixed point at the centre of the Geva-Zatorsky limit cycle;
  starting a forward solve there is unphysical and numerically hostile, which
  is what "the Newton fixed point produces an initial condition the forward
  solver can't handle" means. The module docstring assumes "any limit cycle
  belongs to the perturbation" — GZ06 oscillates unperturbed, so the assumption
  does not hold for any composite that includes it. Currently blocking a real
  user, whose only recourse was to abandon equilibration entirely.
  *Fix:* P3.4 — partial equilibration. Equilibrate the non-oscillatory
  subsystem, hold the oscillator at its published initial condition. Until then
  `equilibrate=True` should refuse with this explanation rather than fail in the
  solver.

- [ ] **P1.11 — A `ParameterRef` is never validated against the field it
  names.** Point one at a tuple-valued field — `HillActivationEdge.K`, `.n`,
  which are `tuple` and deliberately *not* `calibratable` — and nothing objects.
  Substitution writes a scalar over the tuple and the run dies inside
  `HillActivationEdge.derivative` with `TypeError: iteration over a 0-d array`,
  several frames from anything the user wrote. Observed in an agent session,
  which lost a fit step to it and concluded the framework had a bug in the edge.
  *Fix:* validate every `ParameterRef` at problem construction — the field must
  exist, must not be static, and its current value must be a scalar. Say which
  of those failed and, for a non-`calibratable` field, that fitting it is
  unsupported. The check is cheap and the failure it replaces is unreadable.

---

## P2 — cannot see what was built

- [ ] **P2.1 — No wiring report.** Nothing lists, per store path, who writes it
  and who reads it. NF-κB being write-only was invisible until someone wrote a
  script.
- [ ] **P2.2 — Declared and live edges are indistinguishable.** A topology arrow
  carrying 0.0077 of its target's derivative looks exactly like one carrying
  1.85. `CouplingAuditor` could report the measured share.
- [ ] **P2.3 — No way to see inside an imported SBML model.** Species,
  reactions, rate laws as readable infix, and the stoichiometry matrix (now
  extractable via `proc.stoichiometry()`) are all available and none are
  surfaced. Reading the XML is not an answer.
- [ ] **P2.4 — SBML `substanceUnits` discarded** (`sbml_import.py:364` hardcodes
  `dimensionless`), so the UnitChecker cannot fire on the framework's primary
  model source, and declaring a real unit on a port that touches an imported
  species is a hard error.
- [ ] **P2.5 — sbmltoodejax's `w` vector is computed and thrown away.**
  Assignment-rule species report stale constants; a model's own conservation is
  visibly violated in the output with no warning. Fluxes are unreadable and
  unusable as coupling sources.
- [x] **P2.6 — Three documents describe three different reporter sets**, none
  matching the code. *Fixed 2026-08-23.* All three now state the live set —
  CDKN1A, GLB1, BNIP3, DDB2, MDM2, NFKBIA with their real store paths — and
  `tests/unit/test_gene_reporters.py::TestPublishedReporterTable` parses each
  one and compares it to `MULTI_HALLMARK_REPORTERS`, so they fail rather than
  drift. The two markdown tables sit inside `<!-- reporters:start/end -->`
  markers, leaving prose elsewhere free to name any gene. Also corrected while
  in there: `dataset.md` described the held-out arm as `RAPA_vs_DDIS` against a
  time-matched comparator, where the code runs `RAPA_vs_ctrl` normalised within
  the arm to `ETOPOSIDE_D00`; `calibration.md`'s worked example said the same
  and described summaries as co-solved `RunningIntegral`s, which the flagship
  stopped using in favour of post-hoc zero-phase filters.
  **Found while fixing, not fixed:** `demos/multi_hallmark_hybrid.py:492` reads
  `gz06/x2_integral`, a store path the composite no longer has — that demo
  cannot run.

---

## P3 — capability gaps

- [ ] **P3.1 — Severity cannot be a state.** A hallmark dial is a constant set
  before the run, so aging is imposed as an initial condition. For an attractor
  to change, severity must evolve — a depleting repair capacity, a ratchet.
  This is the prerequisite for any bifurcation claim.
- [ ] **P3.2 — No real-eigenvalue continuation.** *Second instance,
  2026-08-24: the outside fibroblast model ran a "bifurcation" scan over
  irradiation dose and produced `bifurcation_dose.png`. Dose enters only inside
  a pulse window there, so after the pulse the vector field is dose-independent
  and no transition can exist — the figure is a monotonic response curve
  labelled as a bifurcation diagram. Both the missing fold scan and the absence
  of any check that a scanned parameter still appears in the RHS after the scan
  window.* `hopf_scan` finds oscillation
  onsets and returns `[]` on a textbook transcritical. Fold and transcritical
  crossings — bistability, invasion thresholds, senescence commitment — are the
  transitions this domain actually has.
- [ ] **P3.3 — No `resilience_scan`.** Recovery rate, AR(1) and variance versus
  age is a 60-line hand-rolled script for the repo's own headline claim.
- [ ] **P3.4 — No partial equilibration.** `steady_state` is all-or-nothing;
  a composite mixing fixed-point states with ratchet states has no whole-system
  fixed point.
- [ ] **P3.5 — No quasi-steady-state / DAE facility.**
- [ ] **P3.6 — No stochastic support.** Batched `y0` pushes a distribution
  through one deterministic flow onto one attractor; that is not a population.
- [ ] **P3.7 — No batched parameter sweeps.** Severity varies the pytree, not
  the state, so a sweep is a Python loop.
- [ ] **P3.8 — No observable layer.** Nothing carries assay, unit, normaliser
  or transducer, so a millivolt is compared to a ratiometric dye reading.
- [ ] **P3.9 — No uncertainty.** Outputs are lines; a lab needs bands.
