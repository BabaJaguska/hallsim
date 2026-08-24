# Known problems

Defects found by review, ordered by priority. Distinct from
[roadmap.md](roadmap.md), which is planned work — everything here is something
that is wrong now. Each entry carries the evidence that established it, so
nothing has to be re-argued.

Evidence sources: the mitochondrial stress test (2026-08-18) and the flagship
review (2026-08-19), both by the review panel in `.claude/agents/`. Findings
confirmed by two independent reviewers are marked ✓✓.

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

- [ ] **P0.6 — `timescale` is a declared unit, not a rate, and it decides
  execution order.** `timescale = native_time_seconds`; `auto_groups` sorts by
  it, putting NF-κB first, so every cross-group edge runs backwards.
  `_effective_coupling` tests only `a < b`, returns `frozen`, and interpolated
  coupling is unreachable. NF-κB integrates 3.5 days against a 4-point staircase
  of its own driver.
  *Fix:* order groups by a rate, not by a unit; make coupling mode independent
  of declaration order.

- [ ] **P0.7 — Prior σ is documented as log10 and passed linear.** Two priors
  are inoperative: `etoposide_potency` (σ=9000) and `psi_K` (σ=200), max penalty
  1.3×10⁻⁸ across the whole clamp box.

- [ ] **P0.8 — A contested initial value is resolved by a rule the user cannot
  see.** Was insertion order (and changed under any pytree round-trip — `jit`,
  `vmap`, `tree_at`); now writer-outranks-reader, which is defensible but still
  invisible from the topology.
  *Fix:* raise when two ports disagree about a path's initial value, and let the
  topology name the winner explicitly.

---

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
- [ ] **P3.2 — No real-eigenvalue continuation.** `hopf_scan` finds oscillation
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
