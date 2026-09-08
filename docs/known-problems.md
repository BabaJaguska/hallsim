# Known problems

Defects found by review, ordered by priority. Distinct from
[roadmap.md](roadmap.md), which is planned work — everything here is something
that is wrong now. Each entry carries the evidence that established it, so
nothing has to be re-argued.

**Open entries only.** A defect that is fixed moves to
[fixed-problems.md](fixed-problems.md) rather than being deleted, because the
reviews and the diary cite it and a citation has to resolve. Ids are unique
across both files and `tests/unit/test_known_problems.py` checks that, that
this file holds no closed entry, and that no code cites an id at all.

Evidence sources: the mitochondrial stress test (2026-08-18), the
multi-hallmark demo review (2026-08-19), and the DallePezze 2014 referee pass
(2026-08-25) — all by the review panel in `.claude/agents/` — plus two outside
models calibrated against GSE248823 by agents who did not have this list, and an
external systems-design review of the framework itself (2026-08-31) which read
this file first and reported against it. Findings confirmed by two independent
reviewers are marked ✓✓. The panel's raw reports are gitignored; what survived
review is here.

**On the multi-hallmark demo.** Several entries below are stated against it,
because it is the largest composite on hand and therefore the one that
exercises the most framework surface. That makes it a *test workload*, not a
result. It is known-broken as biology — P0.14: its control arm is not a
control, so its concordance number means nothing and a constant null beats it
(P1.6). Read every demo number here as a diagnostic of the framework path it
exercised. Nothing about HallSim's value depends on that model being right,
and no entry should be read as if it did.

## The standing criterion

**A user must never have to edit the framework to finish a task.** Every time
one does, that is a P0 regardless of how small the edit was, because the edit
does not survive their next pull and the next user repeats it. Instances so far:

| what forced it | status |
|---|---|
| `screen_sensitivity` would not take a `registry=`, so a model could not ship its own hallmark mappings | fixed |
| `_same_default` broke the trace, so a calibration loss could not run | fixed, and guarded by `tests/unit/test_trace_safety.py` |
| no public parameter setter (P0.9) | open |
| `steady_state` was intractable at 910 nodes × 300 conditions, so the inner solve was replaced by a hand-written linear solve (P3.10) | open |
| semantic validation had to be switched off wholesale to compose 910 generated ports (P3.11) | cost fixed 2026-08-29 (SCC); annotation granularity open |

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
  one. Multi-hallmark demo: max|g| 270591 cold = 270591 warmed, against 6.19×10²³⁶ before.
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

- [ ] **P0.2 — Strang splitting does not attain second order.**
  *Contradicted 2026-09-05, see P0.47: observed order is **p = 2.33 then 2.07**
  over `dt` 0.5 -> 0.25 -> 0.125, and Strang is 105x more accurate than Lie at
  `dt = 0.125`. The earlier readings sampled steps at or above the fastest
  period, where no splitting scheme is asymptotic. Needs re-taking at steps
  small enough for the claim to be testable.*
  Re-measured
  2026-08-31 on a forced two-group split (DP14 slow / GZ06 + edges fast),
  against a 32×-refined Lie reference, four observables:

  | scheme | macro_dt 3.5 | 1.75 | 0.875 |
  |---|---|---|---|
  | Lie | **36.3%** | 18.3% | 9.4% |
  | Strang | **4.8%** | 2.1% | 1.4% |
  | Lie + interpolated | 36.3% | 18.3% | 9.4% |

  **This corrects the entry's previous headline.** It read "Strang's error is
  34× larger at the demo step" and "sign-wrong"; Strang is in fact **7.6×
  smaller** than Lie there and sign-correct on all four observables. What
  survives is the order deficit: observed order decays 1.16 → 0.63 against the
  2.0 Strang should attain. Lie is *worse* than previously recorded — 36% at
  `macro_dt=3.5`, not 17% — though it converges cleanly at O(dt¹) (ratios 1.98,
  1.96), so the scheme is fine and the 3.5-day step is not. CDKN1A is the
  casualty: 30.98 against a reference 22.73.

  The old numbers came from the three-group configuration containing NF-κB, so
  the discrepancy is unseparated between (a) NF-κB's 1.6 h oscillation breaking
  Strang specifically and (b) the P0.20 span-truncation fix repairing it — a
  wrong window count would break Strang's forward-then-reversed symmetry
  directly. Worth separating before designing around either.

  *Usable today:* Strang at `macro_dt=1.75` gives 2.1%, at 0.875 gives 1.4% —
  a defensible multi-group configuration without waiting for the fix.
  *Fix:* find why Strang loses its second order; add the order test
  `design-multiscale-scheduler.md:445` has called for since day one.

  **Third configuration, 2026-09-05** — the natural three-model split with
  Kallenberger in (DP14 slow / GZ06 + K14 + edges fast), six reporters, against
  a Lie@0.05 reference. The sign is **opposite** to the row above:

  | macro_dt | 3.5 | 1.0 | 0.5 | 0.25 | 0.1 | 0.05 |
  |---|---|---|---|---|---|---|
  | Lie | 9.95% | 1.86% | 1.14% | — | — | ref |
  | Strang | 72.0% | 29.1% | 8.8% | **17.5%** | 0.25% | 0.15% |

  Strang converges to the same limit, so it is not wrong — but it is 7× *worse*
  than Lie at the demo step and **non-monotone** (0.5 → 0.25 gets worse before
  0.1 recovers). Neither the forced-split row above nor this one is the whole
  story, and the difference between them is now the cleanest lead on the order
  deficit: same two-group shape, opposite verdict.

  **What separates them is P0.47.** This configuration's two groups are joined
  by a *cycle*, so both schemes are cutting a feedback loop and neither order
  of operations is defensible; the forced split above was chosen by hand and
  may not have been. Re-run this study on a composite whose groups are
  genuinely acyclic — P0.47's merge now guarantees `auto_groups` produces only
  such splits — before drawing an order conclusion from either row. That also
  retires the "NF-κB oscillation vs P0.20 span-truncation" ambiguity as the
  only candidate pair.

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
  unwarned.** Lie splitting was measured at 17% error on the multi-hallmark demo at
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

- [ ] **P0.14 — The multi-hallmark demo's control arm is not a control.** ✓✓ DallePezze
  2014 is monostable: 512 leaf-preserving Newton seeds and 2001 projected seeds
  each find exactly **one** non-negative fixed point, stable at max Re λ
  = −0.0730/day, and 64 random ICs integrated 200 days all land on it (spread
  1.7×10⁻⁵). Deleting the irradiation route entirely leaves
  ‖y_dosed(400) − y_nodose(400)‖ = 1.8×10⁻⁵, and across six decades of dose the
  day-400 endpoint is identical to six significant figures. The undosed arm is
  *more* senescent at 14 d (γH2A.X ×9.03 vs ×8.28, SA-β-gal ×14.08 vs ×10.89):
  the mitochondrial arm is a positive feedback loop and CDKN1A is its only
  brake, so the dose acts as a brake on senescence. `ctrl` and `DDIS` therefore
  end at the same attractor and the fold-change contrast is mostly timing —
  the shape that lets a constant null beat the model on magnitude while the
  training loss falls. Reproduced independently on an outside fibroblast
  composite.
  Worse, the published initial condition is not a state of the model:
  ‖f(y₀)‖₂ = 67 593, and any rest state must satisfy
  `SA_beta_gal = 0.45287·ROS` and `DNA_damage = 0.36495·ROS`, so ROS = 10
  forces SA-β-gal 4.53 against the published 0.81. No parameterisation fixes
  it; the PottersWheel source marks all 23 ICs `fix`.
  *Mechanism located 2026-09-04* (critique §5.2): the loop is
  `Mito_mass_new → ψm(new) ⊣ AMPK_pT172 → Mitophagy ⊣ Mito_mass_new`, and
  cutting any one of its four links turns the young state's max Re λ from
  +0.1599 negative. ψm is produced ∝ mass and cleared first-order, so it *is*
  mass renamed, while the paper measures it as a TMRM/MTG ratio — an extensive
  variable fitted to an intensive assay and then read as the intensive gate on
  AMPK. Making the gate intensive moves the young state to −0.0099 and reverses
  the inverted dose–response (dosed/undosed SA-β-gal, day 14: 0.773 → 1.036),
  but the model stays monostable: `mitophagy_old` is 183× smaller than
  `mitophagy_new`, so damaged mitochondria are the pool the model clears least.
  *Fix:* a control arm needs a **parameter** change, not a withheld dose —
  `AMPK_T172_phos × 10` breaks the loop at its hinge and gives an unirradiated
  rest state (SA-β-gal 1.30, γH2A.X 1.05, ROS 2.88), the best any single
  constant achieves. Open question whether changing the model between arms is
  defensible. Until then no multi-hallmark demo concordance number means anything.

- [ ] **P0.43 — A driver and its consumer in different `auto_groups` buckets
  makes the consumer read a static default forever, while the driver's saved
  trajectory still shows the correct signal.** Filed 2026-09-04, and it is the
  most dangerous defect found this session because every diagnostic looks
  right.

  Measured on Kallenberger 2014 (BIOMD0000000524). Driving `CD95L` through
  `with_param_input` works exactly: a step at t=60 reproduces the time-shifted
  reference to 2.37e-06 with zero pre-step activity. Wiring the same step the
  obvious way instead — a `StepSource(timescale=None)` — puts source and
  consumer in different timescale buckets. The consumer then reads the
  ASSIGNED port's **static default** for the whole run: `p43(240) = 0` against
  a correct 27.172. **And the saved `stim/CD95L` trajectory still shows the
  step**, so plotting the input, checking the source fired, and inspecting the
  store all confirm a stimulus the model never received.

  The referee caught it only by holding an independent reference trajectory.
  Nothing about it is specific to this model: any composite whose forcing
  source lands in a different group from its target has it.

  *Fix:* a source wired to a consumer in another group is either an error or
  must force them into one group. At minimum `auto_groups` must report when a
  wiring edge crosses a bucket boundary — silently substituting a port default
  for a live signal is the exact shape of failure the validation layer exists
  to prevent. Related: `drive_step` raises `KeyError` on `CD95L` because it is
  an SBML *parameter* rather than a `w` input, so the working path
  (`with_param_input`) is not the one the helper leads you to.

- [ ] **P0.38 — Two classes of unrunnable event model are detectable from the
  trigger syntax alone, and triage imports them anyway.** Filed 2026-09-04 from
  the Stucki 2005 referee pass. Both are syntactic matches on
  `SBMLEvent._trigger_ir` and cost nothing to check.

  **Chattering complementary pairs.** Stucki's `cascade__1` triggers on
  `cascade <= 20 and c3 >= 4.5`, `cascade__2` on `cascade > 20` — exact
  complements *including the boundary*, so at `cascade == 20`, which is
  precisely where a root-finder lands, both are satisfiable and the pair
  chatters at its own root. Consequence measured in COPASI with the deposit's
  own SED-ML settings: `cascade(7000)` returns **2.0000e+01 on one invocation
  and 7.2812e+20 on another**, latch time moving 1099 → 2002. Nineteen orders
  of magnitude, same file, same solver, same tolerances. A model whose
  published readout is round-off dependent cannot be composed, and the
  overlapping-boundary test is a comparison of two IR trees.

  **Equality triggers on time.** Stucki's release event is
  `('eq', ('time',), ('const', 2000.0))`. An equality on time makes `macro_dt`
  decide *whether* the input fires at all, not merely when — `macro_dt = 10`
  put a sync point on 2000.0 and delivered it; `macro_dt = 7`, the SED-ML's own
  step, never lands there. This is P0.34 in its sharpest form: there, the knob
  shifted the dose; here it deletes it. Any `eq` against time in a trigger
  should be rejected at intake, or rewritten to a threshold crossing.

  *Fix:* both checks belong in `intake.triage_sbml` as rejects, before import.

- [ ] **P0.39 — Identically-zero states and the parameters they strand are
  invisible.** `survivin` and `sursmac` are identically zero for the whole of
  Stucki's own SED-ML run — 2 of its 8 declared outputs — which leaves **5 of
  23 parameters structurally unidentifiable**, and nothing reports it. Triage
  already performs the single solve this needs. A state that never leaves zero
  over the screening horizon, and the parameters reachable only through it,
  should be named in `ScreenReport`.

- [ ] **P0.46 — An undriven imported constant keeps the deposit's own
  experimental dose, so an unwired control arm is silently a dosed one.**
  Filed 2026-09-05. A deposit's stored constants encode *the experiment the
  paper ran*, not a resting condition. Kallenberger 2014 ships `CD95L = 16.6`
  (500 ng/ml T4-CD95L) — a saturating death-ligand dose. Compose it without
  attaching a driver to `CD95L` and every arm runs fully ligated: measured, the
  equilibration pass reached `tBid/Bid₀ = 0.9961` before the challenge window
  opened, and the "unchallenged" control had already executed apoptosis. The
  demo hides it, because the arm still integrates, still stays bounded, and
  still returns a plausible number — the same signature as P0.36.

  This is not specific to ligands. Any constant a paper set to run its own
  experiment behaves this way, and the framework currently gives no signal:
  `with_param_input` reports nothing about the constants left undriven, and
  `triage_sbml` scores the deposit as-shipped, which is exactly the dosed state.

  *Fix:* triage should report which constants carry a nonzero value that the
  model's own dynamics never change — the deposit's encoded stimulus — and
  composing should say which of those went undriven. `native_input_exposure`
  already knows a constant's flat exposure (extended 2026-09-05); the missing
  piece is naming the set and surfacing it, not computing it.

- [ ] **P0.49 — `adaptive_dt=True` makes a cycle-split composite 6-20x less
  accurate, and costs more.** Filed 2026-09-05. It is documented as
  "PLL-inspired adaptive `macro_dt` sizing off the coupling residual", i.e.
  precisely the instrument for the P0.47 case. Measured on the two-group
  multi-hallmark split against the exact solve: **65.2% vs 10.0%** at
  `macro_dt=3.5` and **38.6% vs 2.0%** at 1.0, at 1.5x the wall clock
  (0.436 s vs 0.287 s).

  A knob that trades accuracy away *and* runs slower is not a trade-off.
  **The mechanism is in the design's own pseudocode**
  ([crossgen-suggestions.md](crossgen-suggestions.md) #3): the controller's
  `rho` is `||state_after_split - state_before_split|| / ||state||`, which
  measures how much the state **moved** over the window, not how much the split
  **erred**. A cycle's damage is a stale edge — one group reading a
  macro-step-old value — and that does not present as a large state
  displacement. So `rho` sits below `rho_min`, the lock detector reads
  "locked", and the step *grows* (`adaptive_dt_max` defaults to `macro_dt * 4`)
  in precisely the regime that needed it smaller.

  *Fix:* the residual has to be measured against a **re-solve** rather than
  against the window's own displacement — which is one waveform-relaxation
  sweep (P0.47 / suggestion #4), so the two fixes are the same work. Add a
  regression asserting `adaptive_dt` never loses accuracy against the same
  starting `macro_dt`. Until then it should not be recommended for composites
  where `cyclic_group_sets()` is non-empty.

- [ ] **P0.48 — The demo's whole post-fit report path is unreachable by any
  test, and had rotted into three independent crashes.** Filed 2026-09-05.
  Every one of these sits *after* the fit converges, so a `calibrate` run did
  100 steps, produced its concordance and its fitted parameters, and then
  exited 1 — reporting failure after doing all the work, three times in a row:

  1. `calibration_report.plot_history` does `lo, hi = ref.clamp`
     unconditionally, but `ParameterRef.clamp` is documented optional and
     defaults to `None`, and none of the demo's three refs declares one.
     `TypeError: cannot unpack non-iterable NoneType`.
  2. `multi_hallmark_figures` hard-codes `arms_order = ["DDIS_vs_ctrl",
     "RAPA_vs_ctrl", "RAS_vs_ctrl"]` in `fig_concordance` and the same three in
     `fig_temporal`, but the problem defines two (`ARMS`). `KeyError:
     'RAS_vs_ctrl'`. Directly against "parametrize figures, hard-code nothing".
  3. `fig_concordance` imports `_rows_by_gene` from
     `demos.multi_hallmark_calibrate`. The function is `rows_by_gene` and lives
     in `hallsim.calibration_report` — a stale import surviving a move.
     `ImportError`.

  All three fixed 2026-09-05 (unclamped params plot against their own travelled
  range, dashed; arm lists derive from `problem.data`; the import points at the
  real module). None was caused by the Kallenberger wiring — they surfaced
  because this is the first `calibrate` run to complete in this tree, the
  outputs directory having held only `run` artefacts.

  *The defect that matters is the absence, not the three bugs.* `test_cli.py`
  asserts the subcommands exist; nothing executes them. A path that only runs
  after a multi-minute fit will rot silently every time, and did.
  *Fix:* a fast smoke test — 2 fit steps, 1 reporter, tiny span — that runs
  `cmd_run` end to end including every figure, so the report path is exercised
  on every CI run rather than the next time someone waits out a fit.

- [ ] **P0.53 — `composite_schematic` is drawn by hand and has been depicting a
  composite that does not exist.** Filed 2026-09-05. `fig_schematic` places
  every block, label and edge caption at absolute coordinates with nothing
  linking it to the composite. It still draws **ih04 / BIOMD230 / NF-κB**,
  removed on 2026-08-31, captions two edges that went with it (`mTOR -> IKK`,
  `IKKb -> IKK`), and has no block for Kallenberger. The figure is the one a
  reader would take as the composite's definition, and it is two models wrong.

  Only the readouts are derived (`readouts_for(namespace)` reads
  `MULTI_HALLMARK_REPORTERS`), which is why the reporter labels stayed correct
  while everything around them rotted.

  *Guarded 2026-09-05* — it now raises when the composite's SBML members differ
  from what is drawn, so it cannot silently produce a lie. That is not the fix.
  *Fix:* derive the blocks from `composite.processes` (name, BioModels id from
  `metadata()`, readouts as now) and the edge captions from each edge process's
  own `description`, rendering into the existing hand-tuned slots and raising
  when there are more members than slots. Everything needed is already on the
  processes; only the drawing is disconnected from them.

  Related, fixed in the same pass: `plot_runs_comparison` titled each panel by
  its store path's last segment, so the pre/post figures read `FoxO3a`, `x` and
  `y0` instead of BNIP3, DDB2 and MDM2 — correct data, unreadable labels. It
  now takes `labels` and `calibration_report` passes the gene symbols.

- [ ] **P0.63 — A large integer literal in an SBML file overflows on import,
  and the model is rejected as EXPLODING.** Filed 2026-09-06.

  Proctor 2010 (BIOMD0000000293, 140 species, 88% annotated, curated) fails
  with `OverflowError: ... Got <class 'int'> with value 11390625000000000000`
  — 1.139e19, past int64's 9.22e18. It is a rate constant, not an index, so
  it should be imported as a float; SBML has no integer type for parameters
  and the value is only an int because it was written without a decimal point.

  Rejected as `EXPLODING + TOLERANCE-SENSITIVE, max|y| = inf`, which is the
  P0.59 pattern again — a construction failure reported as a numerical one.
  P0.59's `did_not_construct` covers the composite-build step; this one raises
  inside the solve, so it slips past.

  *Fix:* coerce numeric SBML literals to float at import. One line, and it
  recovers a curated deposit.

- [ ] **P0.65 — A coupling edge with `timescale=None` gets its own scheduler
  group, so `macro_dt = span` freezes the model it couples.** Filed
  2026-09-06 (Proctor 2013 maths review, FW3).

  `HillEdge.timescale` defaults to `None`, so `auto_groups` cannot place the
  edge with the process it drives and gives it a group of its own. Run with
  `macro_dt = t_end` — a natural choice, and the one
  :func:`hallsim.intake.coupling_response` makes — that is a **single Lie step
  over the whole span with every other group frozen**. The RHS is correct
  throughout; only the schedule is wrong.

  It produced "a fully plausible, fully wrong reject-this-model number" on a
  reviewer's first pass. Nothing warns.

  `coupling_response` itself is not currently exposed — it builds a
  single-process composite, so there is exactly one group (verified on
  MODEL2201210001 and BIOMD0000000534). It becomes exposed the moment the
  function is pointed at a composite with edges, which is the obvious next use.

  *Fix:* an edge should inherit the timescale of what it drives, or grouping
  should refuse to isolate a process whose ports all bind another group's
  paths. Failing both, warn when a group contains only edges.

- [ ] **P0.68 — BIOMD0000000105 (Proctor 2007, ubiquitin–proteasome) ships
  `k69 = 0`, against the paper's `1.0E-3 s^-1`.** Filed 2026-09-07. `k69`
  is the proteasome's degradation rate, so in the deposit no substrate is
  ever degraded: over 14 days every native protein misfolds (MisP 998 of
  1000), aggregates reach 5, free proteasome and free ubiquitin go to zero.
  With Table 2's value the model sits at the paper's homeostasis (free
  proteasome 84 of 100, MisP 5.9, ubiquitin 63). The demo imports it with
  `parameters={"k69": 1e-3}` and cites the table; the deposit itself is
  what `intake.triage_sbml` scores, so its `max|y| = 998` flag is this
  defect, not a numerical one. A deposit defect for BioModels to fix.

- [ ] **P0.60 — `coupling_source_verdict` counts the reverse leg of a
  reversible binding pair as production.** Filed 2026-09-06 (Proctor 2013 F11).

  A deposit that writes forward and reverse as separate reactions makes
  `AB -> A + B` a reaction whose *product* is `A`, so unbinding scores as
  synthesis. Proctor 2013's `IL1` and `OSM` returned `suitable` — produced and
  consumed, bounded turnover — while having **no synthesis reaction anywhere**;
  both are pure decaying input pools with no autocrine loop, which is the
  defining SASP feature. That verdict promoted the deposit to lead candidate
  and cost a full review. It is the check every candidate this session was
  cleared on.

  *Fix:* decide production by conservation, not reaction role. If a species'
  total across free and bound forms is conserved, its "production" is
  redistribution. `Process.stoichiometry()` and
  `steady_state.conservation_laws` already supply the moieties.

- [ ] **P0.61 — Reporter-panel rank is not in `triage_sbml`, so a model whose
  named readouts are one variable reaches a reviewer.** Filed 2026-09-06.

  Twice the deciding defect has been linearly dependent reporters, both times
  found by hand after hours of review:

  | deposit | degeneracy |
  |---|---|
  | Hui 2016 | `ADAMTS5 == 10*IL1`; ADAMTS5/IL1A/MMP13/MMP2 are one variable |
  | Proctor 2013 | `MMP1_mRNA == MMP3_mRNA`, `MMP13_mRNA == 0.1*MMP1_mRNA` (max abs dev 8.5e-14); panel rank **7 of 9**, pairwise r = 1.0000 |

  A constant ratio cancels in a fold change, so a degenerate pair has
  *identical* log2FC by construction. Proctor's benchmark spreads those three
  genes by 1.39 log2 **with a sign flip** — unrepresentable at any parameter
  values, so calibration cannot touch it.

  *Fix:* one solve plus an SVD of the reporter trajectories. Rank below panel
  size means the extra reporters are not independent evidence and scoring them
  double-counts. Would have rejected both deposits for one solve.

- [ ] **P0.62 — `rest_residual` is normalised by `‖y₀‖`, so one large species
  hides every other state's motion.** Filed 2026-09-06.

  Proctor 2013 reports `‖f(y₀)‖/‖y₀‖ = 2.48e-05` and reads as equilibrated. It
  is not: `‖y₀‖ = 100,006` and `Aggrecan_Collagen2` alone is 100,000 of it.
  Per species, `IL1` has **τ = 0.03 h**, and two states drift with no stimulus
  — `Matriptase` to 0.6% of its IC, `TIMP3` to exactly 2× (its IC is half its
  own basal steady state). The aggregate was reported to the user as the
  candidate's strongest property.

  Affects any model with a wide magnitude spread, which is most of them.

  *Fix:* report a per-species relative rate — `|f_i(y0)| / max(|y0_i|, floor)`
  — and flag the worst, alongside the norm.

- [ ] **P0.57 — Triage cannot tell a deposit that was written for stochastic
  simulation, so it imports one as an ODE and silently deletes a mechanism.**
  Filed 2026-09-05, found independently by both reviewers on Hui 2016.

  A model deposited for Gillespie simulation is a different object from the
  same file read as a mean-field ODE, and nothing in `triage_sbml` notices.
  Hui 2016's published results are **population statistics over 500 stochastic
  runs** — every model-data comparison in the paper is a percentage of cells
  crossing a threshold. Run stochastically it reproduces the paper (200 COPASI
  Gillespie replicates: MMP-13-positive cells 13.0 ± 4.7% → 85.0 ± 4.9% against
  a measured 16.0 ± 8.3% → 80.0 ± 4.1%; 80.8% mean collagen loss against a
  stated 80.5%). Read as an ODE the same file returns 0%/100% step functions
  and is **9× wrong on `Smad1_P_Smad4`** — the species carrying the
  ALK1→Runx2→MMP-13 arm the paper credits with its MMP-13 bursts, and which is
  exactly zero at 80.6% of stochastic run-timepoints. Deterministic import
  deletes one of the paper's two mechanisms without a word.

  **The signature is declared and checkable**, verified across three deposits:

  | deposit | `substance` unitDefinition | `hasOnlySubstanceUnits` | ICs integral |
  |---|---|---|---|
  | Hui 2016 (560) | **`item`** | **True** | **65/65** |
  | DallePezze 2014 (582) | dimensionless | False | 36/40 |
  | Benary 2019 (794) | mole ×1e-9 | False | 1/17 |

  `item` is SBML for "count of things". A count-valued model with small integer
  initial amounts is a stochastic model; the ODE limit of 2 ROS and 30 Bcl-2
  molecules is not a limit anyone took.

  *Fix:* `triage_sbml` should resolve the `substance` unit and flag `item`
  (corroborated by `hasOnlySubstanceUnits` and integral ICs) as
  **stochastic-intent — mean-field import is a different model**. Integrality
  alone is not the rule: DP14 is 36/40 integral and is a concentration model.
  The same check is the admission gate for the stochastic lane, because a
  concentration model's rate law is not a propensity — see
  `docs/design-stochastic-lane.md` and P3.6.

  Two smaller items from the same review, reported but not independently
  verified here — see `docs/review-hui2016-wetlab.md`: `classify_ontology`
  returns PROTEIN for the model's three real mRNAs, one of which carries
  `bqbiol:encodes`, the qualifier that says otherwise; and
  `recommend_reporters` offers regulon candidates for genes the composite does
  not contain, and no identity mappings.

- [ ] **P0.56 — `triage_sbml` judges "not at rest" at `t_end=10` in units it
  does not know.** Filed 2026-09-05. `intake.py:188` defaults `t_end=10.0`, and
  the rest verdict is taken over that window — but the flag immediately above
  it in the same report is *"no declared time unit"*. On a model whose clock is
  seconds, 10 units is ten seconds, and nothing has moved yet.

  Measured on Hui 2016 (native seconds): `not_at_rest` reads **False at
  t_end=10** and **True at t_end=1.2096e6**, the horizon the model would
  actually be run over, with the same τ. The triage that passed it on rest is
  the one that does not know its clock.

  *Fix:* make the rest window relative to something the model asserts — its own
  slowest time constant, or the caller's horizon — and refuse to report a rest
  verdict at all when `native_time_source == "assumed"`, since the window is
  then uninterpretable. Same class as P0.40.

- [ ] **P0.54 — The calibratable surface is scalar-only, so a per-source
  parameter cannot be fitted.** Filed 2026-09-05.
  `Process.calibratable_params` does `float(getattr(self, f.name))`, so a field
  marked `calibratable` must be a scalar. `HillEdge`'s `K` and `n` are
  per-source tuples, so neither can go on the surface however much a fit would
  want them.

  This is what forced the old `HillActivationEdge` / `HillSignalEdge` split:
  the signal edge was single-source, so its `K` could be a scalar and
  calibratable; the activation edge was multi-source, so its `K` had to be a
  tuple and was not. Two classes for one rate law, to work around a scalar
  assumption one layer down. Merging them into `HillEdge(mode=...)` (2026-09-05)
  makes the constraint explicit rather than removing it: `K` is now placed with
  `place_hill_gate` and never fitted.

  *Fix:* expand a tuple-valued calibratable field into one
  `CalibratableParam` per element (`K[0]`, `K[1]`, …), which also needs
  `read_param`/`write_param` to accept an index. Then a per-source threshold is
  fittable and the merge loses nothing.

- [ ] **P0.50 — The demo's frozen-parameter rationale is stale, and wrong on
  the parameters that matter.** Filed 2026-09-05. `multi_hallmark_calibrate`
  fits **3 of 89** available calibratable parameters, freezing the rest with the
  blanket comment "non-identifiable here (gain-degenerate / flat gradient)".
  Checked by scaling each x2 on the composite and measuring the scored log2
  fold-change:

  | frozen parameter | max Δ lfc | verdict |
  |---|---|---|
  | `gz06.alpha_y` (Mdm2 degradation) | 1.53e-01 | **wrong** — most identifiable param in the problem |
  | `dp14.mTORC1_S2448_phos_by_AA_n_Akt_pS473` | 1.66e-01 | **wrong** — comparable to the fitted set |
  | `dp14.CDKN1A_inactiv_by_Akt_pS473` | 2.36e-01 | never considered; largest of any checked |
  | `dp14.JNK_pT183_inactiv` | 1.11e-02 | weak, but the only knob on BNIP3's path |
  | `dp14.CDKN1B_inactiv_by_Akt_pS473` | 6.9e-09 | genuinely flat |
  | `dp14.IKKbeta_inactiv` | 1.1e-07 | genuinely flat |

  `alpha_y` unfrozen 2026-09-05 and it is now the **most identifiable parameter
  in the fit** — rel. sens 1.00, σ 0.55 decades, against `sa_beta_gal_decay`'s
  0.95 / 0.85 — with the Fisher condition number unchanged (21 → 25). Adding it
  moved fit-arm mean|err| 0.314 → 0.281 (D07) and 0.388 → 0.344 (D14), held-out
  ρ +0.49 → +0.60, and MDM2's own error 0.357 → 0.267.

  The consequence is that **reporters with no fitted parameter on their path
  cannot move**, and the fit is dominated by one pair: `sa_beta_gal_decay` →
  GLB1 carries sensitivity 1.559, 5x the next, so the optimizer spends its
  budget there and every other reporter drifts as a by-product. BNIP3's only
  knob is `CDKN1A_transcr` at 0.268; it moved 0.048 against 0.050 predicted by
  the linearization. Nothing is wrong with the optimizer — the fit is doing
  exactly what its sensitivities allow.

  **Screened properly 2026-09-05** over all 84 candidates (forcing sources
  excluded as `u(t)`): **33 are numerically flat**. Two more freeze rationales
  fell — `mTORC1_S2448_phos_by_AA_n_Akt_pS473` ("flat gradient") reduces the
  loss by 0.557, and GLB1's production constant, frozen because "the pair sets
  a level, which cancels in a fold change", is the **largest single loss
  reducer in the fit** (residual 1.95 → 0.88). A rate whose effect runs through
  ROS dynamics does not cancel the way a bare gain would.

  The screen also caught `dp14::Cell` ranking **first** — a compartment of
  size 1, which tops the ranking precisely because it scales every rate at
  once and is therefore maximally degenerate. Compartments are now excluded
  from `calibratable_params()` at import rather than by name-matching.

  **Selecting the top 5 by loss reduction overfits — measured, not argued.**

  | | out-of-box | 4-param set | 5-param screened |
  |---|---|---|---|
  | fit D07 | 0.376 | 0.281 | **0.245** |
  | fit D14 | 0.456 | **0.344** | 0.411 |
  | held-out D07 | 0.248 | **0.148** | 0.285 |
  | held-out D14 | 0.460 | **0.428** | 0.612 |

  The screened set beats the incumbent on the arm it was selected against and
  is **worse than not fitting at all** on both held-out timepoints. The
  identifiability verdict predicted it before the fit ran: 0 identifiable
  parameters against the incumbent's 2, everything prior-dominated at
  σ 1.06-2.86 decades. Gauss-Newton loss reduction optimises exactly the arm it
  is computed on; with 12 residuals and unconstrained directions, that is
  overfitting by construction.

  *Fix:* selection must be ranked by loss reduction **and** filtered by what
  the data constrains — neither alone. Reverted to the 4-param set. The
  remaining defect is the original one: the frozen set is still justified by
  comments rather than by numbers, and `sensitivity_jacobian` should be run as
  a screen whenever the composite changes shape, since it just did and three of
  the stated rationales were wrong. BNIP3 still has no well-constrained knob on
  its own axis (`JNK_pT183_inactiv`, its only candidate, measures 1.1e-02).

- [ ] **P0.45 — The p53 → CD95 edge cannot carry a measured induction, because
  GZ06's p53 has no DC signal to give.** Filed 2026-09-05. Wild-type p53 raises
  surface Fas 3–4× (Owen-Schaub 1995). Measured in the composite, days 7–14:
  GZ06's mean p53 is `0.3519` in ctrl and `0.3504` in DDIS — the ψ-cancellation
  the DDB2 reporter already documents — while only the amplitude separates
  (`0.346–0.357` vs `0.014–0.705`). Any edge driving a *level* out of it must
  therefore rectify the oscillation, and rectification is weak: mean Hill
  occupancy at the placed `K = 0.498` gives ctrl/DDIS folds of **0.98 (n=2),
  1.04 (n=3), 1.14 (n=4), 1.92 (n=7)**. `place_hill_gate` asks for n=7; p53
  binds DNA as a tetramer, so n=4 is the ceiling with a structural basis, and
  the edge ships at 1.15× against a measured 3–4×.

  The consequence is measurable downstream. Kallenberger's `CD95act` is cubic
  in `CD95`, so the deposit converts a real 12 → 42 receptor change into
  **19.5% → 82.8% of Bid cleaved (4.25×)** at the placed ligand dose. The
  composite's own arms differ by 17.33 → 19.91 in receptor and **38.7% → 43.8%
  in death fraction (1.13×)**. The apoptosis arm is not the limitation; it
  amplifies faithfully whatever it is handed.

  This is P0.14 seen from a new direction — the demo's ctrl arm carries a p53
  startup transient (peak 0.54) as large as the damaged arm's, so the control
  is pre-sensitised (`CD95 = 17.33`, not the deposit's 12) before any damage.
  Equilibrating fixes the *sign* (ctrl 17.33 < DDIS 19.91, inverted without it)
  but not the magnitude.

  *Not a defect in the edge*, and **not what the concordance scores.** The
  numbers above are between-arm ratios at one instant. `NORMALIZATION =
  "baseline"` divides each arm by its own day 0, so `Delta_sim` is a *within-arm*
  rise over time. Measured at `macro_dt = 0.5` (P0.47 — an earlier reading of
  this table at 3.5 carried ~10% splitting error and is superseded):

  | gene | ctrl@7 | DDIS@7 | DDIS-ctrl@7 |
  |---|---|---|---|
  | CDKN1A | 1.0441 | 0.7837 | **-0.2604** |
  | GLB1 | 1.5652 | 1.2613 | **-0.3039** |
  | BNIP3 | 0.1612 | 0.0451 | **-0.1161** |
  | MDM2 | 0.2384 | 0.2053 | **-0.0331** |
  | DDB2 | 0.1896 | 0.4018 | +0.2122 |
  | FAS | 0.1897 | 0.3835 | +0.1938 |

  **Only two of six reporters separate the arms in the right direction**, and
  one of them is the new FAS. All three DallePezze reporters separate them
  *backwards* — the control arm rises more than the damaged one — which is P0.14
  stated in the scored quantity rather than in prose. By D14 every reporter has
  collapsed to =0 or gone negative.

  Against data, FAS at D07 is `+0.3835` model vs `+0.510` measured (|err| 0.126,
  the smallest of the six at that timepoint). On the held-out RAPA arm the fit
  brings it to `+0.1819` vs `+0.164` measured — |err| **0.018**, the best of any
  gene in the run.

  So the p53 -> CD95 edge is weak in absolute receptor terms and simultaneously
  one of the better-behaved things in this composite. Both readings are true;
  quote the one that matches the quantity being discussed.

- [ ] **P0.40 — `native_time_seconds = 1.0` launders a tool default into a
  fact.** COPASI writes `s` as its untouched default time unit; the importer
  reads that as a declaration and `reconciled_to` then silently trusts it.
  On Stucki the same defaults block would make `c3` a 0.71 **molar**
  concentration, so the unit is plainly not asserted by anyone. `native_time_declared`
  exists and is the right signal — the defect is that a tool-default unit is
  recorded as declared rather than as absent. Distinguish "the file says
  seconds" from "the file says nothing and the writer defaulted to seconds".

- [ ] **P0.37 — `rest_residual` is a global ratio, so one large state with zero
  derivative hides that every other state is moving.** Filed 2026-09-04.
  `‖f(y₀)‖/‖y₀‖` puts every state in one quotient. A species that is large and
  stationary contributes to the denominator and nothing to the numerator, so it
  divides the residual down and the model reads "at rest".

  Measured on Stucki 2005 (BIOMD0000001059). Reported `rest_residual` **0.0415**
  — comfortably the best of any candidate screened this session, and the reason
  it was promoted past four rejected models. The `smacmit` pool sits at 10 with
  derivative exactly 0 until its event fires, contributing ~10 to ‖y₀‖ and 0 to
  ‖f(y₀)‖. **Per state, τ is 1.2–3.0 s against a 7000 s horizon.** Nothing in
  that model is at rest.

  What it hid: with the insult removed entirely (`k7 = 0`), active caspase-3
  rises 0.7104 → 6.65 and crosses the model's own commitment threshold
  `c3 ≥ 4.5` at **t = 690 s**, 1310 s before the insult is scheduled to arrive
  at t = 2000. The apparent switch is relaxation from a non-rest IC to the
  single attractor, which sits above the threshold. The entire pro-apoptotic
  insult moves caspase-3 by **log2FC +0.11**. This is DallePezze's P0.14 defect
  in a sharper form, and the screen that was supposed to catch it reported the
  best rest residual of the day.

  *Fix:* report `rest_residual` **per state** alongside the global ratio, and
  make `ScreenReport` flag the case where the global figure is dominated by
  states with near-zero derivative. `diagnostics.rest_timescale` already
  computes per-state τ — the intake summary just does not surface it.

  **Related, and now overdue:** the zero-perturbation control run specified in
  [design-spontaneous-endpoint.md](design-spontaneous-endpoint.md) would have
  caught this in seconds, without a reviewer. It has now been the deciding
  check on two models (DallePezze, Stucki) and remains unbuilt. It belongs
  before the reviewer panel in the intake order, not after it.

- [ ] **P0.36 — A composite silently drops a model's events, so the model runs
  with its own mechanism disabled and reports plausible numbers.** Filed
  2026-09-04 after causing the same wrong conclusion **twice in one session**,
  which is what makes it a framework defect rather than a user error.

  `process_from_sbml` translates events onto the process but does not compose
  them; that requires a separate `sbml_events.expand_events(proc)` call. Put
  the process straight into a `Composite` — which is what
  `single_process_composite` does, and what every screening probe does — and
  the events are simply absent. The composite builds, validates, integrates,
  and returns a smooth trajectory. The only signal is one INFO line at import,
  `"imported N SBML event(s); compose with sbml_events.expand_events(proc)"`,
  which scrolls past in the middle of solver logs.

  Both failures were parameter-target events, i.e. the model's entire input
  route:
  - **Kollarovic 2016**: dose enters by an event assigning to `TAF`. A
    continuation sweep with events uncomposed reported *no hysteresis*, with
    max Re λ pinned at −0.0146 across the whole sweep. The model is in fact
    bistable over DDR ∈ (5.85, 12.63). The false negative went into a tracked
    doc before being caught.
  - **Stucki 2005** (BIOMD0000001059): the mechanism is three events — a timed
    Smac release at `t > 2000` assigning `k7`, and two `cascade`-triggered
    latches assigning `k4`/`k10_cascade`/`k17`. Uncomposed, an eight-point
    input sweep returned **byte-identical results at every point** and looked
    like a model that ignores its input.

  In both cases the uncomposed run is not obviously broken — it is flat,
  bounded, and tolerance-insensitive, so every existing screen passes it.

  *Fix — compose events by default; make discarding them explicit.* A warning
  is not enough: an INFO line already exists at import and both failures
  scrolled straight past it, and a `Composite(..., allow_uncomposed_events=True)`
  flag would be the same shape of defect, something to remember.

  1. **`Composite.__init__` expands a member process's events and merges the
     resulting topology rows into that process's own row.** Event ports are
     namespaced (`__set_*`, `__par_*`) so they cannot collide with user wiring,
     which makes the merge well-defined. Then `expand_events` is never a second
     call anyone can forget, and `single_process_composite` — the entry point
     both failures went through — is correct without changing.
  2. **The opt-out is a method on the process: `proc.without_events()`**,
     returning a copy with `_events = ()`. Put on the process rather than on
     the composite so the discard is visible and greppable exactly where it is
     decided, e.g. `Composite(processes={"dp14": dp14.without_events()}, ...)`.
     **Open, found 2026-09-04:** it is all-or-nothing. A model whose events
     carry both the input and the readout — Stucki 2005 has a timed release
     event and two latch events — cannot have the input replaced by an external
     `u(t)` driver without also deleting the readout mechanism. It needs to
     take event names.
  3. Composing a process that still carries events is then an error, not a
     warning, because the only way to reach it is to have gone around (1).

  **The opt-out has a real use and must exist.** Replacing a model's own
  event-delivered insult with an external `u(t)` driver is what acceptance
  test 5 in [senescence-model-rebuild.md](senescence-model-rebuild.md) asks
  for ("arms differ only in `u(t)`"), it is what the multi-hallmark demo
  already does to DallePezze's irradiation via a forcing source, and it is
  what was done deliberately to Kollarovic to drive `TAF` externally. It is
  the rare deliberate case, not the default.

- [ ] **P0.34 — A zero-delay SBML event fires at `t = macro_dt`, so a
  scheduler knob silently sets the delivered dose.** Found 2026-09-04 on
  Kollarovic 2016 (BIOMD0000000632), whose irradiation event triggers on
  `time > 0`. The event cannot fire before the first sync point, so the model
  integrates undosed until `macro_dt` and the dose lands late by that much.
  Measured: **7.3% endpoint shift between `macro_dt` = 1 h and 6 h**, with no
  warning and no indication in the result that the answer depends on a knob
  that is nominally an accuracy setting.
  This is not specific to one model — an event triggering at `t > 0` is the
  standard SBML idiom for "deliver the insult at the start", and it became
  reachable for a whole class of models when zero-delay events started
  importing (P3.0).
  *Fix:* fire due events at `t0` before the first continuous step, or refuse to
  run when an event's trigger is already satisfied at `t0` and it has not been
  applied. Either way the delivered exposure must not be a function of
  `macro_dt`; `native_input_exposure` already exists to check that a driven
  dose matches its calibration, and the same comparison should guard events.

- [ ] **P0.33 — The leading eigenvalue can belong to a subsystem that is not
  the one under investigation, so `max Re λ` looks flat while a fold sits in
  another block.** Filed 2026-09-04 after it produced a false negative on a
  live decision. Sister defect to P0.16: there the masking came from
  conservation directions and was fixed by projecting onto the leaf; here it
  comes from **block-triangular structure** and the leaf projection does not
  touch it.

  Measured on Kollarovic 2016 (BIOMD0000000632). A one-way cascade
  `DNADamage → p53 → p21` feeds the cell-cycle block and reads nothing back:
  `‖J[damage, cycle]‖ = 0.000e+00` exactly. The whole system's leading
  eigenvalue is then **−0.014600467889**, which is `k4a` = **0.014600467889**
  to twelve digits — a pure cascade mode whose eigenvector sits on `p21` and
  `p53` (both 1.000) with 0.117 on `CycECdk2a`. The cell-cycle block's own
  leading eigenvalue is **−0.31317**, two orders faster and invisible in the
  reported number.

  Consequence: sweeping a parameter and reading `max Re λ` returned −0.0146 at
  every point, which was read as "no eigenvalue crosses zero, therefore no
  fold". The model is in fact bistable over `DDR ∈ (5.852, 12.624)` γH2AX
  foci, with a saddle carrying `max Re λ` **+1.67 to +23.4** between two stable
  branches. The reported number was correct at every point sampled and
  supported none of the conclusion drawn from it.

  **`codim1_scan` does not merely miss the fold — it reaches one and throws it
  away.** `critical_eigenvalue` returns the eigenvalue nearest the imaginary
  axis. With a block-triangular mode pinned at −0.0146, that mode is the
  nearest at *every* point on the sweep including the exact fold, so the guard
  at `bifurcation.py:426` (`abs(lam.real) > 1e-3`) fires and the fold is logged
  as "Newton failure, not a fold" and skipped. The folding mode at the saddle
  is **+26.3081 /h** and never reaches the test. The guard is right in
  principle — a Newton failure should not be reported as a bifurcation — and
  wrong here because it is applied to the wrong eigenvalue.

  Two failures compound, and both need addressing:
  1. **Nothing surfaces the decomposition.** `spectrum` returns eigenvalues; it
     does not say that the leading one belongs to a block that no state of
     interest feeds back into. A Jacobian in block-triangular form has an
     exactly decomposable spectrum, which is cheap to detect and cheap to
     report. `critical_eigenvalue` and the `codim1_scan` guard must be taken
     over the block that is actually folding, not over the whole spectrum.
  2. **A single-seed continuation sweep is not a bistability test, and reads
     like one.** Both branches of a hysteresis loop are stable, so `max Re λ` is
     negative on each; only the fold itself shows a crossing, and continuation
     from one start never leaves its branch. This is the same gap Phase 0 of
     [senescence-model-rebuild.md](senescence-model-rebuild.md) already names
     ("multi-seed sweep for hysteresis") — it now has a second instance and a
     measured cost, so it should be built rather than kept as a note.

  *Second instance, 2026-09-04 (Stucki 2005):* the Jacobian's `cascade` column
  is identically zero off-diagonal — exactly block-triangular — and
  `max Re λ = +0.009` with its eigenvector supported **entirely on `cascade`**,
  a state nothing reads back. The core's `max Re λ` is 0 in all three latch
  regimes. Attaching the eigenvector's support to the reported eigenvalue would
  have made this legible without any decomposition machinery, and is the
  cheapest half of the fix.

  *Fix:* (a) report the block decomposition alongside the spectrum — per-block
  leading eigenvalues, which states each block spans, and the leading
  eigenvector's support — so a masked mode is visible rather than inferred; (b) provide the multi-seed equilibrium sweep
  Phase 0 asks for, and make the single-seed path decline to answer "is this
  bistable?" rather than answering it wrongly. Until (b) exists, no
  no-hysteresis claim from a continuation sweep is admissible evidence.

- [ ] **P0.18 — `suggest_hill_gate` exists and no one runs it, so a coupling
  edge can be placed outside its driver's entire range.** ✓✓ `psi_bridge` gates
  GZ06's ψ on `dp14/DNA_damage` at `K = 52`, `n = 2`. Crossing the p53 Hopf
  (ψ_H = 0.685416) needs `DNA_damage > 57.557`. Measured maxima: **9.6 control,
  25.97 at the published DDIS dose, 48.8 at 2× dose** — ψ tops out at 0.4397,
  36% short, so `gz06/x` is constant to 1e-4 in every arm and DDB2's arm-to-arm
  log2FC is −0.0004. K also sits **4.28σ above the largest γH2A.X count in
  DP14's own calibration data** (34.17 ± 4.17 foci). Working range is
  **7 < K < 23.5**. `screen_sensitivity` already reports
  `DDB2 ← Genomic Instability: FLAT (dead in this regime)`; nothing runs it
  either.
  *Fixed 2026-08-30.* `Composite.hill_gates()` enumerates every gate
  structurally; `CalibrationProblem.check_hill_gates()` runs at the end of
  `__init__`, comparing each `K` against its driver's realised range and
  reporting the replacement from the *same* operating ranges (one solve, not
  two). It cannot live on `Composite`: the range is a trajectory, a trajectory
  needs a horizon, and a composite has none — the calibration problem is the
  first object with one honestly. Cost is one condition-set solve and only for
  composites that declare a gate, so the unit suite is unchanged.
  `psi_bridge` was reset to **K = 10.79**, inside the Hopf-crossing window
  `8.67 < K < 24.56` (ψ crosses at `x = 1.1069·K`). Out-of-the-box sign
  agreement **13/24 → 17/24**, gz06's two reporters **0/8 → 5/8**, DDIS@t7
  **4/6 → 6/6**; DDB2 now shows the p53 limit cycle days 3–9.
  *`critical=` landed 2026-08-30 (night).*
  `hill_edge.place_hill_gate_for_crossing(off, on, basal=, hi=, critical=)`
  solves for the driver level at which the *signal* reaches a named downstream
  value — `D* = K·(h/(1−h))^(1/n)` with `h = (critical−basal)/(hi−basal)` — and
  returns the `K` window straddling the conditions plus the `margin` to the
  nearer one. Reachable from the problem as
  `suggest_hill_gate(..., critical=, basal=, hi=)`, and refuses a `critical`
  outside the edge's own range. Works for `hi` above or below `basal`.
  It succeeds exactly where the 10/90 criterion cannot: a crossing needs only
  `off < on`, no separation, so it places a gate on the r = 1.26 driver that
  `place_hill_gate` rejects with "needs n = 19". `GZ06_DAMAGE_DRIVE_K` is now
  derived from it rather than a literal, and it re-derives the 12.5% control
  margin independently of the trajectory measurement.

  The check's own finding outlives the fix and belongs to P0.14: **`DNA_damage`
  ceilings at 9.59 undosed and averages 12.13 dosed** — separation `r = 1.26`,
  needing `n = 19` for a clean gate. No Hill gate on this driver can switch,
  because the model damages itself almost as hard as etoposide does.

  *Superseded 2026-08-30 (night).* The ψ edge is gone: ψ is the paper's ξ, a
  production-noise gain, and was never a damage variable. Damage now enters on
  `alpha_x` via `damage_bridge` (K = 6.22), chosen by scanning all three
  degradation channels for a Hopf — only `alpha_x`'s damage direction crosses
  one. See the 2026-08-30 (night) diary entry. The r = 1.26 finding above
  survives intact and still bounds the result: the admissible K window is
  5.53–7.00, so the control arm sits only 12% clear of the Hopf.

- [~] **P0.29 — `Scheduler`'s three caches are keyed on a signature that does
  not identify the composite, so a reused `Scheduler` returns another
  composite's answer.** Found 2026-08-31 (external systems review).
  *Both wrong-answer halves fixed 2026-09-01; the stateless-runner half is
  still open — see the end of this entry.*
  `_integrator_cache` / `_omega_cache` / `_core_cache` (`scheduler.py:460-464`)
  are process-lifetime dicts on a mutable `Scheduler`.
  `_integrator_signature` (`scheduler.py:1315-1322`) is
  `(group_name -> sorted(process_names), int(state.shape[-1]), float(macro_dt))`;
  `_continuous_core`'s signature (`scheduler.py:948-971`) adds
  coupling/splitting/t-span/solver-class-names/dtype/jump_ts. **Neither carries
  any identity of the composite** — not topology, not process classes, not
  parameter values — while the cached core closes over the *first* composite
  (`own = composite.evolved_indices(...)`, `scheduler.py:981`; scaled tolerances
  at `:985-987`) and is later invoked with the second.

  *Measured, two independent failures.* Same `Scheduler`, two composites
  differing only in one rate constant, signature identical
  `((('default', ('p',)),), 2, 1.0)`:

  ```
  k       fresh Scheduler/arm     one shared Scheduler    rel. diff
  1e0     -5.518204e-01           -5.518193e-01           2.0e-06
  1e3     -3.686172e-07           -3.677230e-07           2.4e-03
  1e5     -3.678876e-11           +1.419227e-09           4.0e+01
  1e6     -3.678810e-13           +2.955381e-10           8.0e+02
  ```

  The gradient **changes sign** at k ≥ 1e5; the stiff arm inherits the soft
  arm's `Tsit5` verdict and goes 32 steps → **362,761 steps, 78,238 rejected**,
  well inside `DEFAULT_MAX_STEPS` so nothing raises. Second failure, wrong
  dynamics outright: two composites both `{"p": <Process>}`, `n_vars=2`, group
  `"default"`, but A evolves `a/x` while B evolves `a/y` — identical signature.
  Run A then B on one `Scheduler` and B's state never moves
  (`a/x = 1.0, a/y = 1.0` against a correct `1.0 / 0.36787948959`), a flat
  plausible trajectory with no warning and `len(_core_cache) == 1`.

  **Worse than P0.1, which it is not.** P0.1 is the *cold* cache: it warns and
  degrades toward the implicit solver, i.e. toward correctness. This is the
  *hot but wrong* cache — no warning at all, degrading toward the explicit
  solver, the direction P0.1's own text calls "finite, plausible and wrong".
  `warm_up()` makes it worse: it pins the first composite's verdict.

  **Live in the repo, verified.** `diagnostics.py:914-935` — `screen_sensitivity`
  builds one `sched` and calls `sched.run` on a new `Composite` per severity
  vector (`_build(hm)` at `:917`), so every arm after the first inherits arm 0's
  verdict. That is the framework's own sensitivity screener.
  `calibration.py:1105-1112` — `self._scheduler` is deliberately persistent so
  the verdict is "resolved once and reused under tracing"; the unstated
  assumption is that the stiffness verdict is invariant over the whole parameter
  search space, and nothing checks it. `docs/architecture.md` (Population
  studies) tells users a sweep "is not a `y0` batch; build one composite per
  arm", which produces exactly the loop that triggers this.

  **Design decision, 2026-08-31: `Scheduler` is a stateless runner.** The
  composite is an argument to `run()`, not to `__init__`, and that is intended —
  so the fix is not "bind one composite at construction". Four things contradict
  it today: the three caches above, plus `self._jump_ts` (`scheduler.py:476,
  544`) assigned per `run()` on a shared object, which also makes `Scheduler`
  non-reentrant and not thread-safe and feeds `_continuous_core`'s signature;
  and `self._warned_save_res` (see P0.30).
  *Fix:* the derived artefacts move off the runner and become a value.
  `Scheduler` keeps policy only (tolerances, splitting, coupling mode, solver
  classes); a plan object built per `(composite, t_span, macro_dt)` carries the
  resolved integrators, the traced core and the jump times. A plan carries the
  composite it was built for, so there is no key to get wrong and the collision
  class closes by construction. `Composite` cannot host this — it is a frozen
  `eqx.Module` (`composite.py:342`) that round-trips through pytree
  flatten/unflatten. Cheaper interim, not the target: put a real composite
  fingerprint in both signatures —
  `(tuple(sorted(composite.store_keys())), tuple((n, type(p).__name__) for n, p
  in sorted(composite.processes.items())), sorted topology triples)` — which
  closes the wrong-dynamics failure but **not** the gradient one, since those two
  arms are structurally identical and differ only in traced values. A stiffness
  verdict is a function of parameter values and cannot be cached on structure at
  all.
  *Missing test, by construction:* every scheduler test in
  `tests/unit/test_multiscale.py` constructs a fresh `Scheduler()` per assertion,
  so the reuse contract is never exercised and no test can fail on it. Generic
  form: run each scheduler test twice on one instance, the second against a
  differently-wired composite of the same width, and assert equality with the
  fresh-instance results.

  **Fixed 2026-09-01 (both wrong-answer halves).** `Composite`
  gained `structural_fingerprint()` — store layout, process classes, wiring —
  and it is the first element of both cache signatures, which closes the
  wrong-dynamics collision. For the gradient half a structural key cannot
  work, so `_param_digest(composite)` hashes the concrete parameter leaves and
  keys the verdict on `(structure, values)`; it returns `None` when any leaf is
  a tracer, and that case falls back to `_eager_verdict[structure]`, the last
  verdict resolved from concrete parameters. That keeps calibration's
  warm-up-then-differentiate contract exactly as it was while giving each
  eager sweep arm its own measurement. Two regression tests in
  `tests/unit/test_multiscale.py`. Discrimination measured by neutralising
  `_param_digest`: the sweep then differs by **1.6e26** relative and flips sign
  on three of four arms; with it, the arms agree bit-for-bit.
  **Stateless-runner half landed 2026-09-01.** `RunPlan` is a frozen value
  carrying the resolved groups, coupling mode, per-group integrators,
  anti-aliased `save_dt`, discontinuity times, adjoint and traced core — plus
  the composite they were built for. `Scheduler.plan()` returns one and
  `run(plan, y0=...)` executes it; passing `t_span`/`save_dt`/`adjoint`
  alongside a plan raises rather than being silently ignored. `run(composite,
  ...)` is unchanged and now resolves through a **one-entry** memo instead of
  three growing dicts: its key only has to be *conservative* (any doubt
  re-plans), where a growing cache's key has to be *complete* or it hands one
  composite's resolution to another — which is the whole defect above.
  `self._jump_ts` is gone from the runner and rides on the plan, threaded
  through `_reduced_solve` / `_group_step` / `_solve_group` /
  `_solve_group_interpolated`, so `Scheduler` no longer mutates per run.
  Verified bit-identical over 22 trajectory arrays spanning the fast, scan and
  eager paths, batched runs, Strang, both coupling modes and all three shipped
  composites: **max|delta| = 0.000e+00**. Suite 480 -> 485 passed, 1 xfailed.
  **Still open:** `self._warned_save_res` still latches once per instance
  (P0.30). And `_integrator_cache` / `_eager_verdict` remain, because
  `calibration.py` warms up on *one* condition's composite and runs all of
  them — a deliberate cross-condition sharing of the verdict that a plan
  cannot express, since each condition is a different composite. That sharing
  is now the only route left to an inherited verdict, and it is still
  unchecked (review open question 3).

- [ ] **P0.30 — `save_dt` means two different things on two of the three
  execution paths, and the anti-aliasing guardrail logs that it fired on the
  path that discards it.** Found 2026-08-31 (external systems review).
  `Scheduler.run` has three implementations of "advance the composite": the fast
  path (`scheduler.py:640-665`), the `lax.scan` path (`_run_scan_continuous`,
  `:1121`), and the eager `while t < t1` loop (`:763-905`), selected implicitly
  at `:625-638`. The eager loop — taken whenever the composite has a DISCRETE or
  EVENT process, `adaptive_dt=True`, or `debug=True` — saves at `:899-902` on
  macro-step values only, so the effective resolution is
  `max(save_dt, macro_dt)` and dense output is not used. The public docstring
  (`scheduler.py:516-520`) says the opposite: *"Output density, decoupled from
  `macro_dt` via dense output — it costs memory, not ODE steps."*

  *Measured.* `t_span=(0,10)`, `macro_dt=2.0`, `save_dt=0.1`, one 20 rad/s
  oscillator; the only difference is one added DISCRETE process:

  ```
  continuous only (scan/fast path)  save_dt=0.1 -> n_saved=319  actual dt=0.0314
  +1 DISCRETE proc (eager path)     save_dt=0.1 -> n_saved=  6  actual dt=2.0000
  ```

  **Both runs logged** `auto-reduced save_dt 0.1 -> 0.03142 (fastest oscillation
  period 0.3142; ~10 samples/period so raw-state readouts don't alias)`. On the
  eager path that refinement is computed, logged as applied, and thrown away —
  the trajectory returns 64× coarser than the grid the log line promised,
  against a 0.314 s period. A diagnostic asserting a property the code does not
  deliver removes the reason to check.
  *Blast radius:* everything reading the trajectory as the trajectory —
  `gene_reporters` summaries, `calibration`'s fold-change loss, every demo plot,
  `diagnostics.screen_*`. It hits exactly the mixed continuous/discrete/event
  composites the framework is named for, while the continuous-only composites the
  tests use are fine; that asymmetry is why it survived.
  *Two more divergences on the same seam, same root cause (three
  implementations, no equivalence test):* `self._warned_save_res`
  (`scheduler.py:465, 1448`) fires the auto-reduction notice **once per
  `Scheduler` instance, ever**, at `log.info` — invisible under the default
  config, and silent by construction on runs 2..N of a reused instance, so a
  change to the output grid every downstream number depends on is unannounced.
  And `Scheduler(debug=True)` moves execution from the scan path to the eager
  loop (`scan_eligible` requires `not self.debug`, `:637`) — a debug flag that
  changes the numerical path. They agreed to 1.1e-16 on a two-group frozen/Lie
  problem, but nothing pins that.
  *Fix:* (1) make the eager loop use `dfx.SaveAt(ts=...)` per macro step — the
  machinery exists, `_solve_group` already gets a `SaveAt` on the scan path — and
  concatenate, so `save_dt` has one meaning (~1 day). (2) Until then, **raise**
  when `save_dt < macro_dt` selects the eager path (~1 h). (3) Promote the
  auto-reduction notice to `log.warning`, drop the once-per-instance latch, and
  put the *resolved* `save_dt` into `SchedulerResult.stats` so it lands in the
  artefact rather than a log line no handler is listening for (~2 h).
  (4) The structural fix is the equivalence test in P1.16.

## P0.35 — the stop rule fired: the Scheduler is 2395× slower than the
## hand-rolled path on an event composite

- [ ] **P0.35 — `Scheduler` + `expand_events` costs 223.80 s where one jitted
  `dfx.diffeqsolve` over the same RHS costs 0.0934 s warm — 2395×.** Measured
  2026-09-04 on Kollarovic 2016 (BIOMD0000000632, 8 species, one event), both
  sides same maths, same machine, reported in
  `docs/review-kollarovic2016-maths.md` §7.
  **This is the halt condition in CLAUDE.md**, not a performance note: a user
  is 2395× better off bypassing the framework on this shape of problem, which
  is the single most informative signal the repo can produce about itself.
  The cost is **~1.2 s of fixed overhead per macro step, independent of span**,
  so it scales with the number of sync points rather than with the work done.

  **Narrowed 2026-09-04 — it is the event machinery, not macro stepping.**
  Second data point on Kallenberger 2014, 16 species and *no events*: a
  single-group composite takes the fast path and ignores `macro_dt` entirely
  (endpoint difference identically 0 from `macro_dt` 240 down to 1), and
  forcing two groups gives a warm wall of **4–7 ms total** for 1 to 60 macro
  steps — **0.1–4.5 ms per macro step against 1.2 s**. Macro stepping is not
  intrinsically expensive. The 2395x is specific to the event path, which is
  where the fix should look.
  An event composite is exactly the case that forces many macro steps, so the
  overhead lands hardest on the feature that motivated the multi-rate design.
  Note this is dispatch and orchestration cost, not solver cost — the same RHS
  integrates in 93 ms.
  *Fix:* find what costs 1.2 s per macro step and remove it. Candidates to
  measure first: re-tracing the group solves per macro step (see the "tracing
  is not compilation" invariant in CLAUDE.md — 0 recompiles is necessary, not
  sufficient), rebuilding the store view or the port dicts per step, and
  event-condition evaluation outside jit. Until this is closed, no timing
  number from an event composite means anything, and the multi-rate path
  cannot be recommended for the models it was built for.

- [ ] **P0.52 — `single_process_composite` cannot run 4% of BioModels.**
  Found 2026-09-05 while building a Jacobian corpus. The shipped helper for
  running one Process alone (`composite.py:1006`) raises on **BIOMD 87, 265,
  718, 727** with e.g. *"m/ExoXI is claimed by 2 ports of the same role with
  differing initial values"* — its identity topology collides when a model's
  `ports_schema()` emits two ports that auto-path to the same store path.
  4 of 95 cached models. The helper is what `screen_process` and every
  single-model diagnostic goes through, so those models cannot be screened at
  all.

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
- [ ] **P1.5b — Two GZ06 reporters score ~0 for a reason that is not the
  reporters.** DDB2 and MDM2 come back at −0.027 and +0.025 log2 on every
  arm-day. Both readouts are live: measured on GZ06 alone at ψ 0.3 → 1.0,
  `rms(x)` moves **+0.416** and `mean(y)` moves **+1.801**. Two separate
  defects produce the zero, and they need separating:
  1. **Identical between arms** — ψ does not differ between ctrl and DDIS,
     because `psi_bridge` gates on `dp14/DNA_damage` and dosed-minus-undosed
     damage at day 14 is −0.753 (P0.14 reaching the p53 axis).
  2. **Flat within an arm** — normalization is `baseline`, so the scored
     quantity is the day-0-to-day-14 fold change *inside* one arm. A ψ(t) that
     is constant over the run gives a constant oscillation amplitude and
     log2(RMS₁₄/RMS₀) ≈ 0. This survives even if the arms did differ.
  There is 1.8 log2 of signal available on the MDM2 axis that the composite
  fails to deliver.
  *Fix:* measure ψ(t) per arm and check `psi_bridge`'s Hill against the
  driver's real operating range — a gate placed off it pins ψ at basal or at
  `hi` for the whole trajectory, which is the placement error CLAUDE.md warns
  about. Distinct from P1.5: nothing here is a zero *gradient*, the readouts
  and the module both work.

- [ ] **P1.14 — The tolerance screen tests a tolerance nobody uses, and files
  the guaranteed disagreement as a blocker.** `screen_process` compares
  `rtol_loose = 1e-3` against `rtol_tight = 1e-7`; the Scheduler runs at
  `DEFAULT_RTOL = 1e-6`. `diagnostics.py:8-10` already documents that GZ06
  diverges at 1e-4 and is bounded from 1e-5 down — so the screen tests two
  decades past the known-bad point, gets the disagreement it was guaranteed, and
  returns `[REJECT] ... escalate=False`. The composite runs at 1e-6, inside the
  converged range. **Enforcing that verdict would have blocked a model that is
  correct where it actually runs.** Tolerance sensitivity is not a property of a
  broken model; it means use the right tolerance.
  *Fix:* ask whether the result is converged **at the operating tolerance**
  (1e-6 vs something tighter), not at a fixed pair. Better: bisect for the
  loosest rtol at which the trajectory is converged and report *that* — a model
  then carries "needs rtol ≤ 1e-5" as a property the Scheduler can check, which
  keeps the anti-damping purpose and makes the verdict actionable instead of a
  rejection. Note P0.17 is invisible to this screen either way: that failure is
  tolerance-*insensitive*.

- [ ] **P1.15 — An unweighted MSE lets one badly-fit reporter monopolise the
  fit.** The 2026-08-30 calibration dropped the loss 43% while making five of six
  reporters slightly *worse*:

  ```
  squared error    before → after
  GLB1 (d7+d14)     2.862 → 0.833     67% of the objective
  everything else   1.377 → 1.382     net worse
  ```

  GLB1 accounts for 2.03 of the 2.02 total reduction. Least squares behaving as
  specified — GLB1 was off by 1.14/1.25 against 0.02–0.59 elsewhere, so all four
  parameters went to it. Log-space fitting does not help: it reparameterises the
  *parameters*, not the residuals.
  *Fix:* pass the per-entry `weight` (precision) argument `gaussian_nll` already
  accepts, so a reporter's contribution reflects its measurement precision rather
  than the magnitude of the model's error on it.

- [ ] **P1.6 — No null-model baseline is reported.** Nothing in the run computes
  the constant null, so a concordance number is quoted with no floor to beat.
  Current state on the two-arm, 24-call configuration (2026-08-30, out of the
  box, corrected dose, damage on `alpha_x`): the composite scores **19/24** and
  the best constant predictor also scores **19/24** — a tie, not a win. The
  data is 19 up and 5 down, so "every reporter rises" is the *majority-class*
  predictor. The progression, same data and dose throughout: 13/24 (ψ at
  K = 52) → 17/24 (ψ at K = 10.79) → 19/24 (`alpha_x`), against a 19/24 null
  the whole way.

  **The deeper problem is the metric, not the missing baseline.** Sign
  agreement is accuracy on a 19:5 imbalanced set, where the choice of null is
  itself arbitrary — all-up is the best constant, all-down (5/24) the worst,
  and neither is neutral. Metrics that need no null, on the same 24 calls:

  | | model | any constant predictor |
  |---|---|---|
  | balanced accuracy | **0.868** | 0.500 |
  | MCC | **+0.607** | 0.000 |

  Confusion `TP=14 FP=0 FN=5 TN=5`: **zero false positives**, all 5 down-calls
  correct (the all-up null gets 0/5), and all 5 errors the same failure — a
  reporter the model relaxed and the data did not. Real skill, invisible in the
  tie. *Actionable:* report MCC and balanced accuracy alongside sign agreement,
  and stop quoting sign agreement alone.
  **An outside notebook (2026-08-29) implemented the check**, emitting
  `oob_null_baseline.csv` / `postfit_null_baseline.csv` per run. That work is
  uncommitted in a sandbox tree, so the reporter is not reproducible here yet —
  porting it is the actionable, and it should land before any concordance number
  is quoted again.
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

- [ ] **P1.13 — Structurally redundant parameters are invisible before a fit.**
  DallePezze's `k33` and `k34` carry the *identical* rate law
  `k·Mito_mass_turnover·mTORC1_pS2448` — the field is invariant under
  `(k33+δ, k34−δ)` to 4.4×10⁻¹⁶ and `∂endpoint/∂k33 = ∂endpoint/∂k34` to ten
  digits — so only their sum is identifiable. `k34` is *named*
  `mito_biogenesis_by_AMPK_pT172` and never reads AMPK. The paper's Figure 6A
  conclusion is an arbitrary split of one coordinate. This is visible from the
  rate laws alone, with no data and no fit, but nothing looks. Distinct from
  P1.3 (Fisher conditioning, needs a fit) and P1.5 (zero-gradient fittables,
  needs arms): this is structural and available at import.
  *Fix:* a collinearity pass over declared rate laws / stoichiometry at
  `Process` construction, naming the redundant group.

- [ ] **P1.16 — The multi-group scheduler has no in-repo workload, so every
  defect in it was found by hand-forcing a configuration.** Found 2026-08-31
  (external systems review). The largest composite in the repository resolves to
  **6 processes, 29 store paths, 35 ports — and `auto_groups()` returns a single
  group** (`{'group_0': 6}`), so it takes the `fast_path_eligible` branch: one
  `diffeqsolve` over the whole span. (Counting only; no number from that demo is
  cited as evidence about the framework.) Consequently
  `_run_scan_continuous` (`scheduler.py:1121-1310`, ~190 lines) and the eager
  `while` loop (`:763-905`, ~140 lines) are exercised only by unit tests, and
  `splitting="strang"`, `coupling_mode="interpolated"`, `adaptive_dt=True` and
  the whole `_InterpFill` / `_FrozenFill` machinery have no standing workload at
  all.
  **P0.2, P0.6, P0.10, P0.13, P0.20, P0.23, P0.29 and P0.30 are all in those
  ~330 lines.** They are not eight independent defects; they are one coverage
  gap producing defects at a steady rate, which is why several are still open
  as "unconfirmed" / "not yet understood" / "unseparated". Fixing them one at a
  time will keep producing new ones.
  *Fix:* not a bigger biological demo — CLAUDE.md is explicitly against that.
  A **synthetic multi-group conformance workload in `tests/`, run in CI**: a
  parametrized fixture over the cross-product of {1, 2, 3, 5 groups} × {lie,
  strang} × {frozen, interpolated, auto} × {adjacent, non-adjacent driving edge}
  × {with/without a DISCRETE process}, built from `hallsim.models` primitives
  only, against an analytic or refined reference. Assert (a) convergence order in
  `macro_dt` per scheme, (b) equivalence across all three execution paths — the
  structural half of P0.30's fix — and (c) that structurally irrelevant
  perturbations (inserting an inert group) do not change the answer. That single
  fixture would have caught P0.23 and P0.30, settles P0.2's order deficit with a
  number instead of an argument, and turns P0.23 from an argument into a failing
  test. 3-5 days. Highest-leverage test in the repo that does not exist.

- [ ] **P1.17 — Nothing checks HallSim against an established simulator; the
  instrument exists and is not a test.** Found 2026-08-31 (external systems
  review). `misc/tellurium_compare.py` builds the same SBML models in
  libRoadRunner/CVODE and compares per-species trajectories, describing
  RoadRunner as "the trusted stiff integrator; this is our ground-truth check" —
  precisely the right instrument. It is a one-off script in `misc/`, its
  docstring points at a wrong path (`demos/tellurium_compare.py`) and leaks a
  venv name into public-facing text against the repo's own rule, and
  `grep -rn "tellurium\|roadrunner\|copasi\|amici" tests/` returns nothing.
  HallSim re-implements a large amount of SBML semantics on top of
  `sbmltoodejax` — event translation (`sbml_events.py`, 358 lines),
  assignment-rule ordering (`_order_assignments`), `functionDefinition`
  inlining, port-boundary unit conversion (`units.py`), time-unit reconciliation
  (`reconciled_to`) — and every one is a place that can produce a
  plausible-but-wrong trajectory. **The entire correctness argument for all of it
  is currently internal consistency.**
  *Fix:* promote it to `tests/conformance/`, marked `slow` and gated on
  `pytest.importorskip("roadrunner")`, asserting a per-species relative-deviation
  bound on the bundled offline SBML. 2-3 days to make deterministic and bounded;
  it is ~80% written. Highest-value test asset available.

- [ ] **P1.18 — The validation layer emits 3 warnings and 3 false positives on
  the framework's own two-process example.** Found 2026-08-31 (external systems
  review). `simulate compose` — the README quickstart composite, the smallest
  thing the framework can build — emits:

  ```
  (graph)    Feedback loop among 2 processes: antioxidant -> ros_prod.
  (graph)    High coupling density: 100% (2/2 possible edges). Consider decomposing.
  (coupling) Potential duplication at 'cytoplasm/ROS': ros_prod.ros and antioxidant.ros
             share description terms: {'oxygen','concentration','reactive','species'}
  ```

  The feedback loop *is* the model (production and removal on one pool); the
  density advice is to decompose a two-process composite; the duplication
  heuristic fired on two processes sharing the words "reactive oxygen species".
  P3.11 diagnoses the wholesale `semantic_validation=False` as a *cost* problem
  (`nx.simple_cycles`, fixed via SCCs) and an *annotation-granularity* problem.
  Both are real, but the behavioural driver is under-measured: **a checker whose
  smallest possible input yields 3/3 false positives trains its users to switch
  it off**, and making it faster does not change that.
  *Fix:* measure precision first — what fraction of emitted warnings correspond
  to a defect a reviewer agrees with, across the composites on hand — and rank
  that above further cost work on this checker.

- [ ] **P1.19 — A composite is not bit-reproducible across a JAX pytree
  round-trip.** Found 2026-08-31 (external systems review).
  `store.build_initial_store` (`store.py:119-123`) documents this hazard and
  guards it: *"The tie-break is by name rather than by dict order on purpose:
  JAX sorts dict keys when it flattens a pytree, so `processes` comes back
  sorted from any `jax.jit` / `vmap` / `eqx.tree_at` round-trip."* The same
  hazard is unguarded in `build_rhs` (`composite.py:558`), `_assignment_pre`
  (`:479`), `auto_groups` (`:748`) and `evolved_indices` (`:600`), all of which
  iterate `self.processes` / `continuous_processes()` in dict order. Measured,
  six processes inserted unsorted, all writing one EVOLVED path:

  ```
  insertion order         : ['zeta','alpha','mu','beta','omega','gamma']
  after eqx.tree_at       : ['alpha','beta','gamma','mu','omega','zeta']
  after eqx.filter_jit    : ['alpha','beta','gamma','mu','omega','zeta']
  RHS at y0: orig = -0.600000015  roundtrip = -0.600000014  bit-identical = False
  ```

  The scatter-add accumulation order changes, so the RHS differs in the last ULP.
  Through a full solve on a six-oscillator composite the divergence stayed
  bounded — 4.8e-15 relative at t=50, 1.8e-14 at t=200, 1.2e-14 at t=1000 — so
  on this evidence it is a **reproducibility** defect, not a correctness one. No
  case was found where the adaptive controller amplified it into a step-sequence
  divergence, and none is claimed to exist.
  Why it matters anyway: `Composite.with_params` (`composite.py:875-899`) is
  implemented with `eqx.tree_at`, so *every ablation and every sweep arm* is a
  round-tripped composite compared against a non-round-tripped baseline — and
  the diary's several "bit-exact" / "max_abs_diff = 0.0" verifications depend on
  which side of a round-trip each ran on, which nothing records.
  *Fix:* iterate `sorted(self.processes)` at the four sites. One line each, no
  behaviour change beyond making the order canonical, matching the precedent
  `build_initial_store` already sets. 1 hour including a test that round-trips
  through `filter_jit` and asserts `build_rhs` is bit-identical.

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
- [ ] **P2.7 — The `ValidationReport` is computed on every construction and
  thrown away, including the interaction graph P2.1 is asking for.** Found
  2026-08-31 (external systems review). `Composite.__init__` builds a full
  `ValidationReport` including `interaction_graph` (`composite.py:407`,
  `validation.py:849-855`), then either raises or `log.warning`s its string
  form. `Composite` is an `eqx.Module` with exactly three fields (`processes,
  topology, initial`), so there is nowhere to put it and nothing attaches it.
  Consequences: there is no programmatic way to ask a constructed composite what
  its validator said (you must re-run
  `CompositeValidator().validate(comp.processes, comp.topology)` — a second full
  pass); no way to ask whether validation ran at all or with which subsystems
  enabled, since `semantic_validation` is a constructor argument leaving no trace
  on the object; and the interaction graph is rebuilt and discarded every time.
  *Fix:* add `report: ValidationReport | None = eqx.field(static=True,
  default=None)` to `Composite` and populate it — static, so it stays off the
  traced pytree. **P2.1 and P2.2 then become a formatting exercise over data
  already in hand rather than new analysis.** ~4 hours.
  Same seam as P0.31's second half: which declared writers actually contributed
  on the first trace is known at trace time and currently discarded. Recording
  it per store path is what turns "a declared EVOLVED port silently frozen" into
  something visible, and it is the same report.

- [ ] **P2.8 — The CLI configures no logging, so every `log.info` in the
  framework is invisible from the documented entry point.** Found 2026-08-31
  (external systems review). `src/hallsim/cli.py` contains **zero** `logging`
  references, while CLAUDE.md insists `simulate <command>` is *the* way to invoke
  anything. So: every `log.info` is dropped — including the auto-reduced
  `save_dt` notice (P0.30), the per-group stiffness verdicts under `debug=True`,
  and the group-ordering decisions; every `log.warning` surfaces through
  `logging.lastResort` as bare stderr text with no level prefix, logger name or
  timestamp, and no way to filter or redirect; and there is no `--verbose` /
  `--quiet` on any command.
  *Fix:* `logging.basicConfig` plus `-v/-q` on the `simulate` group callback.
  ~1 hour, and it converts a large amount of already-written diagnostic text
  from invisible to usable.

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
  Scoped 2026-09-06 in `docs/design-stochastic-lane.md`: the propensity vector
  and stoichiometry are already generated for every SBML import and discarded,
  the Scheduler already routes per-group steppers, and the only structural gap
  is that no PRNG reaches the solve path. Design is an SSA `GroupIntegrator`,
  not a fourth `ProcessKind`. Measured on Hui 2016 (a Gillespie deposit, P0.57):
  6.25e5 events/s at 256 cells on CPU, against 4.67e6 jumps/cell for a 14-day
  window — the composite's own horizon costs ~32 min. Tau-leaping buys 3.4×
  and no more, because 24 of 62 species never exceed 10 molecules.
- [~] **P3.7 — No batched parameter sweeps.** Severity varies the pytree, not
  the state, so a sweep is a Python loop.
  **Re-scoped 2026-08-30 — this is a convenience, not a blocker, and the
  headline cost was a modelling error.** The "~3.6 s rebuild+recompile per
  perturbation, ~2.5 h across 2,500 CRISPRi arms" figure was measured with
  P0.24 live *and* with the swept dimension encoded in the **topology** (a
  `ClampEdge` wired to the knocked-down gene). Topology is static, so each arm
  was a new treedef. Encode the arm as data instead — one clamp over the whole
  panel with traced `(N,)` strength and setpoint vectors, zero except at the
  perturbed gene — and the topology is identical across arms. Measured
  (`scratch/2026-08-30_vcc-perturbation-axis/`), 16 arms, CPU: **0 recompiles
  after the first arm**, and `vmap` over arms warm at **2.1 ms/arm (N=200)** and
  **7.8 ms/arm (N=1,000)**. A 300-arm sweep projects to 0.6 s + 4 s compile at
  N=200 and 2.3 s + 20 s compile at N=1,000. `vmap` over a traced parameter
  already does the job; a `param_batch` argument would only spare the user
  writing it.
  *What is actually missing is guidance:* nothing in the docs says that a
  dimension varied across runs belongs in a traced array rather than the
  topology, and the natural way to write a perturbation puts it in the topology.
  That is the defect worth fixing here.
- [ ] **P3.8 — No observable layer.** Nothing carries assay, unit, normaliser
  or transducer, so a millivolt is compared to a ratiometric dye reading.
- [ ] **P3.9 — No uncertainty.** Outputs are lines; a lab needs bands. Options,
  costs and gains ranked in
  [uncertainty-quantification.md](uncertainty-quantification.md). Two blockers
  named there are defects in their own right: the loss is not a log density
  (`gaussian_nll` means over entries and drops the ½, `data_loss` means over
  arms, `prior_weight` is a free multiplier), so any reported width is scaled by
  an unknown factor; and `identifiability.py`'s `pinv(JᵀJ)` is the MAP
  covariance with the noise scale set to 1 and the prior precision omitted, so
  `std_decades` overstates the spread and `std_tol` files parameters as
  `practical` too readily.
- [ ] **P3.10 — Dense Jacobians are materialised where the topology already
  declares sparsity — in `steady_state` *and* in stiffness analysis.**
  **Second code path confirmed 2026-08-29.** `stiffness.py`'s `jacfwd` builds a
  dense N x N Jacobian, so a 10,001-state composite dies in `analyze_groups`
  with `RESOURCE_EXHAUSTED` (763 MiB allocation, 7.8 GB peak) on a 15 GB card —
  at 381.7 s before the graph shrank and 145.9 s after, i.e. the shrink bought
  time, not headroom. This is now **the** blocker at 10k: N=100 and N=1,000 run
  end to end including gradients, N=10,000 does not. Sparsity is derivable from
  the topology dict, and a matrix-free Jacobian-vector product needs no matrix at
  all. **Stiffness half fixed 2026-08-31:** `stiffness.py` routes anything wider
  than `DENSE_JACOBIAN_MAX_DIM = 512` to `_extremal_eigenvalues`, an Arnoldi
  iteration over JVPs (`scipy.sparse.linalg.LinearOperator` + `eigs`), which
  analyses 10,001 states in 281 s where the dense path exhausted the card. So
  `analyze_groups` is no longer the 10k blocker and the "**the** blocker at 10k"
  claim below is superseded; the `steady_state` half is untouched —
  `steady_state.py:261, 273, 343, 374, 492, 514` still build dense `jacfwd`
  Jacobians, and neither reduction below exists. Original entry, on `steady_state`: ✓ External project, 2026-08-28: a
  910-node linear signalling composite × 300 clamp conditions was intractable
  under `steady_state` (damped Newton + `jacfwd`) on CPU, and was finished only
  by replacing the inner solve with a hand-written linear solve — validated
  bit-exact against `steady_state` (max diff 0.0) on a 40-node subgraph, then
  vmapped over 300 conditions in ~8 s. Two independent reductions are missing,
  and neither costs the nonlinear general case:
  - ~~**Linear fast path.** For an affine RHS, Newton converges in one step and
    the Jacobian is constant, yet it is re-derived per condition and
    iterated.~~ **Withdrawn 2026-09-01: the premise was wrong.**
    `lax.while_loop` traces its body once regardless of iteration count, and
    on an affine residual the loop exits after *one* iteration because a
    single Newton step lands the residual at exactly 0. The general path
    already reduces here. A `is_affine` detector built to exploit it measured
    **3.5 s of a 7.0 s solve at n=240** — every `jax.jvp` re-traces the
    composite RHS — so it cost more than it saved and was reverted.
  - **Sparse Jacobian.** `jacfwd` materialises a dense Jacobian — *n* forward
    passes then a dense factorisation — for a topology whose coupling graph is
    sparse and already known from the topology dict. Sparsity is derivable from
    wiring, not something the user should have to supply.
  - **The cost is in `conservation_laws`, not the Newton solve.** Measured
    2026-09-01, ring of *n* degrading nodes, CPU: `conservation_laws` is
    **12.0 s at n=240** against 7.0 s for everything else in `steady_state`
    put together, and 87% of that is `infer_conservation_laws` — eight sampled
    dense Jacobians plus an SVD of the *n*×8*n* stack. At 910 nodes that is
    ~7,280 forward passes per composite, ×300 clamp conditions. The
    hand-rolled `jnp.linalg.solve` won partly by skipping this entirely.
    *Fixed 2026-09-01 for the law-free case:* a law is kept only when
    `|L·J| ≤ rtol·|J|`, and a unit candidate has `|L·J|_max ≥ σ_min/√n`, so a
    large enough `σ_min` fails every candidate at once — checked on the
    Jacobian the verification already needs, making the guard free. A ring of
    degrading nodes (no moieties, full-rank Jacobian) goes **5.59 s → 0.84 s
    at n=120, 6.7×**; a moiety chain is unchanged (2.97 s → 2.99 s) and still
    finds all 60 laws, since the test is one-directional — a small `σ_min` may
    be a slow mode, which is exactly what the sampling exists to tell apart.
    Composites that *do* have laws still pay nine Jacobians; the exact route
    (declared stoichiometry, one Jacobian) is P1.9.
  *Rule this is an instance of: generality must never be overhead; it must
  reduce for the trivial case.* Until the sparse Jacobian lands, the framework
  still loses a large-but-easy problem to fifteen lines of `jnp.linalg.solve`.
- [ ] **P3.11 — Port semantics cannot be declared per Process, only per port,
  so generated composites switch validation off.** ✓ Same project: 910
  programmatically generated ports, each semantically identical (a protein
  activity, dimensionless), could not be annotated one at a time, so
  `semantic_validation=False` was set for the whole composite — losing unit and
  ontology checking on the *hand-written* processes too, which is where it pays.
  The validation layer assumes annotation is cheap because it was written for
  hand-assembled composites; it becomes all-or-nothing the moment ports are
  generated. Needed: a Process-level declaration (one port class inherited by
  every port it emits) and a per-process, rather than per-composite, opt-out.
  **Measured correction, 2026-08-29 — port count is not the driver, and the
  wholesale switch-off was broader than the problem required.** `Composite()`
  with validation on costs 2.28 / 2.85 / 4.82 s at 100 / 1,000 / 10,000 ports:
  ports are cheap. The cost is `nx.simple_cycles` at `validation.py:569`, which
  enumerates every simple cycle and emits one WARNING each. Cycle count is
  exponential in coupling density and `GraphAnalyzer`'s nodes are *processes*,
  not ports — warnings are flat at 47 from N=100 to N=3,000 ports, and go
  47 → 15,661 as processes go 11 → 31; at 200 genes, 32 modules produced
  3,218,727 warning lines in 36.8 s. Two consequences. Capping the output is not
  the fix — the enumeration is the cost; strongly-connected components answer
  "where is the feedback" in O(V+E) and report a mutually-coupled block instead
  of every loop through it, which is the more useful answer for a composite
  whose feedback is the point. And `composite.py:274` forwards a dict to
  `CompositeValidator(**kwargs)`, so `semantic_validation={"check_graph": False}`
  already drops just the exploding checker and keeps units, ontology and
  redundancy — the checks that are the actual argument against hand-rolling.
  *Cost fixed 2026-08-29:* `validation.py` now reports one warning per
  non-trivial strongly-connected component instead of one per simple cycle.
  A/B on one graph (200 genes, M modules, circulant): 196 / 25,860 / >=500,000
  cycles at M = 8 / 16 / 24 against 1 SCC throughout, 2.06 s -> 0.4 ms. Naming
  the mutually-coupled block once is also the more useful report. **The
  annotation-granularity half of this entry is still open** — there is still no
  per-Process port declaration and no per-process opt-out.

- [~] **P3.13 — P3.12 flattened the per-*port* slope; the per-*process* slope is
  still there, unmeasured, and superlinear in compile.**
  *Scatter fusion landed 2026-09-01: 23 -> 15 eqns/process, compile 1.8x. The
  slope is still linear in process count — see the end of this entry.* Found 2026-08-31
  (external systems review). P3.12's result — jaxpr "871 equations at N=300,
  3,000 and 10,000 alike" — is flatness in *port count within one process*, and
  it reproduces. The composition axis, **process count**, which is the
  framework's whole reason to exist, is linear in jaxpr and superlinear in
  compile, and was not measured anywhere in this file or in `diary.md`.
  Identical mathematics (a ring of N first-order decays with nearest-neighbour
  coupling) spelled two ways — N single-port processes vs one process with an
  N-wide block port. CPU, `HALLSIM_COMPILATION_CACHE_DIR=off`:

  ```
  N     spelling      Composite()  build_rhs  RHS trace   jaxpr eqns   jit compile
  100   N processes   0.305 s      8.5 ms     155 ms       2,701        0.67 s
  100   1 block port  0.001 s      0.5 ms       5 ms           24        0.03 s
  400   N processes   0.148 s     31.9 ms     682 ms      10,801        4.01 s
  400   1 block port  0.003 s      1.3 ms      11 ms           24        0.04 s
  800   N processes   0.298 s     64.9 ms   1,339 ms      21,601       11.66 s
  800   1 block port  0.005 s      2.5 ms      20 ms           24        0.04 s
  1600  N processes   0.600 s    131.9 ms   3,131 ms      43,201       38.33 s
  1600  1 block port  0.014 s      4.9 ms      39 ms           24        0.04 s
  ```

  jaxpr is exactly **27 equations per process** (`27N + 1`), flat at 24 for the
  block. Compile grows as roughly **N^1.7** (100→400 is 6.0× for 4× N; 800→1600
  is 3.3× for 2× N) — **38 s to compile a single RHS evaluation at N=1600**,
  before any solve, ~1000× the block spelling and widening.
  *Where the 27 come from* (measured per-process primitive counts): `4
  convert_element_type, 4 mul, 3 broadcast_in_dim, 3 add, 2 device_put, 2 lt,
  2 select_n, 2 slice, 2 squeeze, 1 gather`, plus the scatter. `slice` +
  `squeeze` are `_port_view` (`composite.py:243-252`) unpacking one gather into
  per-port scalars — P3.12 identified this as "the dict-of-scalars interface
  itself" but scoped it to ports; for a composite of scalar-ported processes it
  is also the per-process cost. `lt` + `select_n` + `add` + `device_put` are
  **one `.at[cols].add(vals)` scatter per process** (`composite.py:346`) plus its
  `np.concatenate` of static index/factor arrays at `composite.py:334-335`, done
  inside the traced `__call__`, per process, per trace.
  *Fix, available and cheap:* `_FlatRHS.__call__` (`composite.py:317-348`)
  already holds every process's `cols`/`facs` as static numpy. Concatenate all
  processes' index arrays **once at build time** and emit a **single** scatter
  over the concatenated contribution vector instead of N scatters. That collapses
  `lt`/`select_n`/`device_put`/scatter-`add` from 4N to 4 and hoists the two
  `np.concatenate` calls out of the traced body. Write semantics are unchanged —
  duplicate indices in one `.at[].add` still sum, which is EVOLVED's contract,
  as P3.12's own migration notes established. Estimated: removes ~10 of the 27
  eqns/process and, more importantly, N separate scatters from the XLA graph,
  the likeliest source of the N^1.7. 1-2 days including a bit-exactness test
  against the current path and a `jaxpr_eqns(N)` **slope** test alongside
  `test_block_port_rhs_is_flat_in_width`. The slope test is the important half:
  without it the next refactor re-introduces the per-process scatter unnoticed.
  *Not measured, wanted:* the same sweep with a full `Scheduler.run` compile
  rather than a bare RHS trace, and with reverse-mode. Both run well past a
  minute at N≥800; the RHS is the inner loop of both, so the N^1.7 is expected to
  carry through and be worse.

  **Scatter fusion landed 2026-09-01.** `_FlatRHS.__call__` now collects every
  process's contribution and emits **one** scatter over the concatenated index
  array; the two `np.concatenate` calls left the traced body. A/B under the
  same machine load, the old spelling monkeypatched back in (visible to
  `make_jaxpr` and a freshly-built `jit`, unlike the Scheduler's `filter_jit`
  core, which is keyed on treedef and silently returns the compiled fused
  version — a probe of mine was blind to exactly that before it was caught):

  ```
    N      scatter     trace     eqns   compile
  100  per-process    208ms     2301     0.68s
  100        fused    193ms     1509     0.36s
  200  per-process    483ms     4601     1.38s
  200        fused    248ms     3009     0.78s
  400  per-process    895ms     9201     2.62s
  400        fused    500ms     6009     1.47s
  400  1 block port     5ms       20     0.06s
  ```

  **23 -> 15 equations per process** (`15N + 9`), compile 1.8x faster at every
  N. Guarded by `test_per_process_cost_does_not_regress` (a *process*-count
  slope test, the axis nothing measured) and
  `test_fused_scatter_still_sums_duplicate_writes`.

  **Not bit-exact, and the residual matters less than the fact of it.** 27 of
  28 reference arrays match exactly, including two composites built
  specifically so many processes write the same columns. `mitochondrial` moves
  **9.095e-13 absolute / 7.450e-15 relative** — bounded, non-accumulating,
  1000x below the solver's `atol=1e-9`. Evaluated eagerly the RHS is
  bit-identical, so it is not accumulation order; it is XLA rounding the
  restructured graph differently (`vals * facs` now runs on one long vector
  instead of one per process, so vectorization and FMA contraction differ).
  Recorded rather than absorbed into the gate: the "bit-exact" claims elsewhere
  in this file are only worth something if the threshold is not moved when one
  fails.

  **Still open: the slope is linear, not flat.** 15 eqns/process is a smaller
  constant on the same axis. The residual is `gather`/`slice`/`squeeze` per
  process — the `dict[str, scalar]` interface, which is the same design
  question P3.12 answered on the port axis and has not been answered on the
  process axis. A block-ported process stays at 20 equations at any N, so for a
  generated network the answer is still "emit one process, not N" (review open
  question 2).

- [ ] **P3.14 — The module layering is acyclic by static import order and a
  14-module cycle by actual dependency; the base abstraction imports the
  optimizer.** Found 2026-08-31 (external systems review). Measured over the 40
  modules under `src/hallsim/`: **top-level imports are 57 edges with zero
  cycles** — the static layering is clean. Counting function-local imports it is
  **98 edges and one strongly-connected component of 14 modules**: `calibration,
  composite, coupling_wiring, hallmarks, models.hill_edge,
  models.running_integral, process, reporter_wiring, scheduler, steady_state,
  stiffness, store, units, validation`. Nineteen modules carry deferred
  `hallsim` imports specifically to break a cycle. The lazy imports make Python
  happy; they do not make the design acyclic, and none of it is visible in the
  import graph a reader sees at the top of a file — so a contract change in any
  of the 14 can require edits in the other 13.
  Sharpest instance, one line: **`process.py:354` — `Process.calibratable_params`
  does `from hallsim.calibration import CalibratableParam`.** The base
  abstraction of the framework imports the optimizer module to construct its
  return value. `process` has fan-in 20, the highest in the repo; `calibration`
  is the largest file (2,147 lines) and the most likely to change.
  *Fix:* move `CalibratableParam` (and `ParameterRef`'s pure-data half) into a
  leaf module — `hallsim/params.py`, or `process.py` itself, since that is where
  it is produced. Same for `composite.py:875` (`from hallsim.calibration import
  CalibratableParam`) and `composite.py:876` (`from hallsim.hallmarks import
  HALLMARK_REGISTRY`, a module-level mutable registry read from inside a
  `Composite` method — already overridable via `registry=`, so the default lookup
  could move to the caller). Half a day, and it shrinks the SCC materially.

- [ ] **P3.15 — Three modules build what an established tool already solved.**
  Found 2026-08-31 (external systems review), against the repo's own principle
  "if COPASI / Tellurium / AMICI solved a problem, borrow it".
  - **`bifurcation.py` (459 lines) — push hardest.** It hand-rolls Newton
    continuation, and its own docstring names the limitation that
    pseudo-arclength continuation exists to remove: *"a branch is followed only
    until it folds — tracing both arms of a hysteresis loop needs a multi-seed
    sweep."* That is the defining feature of AUTO-07p, MatCont, BifurcationKit
    and PyDSTool. **P3.2 ("no real-eigenvalue continuation") is a symptom of
    building rather than borrowing.** The autodiff-through-JAX argument is
    genuine for the *normal-form coefficients*; the continuation stepper itself
    is undifferentiated.
  - **`steady_state.conservation_laws` — borrow.** P1.9 already records that
    moieties are inferred numerically. Exact structural conservation from the
    stoichiometry matrix null space is what COPASI/libRoadRunner ship and what
    `Process.stoichiometry()` was added to enable; finish wiring it rather than
    improving the numerical inference.
  - **`identifiability.py` (270 lines) — borrow the method, keep the code.**
    P1.13's own note says `pinv(JᵀJ)` omits the prior precision and the noise
    scale. pyPESTO / dMod solved this; copying the formulation is cheaper than
    rediscovering it.
  *Justified builds, explicitly not on this list:* `xpp_import.py` (no library
  does this well), `_interp_uniform` (static shape is a hard `lax.scan`
  requirement and the docstring says so), `_ReducedRHS` (nothing off the shelf
  gives a group-restricted implicit solve), `_per_member` batching.

---

- [ ] **P3.16 — The main constructors are flat parameter walls, and three of
  the knobs silently select between three different execution paths.** Found
  2026-08-31 (external systems review); the only finding from that report with
  no entry until now. Measured on the current tree: `Scheduler.__init__` takes
  **27** parameters, `CalibrationProblem.__init__` 20, `Calibrator.__init__`
  19. Eight of the Scheduler's are the `adaptive_dt_*` cluster. The 2026-07-02
  internal review already flagged ~25 and asked for `AdaptiveDt(...)` /
  `Splitting(...)` value objects; the surface grew instead.
  The sharp half is not the count. `splitting`, `coupling_mode` and
  `adaptive_dt` are documented as independent knobs, but they *select the
  execution path*: `adaptive_dt=True` or a DISCRETE/EVENT process forces the
  eager loop, `debug=True` moves off the scan path (`scheduler.py:637`), and
  everything else takes the fast or scan path. So three tuning parameters
  choose between three implementations of "advance the composite" that do not
  agree with each other — P0.30 (`save_dt` means two things) and P0.23
  (interpolated coupling is frozen for non-adjacent groups) are both defects
  *of that selection*, not of the knobs.
  Against the stated goal — "the framework must be easy for AI agents to use"
  — a 27-slot flat constructor where three slots silently reroute the numerics
  is the wrong shape: nothing in the signature says those three interact, and
  nothing in `SchedulerResult` says which path ran.
  *Fix:* group the clusters into value objects, and put the resolved path,
  coupling mode and `save_dt` into `SchedulerResult.stats` so the selection is
  an observable fact rather than an inference. The `stats` half is worth doing
  first and independently — it is a few hours and it makes P0.30/P0.23-class
  divergences visible instead of silent.

## Review notes — 2026-08-31 external systems review

Where the reviewer thought this document's own priorities disagree with its
stated rules. Recorded rather than acted on; triage decisions are the
maintainer's.

- **P3.10 and P3.11 are filed P3 while both appear in the standing-criterion
  table**, which says an edit a user was forced to make is "a P0 regardless of
  how small the edit was". P3.10's own text calls itself "**the** blocker at
  10k" and records a user replacing the inner solve with a hand-written one. By
  this document's own rule they are P0s. The scheme disagrees with itself in
  writing, which will cost the next person triaging by number.
- **P1.1 (no `check_gradient`) should be P0.** End-to-end differentiability is
  stated as a must; P0.1 shows gradients can be finite garbage and P0.29 is a
  second, silent route to the same outcome. `check_gradient` is not a
  nice-to-have validator — it is the **detector** for the highest-severity
  failure class in the system, and the only item on this list that would have
  caught both P0.1 and P0.29 without anyone suspecting them first. Build the
  instrument before the next bug.
- **P1.2 + P1.14 together are a documentation-code disagreement on a safety
  gate, and neither says so.** CLAUDE.md's intake protocol calls tolerance
  sensitivity "the load-bearing check" and makes screening mandatory; P1.2 says
  `screen_process` never varies `atol`; P1.14 says the `rtol` pair it does vary
  is two decades past the known-bad point and produces a guaranteed false
  REJECT. So the mandatory gate does not perform the check the protocol calls
  load-bearing, and performs a different one calibrated wrong. That is a P0 in
  the "config silently disagrees with the code" sense — the protocol document is
  the config.
- **P0.4 and P0.31 are one rule, not two instances.** "If every mapping of an
  applied hallmark misses its target, raise" and "if a returned port name matches
  nothing, raise" are both *an operation that resolves to nothing must say so*.
  Worth generalising.
- **P0.2, P0.6, P0.10, P0.13, P0.20, P0.23, P0.29, P0.30 are one coverage gap**,
  not eight defects — see P1.16.

### Open questions the review could not settle from the code

1. *Answered 2026-08-31:* **is a `Scheduler` reusable across composites?** Yes —
   it is a stateless runner. See P0.29 for what that decision implies.
2. **What is the intended shape of a "thousand-port" generated composite — one
   block-ported process, or a thousand processes?** P3.13's answer differs by
   1000× in compile. If the former, the README should say so and
   `docs/architecture.md` should show the generator pattern; if the latter,
   per-process scatter fusion is a prerequisite, not an optimisation.
3. **Is the stiffness verdict assumed invariant over a calibration's parameter
   search space?** `calibration.py:1105-1112` depends on it; nothing checks it.
   If unsafe, the fit needs a periodic re-verdict; if safe, assert it — measure
   the verdict at the final parameters and fail the run if it moved.
4. **Why do `adaptive_dt` and the eager loop still exist?** The diary records
   `adaptive_dt` as Pareto-dominated. Deleting it removes a 7-parameter cluster,
   one of three execution paths, and one of three implicit path selectors.
5. **What is the acceptance test for "the composition is correct"?** Today it is
   internal consistency plus review panels. `misc/tellurium_compare.py` implies
   the intended answer is RoadRunner/CVODE — see P1.17. Whether cost, flakiness
   or dependency weight is what keeps it out of `tests/` decides whether that is
   a 2-day job or a blocked one.
6. **Is `Composite` intended to stay a three-field frozen module?** Several
   things want to live on it — validation report, interaction graph, resolved
   stiffness verdict, cached `store_keys` — and all are hashable-or-static. A
   deliberate "no, it stays minimal" is a fine answer; an accidental one is
   costing observability (P2.7, P0.29).
7. **Who is the first real user, and what is their first composite?** Everything
   here ranks differently depending on whether that composite is three published
   SBML models on one clock (P0.30, P0.23, P1.16 barely matter; P3.13 not at all)
   or a generated network of hundreds of nodes (P3.13 and P0.29 dominate; the
   SBML path barely matters). The repo currently invests in both and has a
   workload for neither.
8. **Is `demos/` shipped in the distribution on purpose?** `pyproject.toml`
   includes `demos*` because the `simulate` entry point imports it, which makes
   example biology part of the public artefact — against `docs/architecture.md`'s
   own "specific biology is **not** part of the package". Worth deciding whether
   the CLI should live behind an extra.

The full report, including what the reviewer judged well designed and should not
be broken, is the gitignored `docs/review-architecture-systems.md`.
