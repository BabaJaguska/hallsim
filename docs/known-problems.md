# Known problems

Defects found by review, ordered by priority. Distinct from
[roadmap.md](roadmap.md), which is planned work — everything here is something
that is wrong now. Each entry carries the evidence that established it, so
nothing has to be re-argued.

Evidence sources: the mitochondrial stress test (2026-08-18), the
multi-hallmark demo review (2026-08-19), and the DallePezze 2014 referee pass
(2026-08-25) — all by the review panel in `.claude/agents/` — plus two outside
models calibrated against GSE248823 by agents who did not have this list.
Findings confirmed by two independent reviewers are marked ✓✓. The panel's raw
reports are gitignored; what survived review is here.

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

- [ ] **P0.2 — Strang splitting does not attain second order.** Re-measured
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

- [ ] **P0.23 — `coupling_mode="interpolated"` may be a silent no-op.**
  Measured 2026-08-31: bit-identical to `frozen` at `macro_dt` 3.5 / 1.75 /
  0.875, five significant figures, all four observables.
  `_effective_coupling` passes an explicit mode straight through and
  `_run_scan_continuous` sets `interp = coupling == "interpolated" and
  n_groups > 1`, which was true — so it was requested and enabled, and produced
  no difference. Either the flag does not reach the coupling, or the interpolant
  coincides with the frozen value to 5 s.f., which is not plausible.
  Unconfirmed — the mechanism was not traced. *Fix:* establish which, with a
  test that fails if interpolated and frozen agree on a problem with a forward
  cross-group edge.

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
  **The exposed surface doubled on 2026-08-29:** Deregulated Nutrient Sensing
  now targets `nutrient_drive.after` the same way, so a composite built without
  that source silently loses the mTOR dial too.
  *Fix:* raise when every mapping of an applied hallmark misses its target.

- [x] **P0.5 — `_substitute` overwrites `eqx.tree_at` edits on fitted
  `ParameterRef`s.** *Fixed 2026-08-23.* `CalibrationProblem` snapshots each
  fitted field at construction and `_substitute` raises on an edit it would
  overwrite, pointing at `with_overrides` — the single route for changing any
  parameter, fitted or not, by fittable name or by `<process>.<field>` address.
  Overrides are applied last, so they outrank both the fitted iterate and the
  composite's own value; no caller has to know which list a parameter is in.
  Verified on the multi-hallmark demo: zeroing `mtor_nfkb.k_act` — one of the three edges
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
  keeping timescale order on a cycle. The multi-hallmark demo's dp14/gz06 group now
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
  *Fix:* a control arm needs a **parameter** change, not a withheld dose —
  `AMPK_T172_phos × 10` breaks the loop at its hinge and gives an unirradiated
  rest state (SA-β-gal 1.30, γH2A.X 1.05, ROS 2.88), the best any single
  constant achieves. Open question whether changing the model between arms is
  defensible. Until then no multi-hallmark demo concordance number means anything.

- [x] **P0.15 — `conservation_laws` returns rows that are not normalised, so
  `LᵀL` is not a projector.** *Fixed 2026-08-25, in the commit that filed it —
  the checkbox was missed.* Rows come back **orthonormal** via
  `_orthonormal_rows`, on both the declared-stoichiometry and the inferred
  path, and the docstring now states `LᵀL` is the projector onto the conserved
  directions. Guarded by the two tests the fix line asked for:
  `test_laws_are_orthonormal` and `test_projector_step_stays_on_the_leaf`
  (`tests/unit/test_steady_state.py:116,126`), both passing.
  What it replaces: rows mutually orthogonal but with squared norms
  `L Lᵀ = diag(2,2,2,3,2,1)`, so projecting with `LᵀL` — the obvious use, and
  the documented one — silently left the conservation leaf. Cost a reviewer a
  basin scan that looked multistable and was not.

- [x] **P0.16 — `bifurcation.equilibrium` and `hopf_scan` report zero equilibria
  for any model with a conserved moiety.** *Fixed 2026-08-28.* `equilibrium`,
  `spectrum`, `critical_eigenvalue`, `first_lyapunov_coefficient`,
  `fold_coefficient` and `codim1_scan` all take `laws=`. With it the Newton
  runs on the pinned residual (`steady_state.pin_conserved`, one definition
  shared with `steady_state`) and the spectrum is read on the leaf tangent
  space (`steady_state.leaf_basis`). Without it a singular Newton step now
  logs what is wrong instead of returning `None` in silence.
  On DallePezze (6 laws over 23 states) the search returns the
  late-senescence fixed point — SA-β-gal 9.0315, DNA_damage 7.2781, ROS 19.9426
  — and the spectrum splits into the 6 conserved zeros and 17 real modes at
  max Re λ = −0.072969, reproducing the referee's hand-derived −0.0730.
  The second half — real crossings, which is every crossing in that model —
  is covered by `codim1_scan` replacing `hopf_scan`: detection is by change in
  unstable dimension rather than by watching one complex pair, a vanished
  branch is bisected and kept only if the critical eigenvalue really reached
  zero, and each crossing is classified fold or Hopf with its normal-form
  coefficient plus, for a fold, the parameter transversality that separates a
  saddle-node from a transcritical or pitchfork crossing.
  Regression: `test_bifurcation.py` — the analytic two-state moiety, the
  DallePezze endpoint, and the three real normal forms.
  **Remaining limit** (documented, not a defect): continuation is plain
  Newton, so a branch is followed only until it folds. Where a fold joins two
  *stable* branches, Newton steps across to the other arm and the scan sees
  no change; tracing a full hysteresis loop needs a multi-seed sweep per
  parameter value.

- [x] **P0.24 — `write_param` undid the array coercion, so every parameter
  value recompiled.** ✓ External project, 2026-08-29. `write_param`
  (`process.py:82`) is `eqx.tree_at` throughout, and equinox skips
  `__check_init__` on `tree_unflatten` — which `__check_init__`'s own docstring
  states. So construction coerced floats to arrays and the *supported setter*
  handed them straight back as Python floats: static leaves, one compile per
  distinct value. Measured 2.5–2.7 s per `with_params` call at N=1,000 under
  `jax.log_compiles`; compile counts for values 0.02/0.05/0.02 were 1/1/0 with a
  float and 0/0/0 with `jnp.asarray`. This is the exact invariant CLAUDE.md's
  "structure is static, values are traced" exists to protect, on the route P0.9
  added to be the one supported way to set a parameter.
  **Why the guard missed it:** `test_parameter_change_does_not_recompile` built
  its sweep with `eqx.tree_at(..., jnp.asarray(r))` — applying the coercion
  inside the test, so it exercised the hand-rolled route `write_param`'s
  docstring tells callers not to use, with the defect pre-fixed.
  *Fixed 2026-08-29:* `write_param` coerces through `_as_traced`;
  `test_with_params_yields_a_traced_array` and
  `test_with_params_sweep_does_not_recompile` go through the public route with a
  plain Python float, and both fail without the fix.

- [x] **P0.25 — An unrecognised topology entry is skipped silently, and it
  decides splitting order.** ✓ External project, 2026-08-29, found by running
  rather than by grep. `scheduler.py:1068` passes over a topology entry it does
  not recognise without warning, and that loop determines group ordering and
  frozen-vs-interpolated coupling. A composite half-migrated to any new port
  form therefore runs, returns finite numbers, and has mis-ordered its
  operator splitting. *Fixed 2026-08-29:* raises on an unrecognised entry, which also makes a
  port-representation migration safe to do incrementally.

- [x] **P0.21 — A coupling edge could be dead, saturated or sign-inverted and
  only warn.** ✓✓✓ Found across all three NF-κB reviews, 2026-08-31, and the
  fourth instance of the same pattern: the framework printed a correct warning
  and nothing acted on it. `psi_bridge` sat at `K = 52` against a driver
  reaching 27.18 for the whole of its life; `ikkbeta_nfkb` activated on
  `dp14/IKKbeta`, which is **higher in control (33.65 / 22.3 / u=0.1106,
  three reviewers) than in DDIS (22.44 / 18.0 / u=0.1038)**, so it fired
  hardest where the perturbation was absent.

  *Fixed 2026-08-31:* `check_hill_gates` now **raises**. A gate outside its
  driver's realised range and an activating edge whose driver is higher in
  every reference condition than in any perturbed one are both definite
  defects — no value of `K` repairs a sign — and a warning about a definite
  defect is a warning nobody acts on. `allow_dead_edges=True` is the hatch,
  mirroring `allow_unidentifiable`. Both checks reuse the operating ranges
  `check_hill_gates` already computes, so they cost nothing extra. The
  composite as it stood on 2026-08-30 would now refuse to construct, twice.

- [x] **P0.22 — Ihekwaba 2004 removed from the multi-hallmark composite.**
  ✓✓✓ Refereed by all three panel agents (`docs/review-ihekwaba2004-wetlab.md`,
  `docs/review-nfkb-maths.md`, `docs/review-nfkb-physics.md`). The deposit
  itself is sound — maths returned "accept", and the undeclared time unit turned
  out to be seconds, confirmed three ways (58/64 constants are Hoffmann 2002 ÷
  60; period 98 min against a measured ~100). What failed was every seam:

  - **Inert.** 19/24 with both edges live, ablated, or ×10; bit-identical arms
    when ablated.
  - **Driven, not perturbed.** `v64` is an IKK sink with no source anywhere, so
    the edges supplied 100% of the module's IKK; solo it decays to 1.3e-14, and
    `[IKK]* = u/10.368` predicts all three arms to <1%. A 24-state oscillator
    collapsing to a one-dimensional static curve.
  - **Backwards edge** (P0.21 above).
  - **No SASP.** Its only NF-κB-inducible transcript is its own inhibitor — no
    IL6, CXCL8, IL1A, CCL2 or MMP — so it could not emit what the data actually
    moves: **CCL2 +3.05, CXCL1 +2.68, ICAM1 +2.55, IL8/IL6 +1.73** log2FC at
    D14, nine SASP genes above the 96th percentile of 23,104.
  - **Wrong reporter class.** NFKBIA flips sign (−0.36 at D07, +0.32 at D14) —
    IκBα is the early, dose-independent, pulse-tracking target, while the SASP
    genes are the late persistence-requiring class.

  *Consequences of the removal, all measured:* the composite drops to
  **16/20 against an 18/20 majority null** (it had tied at 19/24), because
  NFKBIA had been supplying **3 of the 5 correct down-calls by predicting
  "down" constantly** — scoring like a null while being one. Only **2 negative
  calls remain in the whole evaluation set**, so specificity is estimated from
  n=2 and the metric can no longer discriminate. Fixing that needs reporters
  with real dynamic range in both directions; the data offers an obvious
  down-program — **MKI67 −3.60, TOP2A −3.57, BUB1 −3.28, CCNA2 −3.10, LMNB1
  −2.79**, nine cell-cycle genes below the 1st percentile.

  *Unexpected and load-bearing:* dropping 24 states took the composite from
  three timescale groups to **one**, so it now uses the single-group fast path.
  No operator splitting, no `macro_dt` (verified: bit-identical across an 8×
  refinement), and therefore **P0.2's 17% Lie error and P0.20's span truncation
  no longer apply to it**. Every day-14 number produced while it had three
  groups carries that splitting error.

- [x] **P0.20 — The multi-group scan silently ran a shorter span than it was
  asked for.** ✓✓ Found 2026-08-30 by the NF-κB physics review, reproduced on a
  two-process toy. `scheduler.py:1121` sized the fixed-length `lax.scan` as
  `int(round((t1 - t0) / macro_dt))`, i.e. the *nearest* whole number of macro
  windows rather than enough to reach `t1`. The scan body already clamped its
  last window (`jnp.minimum(t_start + macro_dt, t1)`), so a short final window
  was supported — only the count was wrong. Nothing raised.

  The multi-hallmark demo sat exactly on it: `t_start=-1.0`, `t_end=14.0`,
  `macro_dt=3.5`, so `round(15/3.5) = 4` windows covered 14.0 of 15.0 and the
  run **stopped at t = 13.0**. Every day-14 reporter was the day-13 value, read
  by `jnp.interp` clamping to the last sample — and all five of the run's sign
  errors were at day 14 while both day-7 panels scored 6/6. Rounding could
  overshoot as easily as undershoot; only the sequential path was safe, because
  it steps `while t < t1 - _TIME_EPS`.

  *Fixed 2026-08-30:* add a window when the rounded count does not reach `t1`.
  The three sibling `round()` sites are fine and were checked — `_save_grid`
  uses `linspace(t0, t1, n)`, which pins both endpoints whatever `n` is; the
  subsample stride re-appends `n_macro` explicitly; the discrete-firing check
  floors with an exact-alignment branch. Guarded by
  `TestSpanIsCoveredWhenMacroDtDoesNotDivideIt`, which asserts the *integrated
  value* as well as the endpoint, since a run can label its last sample `t1`
  while having integrated less.

  **It did not change the headline.** Re-scored, the composite is still 19/24;
  every day-14 value moved further in the relaxing direction and no sign
  flipped, so "the model produces an acute response and no durable senescent
  state" is reinforced rather than overturned. Day-7 values moved ≤0.4%, from
  the macro-window boundaries shifting — which incidentally bounds the Lie
  splitting error here at well under a percent.

- [x] **P0.19 — A device OOM is reported as a tracing failure and answered by
  choosing the wrong solver.** ✓ External project, 2026-08-29.
  `scheduler.py:1490` catches bare `RuntimeError`, and
  `issubclass(JaxRuntimeError, RuntimeError)` is `True`. An OOM inside
  `stiffness.py:112` at 10,001 store paths was swallowed and
  `scheduler.py:1503-1518` logged *"cannot measure group stiffness under tracing
  (grad/jvp/vmap)"* with no tracing in progress, then degraded every group to
  `Kvaerno5` — routing the composite onto a dense 10,001×10,001 Newton solve it
  had just proved it could not allocate. The measured spectral abscissa is flat
  at 20 (pure neural) / 70 (mixed) across N=100…3,000, so `Tsit5` is correct at
  every N. The remedy the message suggests, `warm_up`, is the call that failed.
  *Fixed 2026-08-29, verified by measurement:* `stiffness.py` raises a named
  `StiffnessNotConcrete` (a `RuntimeError` subclass) at both sites and the
  scheduler catches that, so resource errors propagate. Re-running the N=10,000
  probe, the same `RESOURCE_EXHAUSTED` now surfaces from `analyze_groups`,
  `scheduler_warm_up` and `scheduler_run_eager` instead of being reported as a
  cold trace, and solver routing is absent rather than a wrong `Kvaerno5`. Narrowing the `except` by ordering would have kept
  the discrimination-by-coincidence: in JAX 0.10 every *tracer* error is a
  `TypeError`, and the `RuntimeError` the scheduler wanted was one
  `stiffness.py` raises deliberately.

- [x] **P0.26 — A batched `y0` writing an ASSIGNED path is silently ignored.**
  ✓ External project, 2026-08-29. Four distinct per-member setpoints written
  into a batched `y0` produced an endpoint spread of exactly 0.0 with no
  warning: the path was ASSIGNED, so `composite.py:169` overwrites the column
  from the process parameter on every RHS call. A population study that varies
  an assigned quantity per member therefore returns one repeated trajectory that
  looks like a legitimate null result.
  *Fixed 2026-08-31.* Reproduced first — four members given setpoints
  0.1/0.4/0.7/1.0 all ended at the process's own 0.5, spread exactly 0.0, no
  warning. `Scheduler.run` now refuses, as one more entry in the existing
  `is_batched` blockers so a caller with several batching problems gets one
  message, and it names the offending path. It fires only when the column
  actually *varies* across members; a uniform value is just the default.
  Regressions in `test_multiscale.py::TestBatchedAssignedPaths`.
  **Known hole:** the check reads concrete values, so it is a no-op when `y0`
  is traced under `vmap`/`jit` — it guards the eager path only, the same shape
  of gap as P0.1.

- [ ] **P0.27 — An affine unit yields a garbage multiplier, silently.**
  `conversion_factor` (`units.py:25`) returns
  `parse_expression(from).to(to).magnitude`, which is **f(1)**. That is the
  scale only for a linear (ratio-scale) unit; for an affine one, f(x) = ax + b,
  it returns a + b, which is not a scale at all. Measured:
  `degC -> kelvin` returns **274.15** (so 0 degC maps to 0 K rather than 273.15,
  and 100 degC to 27,415 K); `degF -> degC` returns **-17.22**, a negative
  multiplier that flips the sign of every value. The RHS then applies it per
  port on every call, with no warning — `except Exception: return 1.0` catches
  only unparseable units, not this.
  Latent today because concentrations, rates and amounts are all ratio-scale.
  It fires the moment a model declares a temperature (Arrhenius kinetics,
  thermal stress) or a clinical scale such as HbA1c NGSP% <-> IFCC mmol/mol.
  *Fix, minimum:* detect non-multiplicative units and raise. Linearity is
  testable without library internals — f(2) == 2*f(1) for a linear unit — and
  the same two probes give the real pair, scale `f(2) - f(1)` and offset `f(0)`.
  *Fix, full:* carry `(scale, offset)` per port instead of a scalar. Note the
  offset is **role-dependent**: an EVOLVED port carries a derivative, and
  d/dt(ax + b) = a dx/dt, so the offset must be applied on reads and on
  ASSIGNED/LATCHED/INPUT values but **never** on an EVOLVED write. Applying it
  there is a second silent-wrong.

- [ ] **P0.17 — `atol_scale` freezes the tolerance on a decaying state, and the
  solve returns 10⁵⁷ with `ok=True`.** ✓✓ From an off-attractor IC on GZ06,
  HallSim returns **−1.53e57** where scipy Radau / LSODA / DOP853 at rtol 1e-10
  all return **+9.9584e-6**, at every horizon ≥ 100 with `macro_dt ≥ 100`:

  ```
  t_end ≤  50   HallSim  9.98e-06   scipy 9.9584e-06   agree
  t_end = 100   HallSim -4.749e+03  scipy 9.9584e-06   ok=True
  t_end = 2000  HallSim -1.530e+57  scipy 9.9584e-06   ok=True
  macro_dt 10 or 1 → correct;  macro_dt 100+ → diverged, ok=True
  ```

  Cause: `scheduler.py:1566`, `atol_vec = max(atol, atol_scale·|y₀|)` with
  `DEFAULT_ATOL_SCALE = 1e-6`. At `x(0) = 13.57` the tolerance on x freezes at
  1.357e-5 for the whole solve — larger than the value x decays to, and 14% of
  the distance to that model's pole at `x = −k = −1e-4`. Confounds separated:
  rtol ±6 orders, `newton_atol` ±8 orders, `dt0` and `max_steps` change nothing;
  pinning `Kvaerno5` fixes it; `atol_scale ≤ 1e-7` is correct.
  **The failure is tolerance-insensitive**, so the loose-vs-tight screen calls it
  converged, and `_guard_result` inspects only diffrax's RESULTS code, never the
  values.
  Not currently active: the composite starts from the deposit's own IC, which is
  bounded at 600 / 2000 / 5000 d. It bites a basin scan, a heterogeneous-IC
  population sweep, or an equilibration probe.
  *Fix:* scale `atol` to the state's running magnitude rather than freezing it at
  `|y₀|`, or check the returned values against a bound rather than trusting the
  solver's status code.

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

- [x] **P0.28 — A cold stiffness cache under `jit`/`grad`/`vmap` crashed with a
  numpy message instead of degrading.** Found 2026-08-30 taking a gradient
  through a perturbation sweep — the first time `Scheduler.run` was called
  inside a transform with an unresolved cache. `stiffness.py:280` did
  `np.asarray(composite.evolved_indices(...))`, and under a transform those
  indices arrive traced, so it raised
  `TracerArrayConversionError: The numpy.ndarray conversion method __array__()
  was called on traced array with shape int32[200]` — uncaught, because it is a
  `TypeError` and the scheduler catches `StiffnessNotConcrete`. The degradation
  path that P0.19 established existed but was unreachable: the raise happens one
  site *earlier* than `_restricted_jacobian`, which is where P0.19 put the
  named exception.
  *Fixed 2026-08-30:* that conversion goes through `_concrete`, so it raises
  `StiffnessNotConcrete` and the scheduler degrades as designed. The shared
  message now names the remedy (`call Scheduler.warm_up(y0) once eagerly before
  differentiating`) rather than describing a Jacobian, since it covers both
  sites.
  **What it cost, measured on the same sweep:** degraded (all groups
  `Kvaerno5`) 571.5 ms/arm; after an eager `warm_up` resolving to `Tsit5`,
  **22.5 ms/arm — 19x.** That is the practical price of P0.1's open half, on a
  real workload rather than the demo.

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

- [x] **P1.12 — `screen_process` passes a model sitting 67 384 units/day from
  its own rest state.** *Fixed 2026-08-25.* Fourth failure mode `not_at_rest`,
  reported as a time: `ScreenReport.rest_tau` / `.rest_state` from the new
  public `diagnostics.rest_timescale(composite, y0)`. Flags when the fastest
  state's τ falls below the save interval — that state has relaxed before the
  first sample, so nothing saved is the declared IC. Advisory (does not gate
  `ok`), and a live time-dependent term at t=0 is named in the detail rather
  than counted as disequilibrium, via the new `steady_state.is_autonomous`
  predicate split out of `warn_if_time_dependent`. DallePezze now screens
  `NOT-AT-REST` at **τ = 0.000148 d = 12.8 s on `dp14/Mitophagy`**, matching
  the reviewer's hand-derived 13 s. Added to the intake protocol in
  `CLAUDE.md` beside the other three.
  What it replaces: the screen checked exploding, vanishing and
  tolerance-sensitivity, none of which notice that a declared initial condition
  is nowhere near a steady state. One RHS evaluation would have caught P0.14 at
  import, before any composite was built. Three independent parties — two
  reviewers, an outside calibration agent, and an outside model-building agent
  that chose a 30-day equilibration blind — each hand-rolled this measurement
  because nothing reported it.

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
  and described summaries as co-solved `RunningIntegral`s, which the multi-hallmark demo
  stopped using in favour of post-hoc zero-phase filters.
  **Found while fixing, not fixed:** `demos/multi_hallmark_hybrid.py:492` reads
  `gz06/x2_integral`, a store path the composite no longer has — that demo
  cannot run.

---

## P3 — capability gaps

- [ ] **P3.0 — SBML events that assign to a parameter are silently skipped,
  so a constituent cannot run its own published experiment.** `sbml_events`
  warns `assigns to non-species 'S' (parameter target) — skipped` and continues.
  The model then imports, screens and composes while the experiment it was
  published to reproduce is unreachable, so the intake protocol's
  constituents-first rule cannot actually be satisfied for it. Hit on Yao 2008
  (BIOMD0000000318), whose serum steps `e1`/`e2` both target the parameter `S`
  — the arrest switch Phase 2 of
  [senescence-model-rebuild.md](senescence-model-rebuild.md) depends on.
  Distinct from the general event translator in [roadmap.md](roadmap.md): this
  is one narrow case (parameter-target assignment → LATCHED param promotion)
  and it blocks a live piece of work.
  *Fix:* promote parameter targets to LATCHED and emit the handler; failing
  that, refuse the import rather than warning past it, since a model that
  cannot run its own experiment should not silently reach a composite.

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
  all. Original entry, on `steady_state`: ✓ External project, 2026-08-28: a
  910-node linear signalling composite × 300 clamp conditions was intractable
  under `steady_state` (damped Newton + `jacfwd`) on CPU, and was finished only
  by replacing the inner solve with a hand-written linear solve — validated
  bit-exact against `steady_state` (max diff 0.0) on a 40-node subgraph, then
  vmapped over 300 conditions in ~8 s. Two independent reductions are missing,
  and neither costs the nonlinear general case:
  - **Linear fast path.** For an affine RHS, Newton converges in one step and
    the Jacobian is constant, yet it is re-derived per condition and iterated.
    Detect (or let a Process declare) linearity and go straight to one solve.
  - **Sparse Jacobian.** `jacfwd` materialises a dense Jacobian — *n* forward
    passes then a dense factorisation — for a topology whose coupling graph is
    sparse and already known from the topology dict. Sparsity is derivable from
    wiring, not something the user should have to supply.
  *Rule this is an instance of: generality must never be overhead; it must
  reduce for the trivial case.* Until both land, the framework loses any
  large-but-easy problem to fifteen lines of `jnp.linalg.solve`.
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

- [x] **P3.12 — A port is structurally a scalar store path, so an N-dimensional
  field costs N ports.** *Closed 2026-08-31.* ✓ External project, 2026-08-29. `Port`
  (`process.py:176-215`) has no shape field, so a Process writing a 10,000-gene
  field declares 10,000 ports and `_port_view` (`composite.py:159`) rebuilds them
  as 10,000 traced scalars on every RHS call. Measured RHS jaxpr size grows at
  **6.00 equations per gene** — 1,631 / 2,831 / 7,031 / 19,031 / 61,031 at
  N = 100 / 300 / 1,000 / 3,000 / 10,000 — against **54, flat at every N**, for
  hand-rolled JAX/Diffrax doing identical maths. At N=1,000 the graph is ~96 %
  `slice` + `squeeze` + `mul`. The whole slope comes from one process; the cost
  is trace and compile, not run (at N=3,000 the reverse pass is 244× on compile
  and 5.9× on run, and the run ratio *falls* with N).
  The per-port work carries real semantics — `idx`, and the `rf`/`wf` unit
  conversion factors — but all three are `eqx.field(static=True)` and therefore
  known at build time, so this is a static contract being re-enforced as traced
  graph nodes on every call.
  *Fix — array-valued ports, prototyped 2026-08-29 in a patched copy:* **243
  jaxpr equations at every N from 100 to 10,000**; at N=10,000 the reverse pass
  goes 304.6 s of trace+compile → 1.13 s (269×), `Scheduler.warm_up` 15.6 → 2.8 s,
  and the endpoint is **bit-exact** through a full Diffrax solve
  (`max_abs_diff = 0.0`). Write semantics survive by measurement: a duplicate
  index inside a block still sums, vector∩vector and vector∩scalar overlaps sum,
  and an EXCLUSIVE clash one element deep raises and names the element —
  provided validation iterates `(port, path)` pairs.
  Two constraints on doing it:
  - **One `ontology` ID for a block breaks merge-or-couple.**
    `analyze_composability` would propose merging two unrelated 10,000-element
    blocks annotated with the same SBO term. Either exclude array ports from
    ontology matching or add an `element_ontology`.
  - **The migration is not incremental until P0.25 is fixed**, because a
    half-migrated composite silently mis-orders its splitting rather than
    failing.
  **Closed 2026-08-31.** A port binds a *list* of store paths:
  `topology[proc][port]` is always a tuple, normalised once in
  `Composite.__init__`, and `Port(elements=...)` declares a block gathered and
  scattered as one slice. Measured on the VCC composite, CPU, against the CPU
  baseline: **871 jaxpr equations at N=300, 3,000 and 10,000 alike**, against
  2,831 / 7,031 / 61,031 — slope 6.00 -> 0.00. At N=1,000 the gradient path is
  **85.2 s -> 8.95 s (9.5x)**, RHS trace 11.8x, grad trace 11.3x. Block and
  scalar spellings agree to exactly 0.0 on store order, initial state and RHS
  output. The LLVM compile wall at N=10,000 is structurally gone. Guarded by
  `test_block_port_rhs_is_flat_in_width` and — because the first scatter
  rewrite silently cost the *scalar* path a broadcast per port —
  `test_scalar_port_cost_per_port_does_not_regress`.
  *Superseded detail — partially addressed 2026-08-29, the multiply half.* Port maps are now
  `(ports, indices, factors)` and `_port_view` does one gather plus one
  elementwise multiply per *process*; the write side stacks once before one
  vector multiply. Framework multiplies went from 2N to **2, independent of N**.
  Re-measured on the same probe: **slope 6.00 -> 4.00 eqns/gene** (1,631->1,342,
  7,031->4,942, 61,031->40,942 at N = 100 / 1,000 / 10,000 — exactly 2N at each).
  At N=1,000 the full gradient path is **107 s -> 70 s** (grad trace 73.71->39.82,
  batch_grad_compile 66.46->51.57). **Run time is unchanged** — XLA already folded
  the identity multiplies — so this is trace/compile only.
  The residual per-gene framework cost is `slice` + `squeeze` — the
  dict-of-scalars interface itself — which only array ports remove. The factor
  arrays this builds are what an array port consumes, so it is a step in, not
  work to unwind.
  **Priority note:** array ports buy throughput but do not change what gets
  allocated, so they no longer head the queue — P3.10's dense Jacobian is what
  blocks N=10,000 outright.
  Refuted alternative: keeping scalar ports and grouping contiguous index runs
  inside `_port_view` measures 9 equations *worse* than the free fix of eliding
  the identity unit multiply, with an identical slope and `slice` unchanged at
  2,203 — it cannot work while `derivative` receives `dict[str, scalar]`.
