# Fixed problems

Defects from `known-problems.md` that are closed. They live here and not there
because that file is the worklist — what is still wrong — while the reviews and
the diary still cite these ids, and a citation has to resolve. Code cites no id
at all: a fixed line states its reason in words.

Moved 2026-09-07. Newest last, in the order they were filed.

- [x] **P0.23 — `coupling_mode="interpolated"` interpolates only the
  *immediately preceding* group; every earlier group stays frozen while the mode
  reports as interpolated.**
  Measured 2026-08-31: bit-identical to `frozen` at `macro_dt` 3.5 / 1.75 /
  0.875, five significant figures, all four observables.
  `_effective_coupling` passes an explicit mode straight through and
  `_run_scan_continuous` sets `interp = coupling == "interpolated" and
  n_groups > 1`, which was true — so it was requested and enabled, and produced
  no difference.

  *Mechanism traced 2026-08-31 (external systems review), and it is one line.*
  `scheduler.py:1253-1262`, inside `_run_scan_continuous`'s `body`, builds
  `_InterpFill(..., idx=write_idxs[gi - 1])` and reassigns `prev = (t_start,
  t_next, gy)` every iteration. So `_InterpFill` (`scheduler.py:248-261`) only
  ever carries group `gi-1`'s trajectory and only re-fills group `gi-1`'s
  columns; groups `0 .. gi-2` are supplied by `_FrozenFill`'s constant. The
  eager path has the identical defect — `prev_idxs = group_write_idxs[gname]`
  is reassigned per group at `scheduler.py:810`.

  Three continuous processes; A a 6 rad/s oscillator, C integrates A, B
  independent. Same composite, same `macro_dt=2.0`, only the *grouping* differs;
  reference is a `macro_dt=10/2048` solve, y(10) = -0.0576600435:

  ```
  grouping                                    frozen y(10)     interp y(10)     identical?  interp err
  {gA:[drv], gC:[dvn]}       edge adjacent    -1.722649856820  -0.051501993040  no          10.68 %
  {gA:[drv], gB:[mid], gC:[dvn]} non-adjacent -1.722649856820  -1.722649856820  bit-identical 2887.60 %
  ```

  Inserting one *unrelated* group between the driver and the driven turns
  interpolated coupling into a bit-exact no-op and multiplies the error by 270×.
  That reproduces this entry's original signature exactly, and it is consistent
  with P0.6 having measured interpolated working (1.7% vs 20.9%) on a
  configuration where the driving edge happened to be adjacent.

  **A design defect, not just a bug.** `auto_groups` clusters by timescale and
  `_order_by_coupling` topologically sorts, but nothing ties *adjacency in the
  group order* to *where the coupling edges are*. The coupling representation
  was written as if the group sequence were a chain; it is a DAG. A topologically
  valid ordering can place any number of groups between a driver and its
  consumer, so adding a fourth model to a working three-model composite can
  silently switch a previously-interpolated edge to frozen, with no diagnostic.
  *Fix:* (1) keep **one interpolant per group**, not one for `gi-1` — accumulate
  `prev` into a list and build the fill from every already-solved group's samples
  this window; the samples are already computed and already a static shape
  (`n_save`), so this costs memory, not compile shape (~1 day including the eager
  path). (2) Make `_effective_coupling`'s verdict **per edge**, not per run
  (`scheduler.py:1074` decides for the whole run from the existence of *any*
  forward cross-group edge), and put the resolved per-edge mode into
  `SchedulerResult.stats` so "interpolated" is an observable fact rather than a
  requested flag. (3) Regression test: insert an inert group between a driver and
  its consumer and assert the interpolated result is unchanged. Today it fails.
  **Bears on P0.2.** That entry calls the Strang/Lie order study "unseparated"
  between NF-κB's oscillation breaking Strang and the P0.20 span-truncation fix.
  There is a third candidate it does not list — whether the NF-κB edge was
  adjacent in the group order in each configuration. Separate it before designing
  around either.
  **Side effect worth knowing.** `coupling_mode="interpolated"` also silently
  changes the output grid: `n_save = max(base_out + 1,
  self.coupling_interp_points)` (`scheduler.py:1170-1172`), so an interpolated
  run returns `coupling_interp_points` samples per macro window regardless of
  `save_dt`. In the measurement above, `macro_dt=save_dt=2.0` gave 6 points
  frozen and 76 interpolated over the same span. A coupling knob should not
  change the shape of the answer.

  **Fixed 2026-09-05.** Both paths now accumulate every group solved in the
  window instead of carrying only the previous one. The groups share the window
  and its save grid, so their dense outputs concatenate into a single
  interpolant — one gather, no per-group loop in the fill. Sites:
  `_run_scan_continuous`'s body (scan) and the eager Lie loop.

  Measured on the P0.23 shape — driver, inert group, consumer, so the driving
  edge spans two positions:

  ```
    grouping  macro_dt         frozen         interp   identical?
    adjacent      2.00   -0.143982589   -0.005120877        False
      spaced      2.00   -0.143982589   -0.005120877        False   (was True)
  ```

  Against a converged reference of -0.005296, the spaced case goes from
  **2618% error to 3.3%**. `spaced` and `adjacent` now agree, which is the
  invariant: inserting an unrelated group must not change the answer.
  Regression tests: `test_interpolated_coupling_survives_an_inert_group_between`
  (both paths) and `test_interpolated_beats_frozen_on_a_non_adjacent_edge`.

  *What interpolation is worth, now that it works* — observed convergence order
  on a two-group split, error vs a converged reference:

  ```
                scheme       2.0       1.0       0.5      0.25     0.125
   feed-forward, lie/frozen   7.2387%   0.6084%   3.6330%   1.9266%   0.9860%
   feed-forward, lie/interp   0.0149%   0.0010%   0.0003%   0.0004%   0.0007%
   feedback,     lie/interp   1.5223%   1.5186%   2.7933%   1.9326%   1.0210%
  ```

  On a **DAG** interpolation does not reduce the splitting error, it removes it:
  the error floors at ~1e-6 relative, which is solver tolerance, not splitting.
  On a **cycle** it reverts to first order (p ~ 0.9) because the backward edge
  still reads a stale value — that residual is what waveform relaxation (P0.47)
  exists to remove, and it is now measured rather than asserted.

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

- [x] **P0.41 — EVENT and DISCRETE processes fired in dict-insertion order, so
  a jit/vmap round-trip changed which one fired first.** *Fixed 2026-09-04, in
  the commit that filed it.* Found by asking whether the round-off dependence
  in Stucki 2005 was our defect. It was not — COPASI reproduces that on the
  same file — but the question surfaced this, which is ours.

  `Composite.event_processes()` and `discrete_processes()` preserved
  `self.processes` insertion order. The Scheduler iterates them and applies
  each delta **immediately** (`scheduler.py:1106-1131`), so a later process
  reads the state an earlier one wrote — order is semantic whenever two of
  them touch the same path.

  And the order was not stable. `store.py:120` already documents the hazard for
  `build_initial_store`: *JAX sorts dict keys when it flattens a pytree, so
  `processes` comes back sorted from any `jax.jit` / `vmap` / `eqx.tree_at`
  round-trip.* Measured on a two-event composite inserted as
  `["z_evt", "a_evt"]`: before a round-trip the firing order is
  `['z_evt', 'a_evt']`, after `jax.tree_util` flatten/unflatten **and** after
  `eqx.tree_at` it is `['a_evt', 'z_evt']`. So a solo run and a jitted or
  batched run of the same composite could fire interacting events in opposite
  orders and reach different states — the same class of defect as
  `test_batched_matches_solo` under P0.1.

  *Fixed at the container, not the consumers.* `Composite.__init__` now stores
  `processes` name-sorted, so every consumer inherits stable order and the
  hazard is unreachable rather than guarded.

  **That is the actual lesson, and it is why this recurred.** The same defect
  was found and fixed in `build_initial_store` (P0.15-era), where the remedy
  was a local sort at that one call site. `Composite` has **nine** places that
  iterate `self.processes`; before this fix two sorted and seven did not, so
  every new consumer was a fresh chance to reintroduce it — and EVENT dispatch
  duly did. An invariant enforced per call site is not enforced. The first
  patch attempted here name-sorted the two accessors, which would have been
  the same mistake a third time; sorting at construction is what makes it
  structural. Sorted order is also exactly what survives the round-trip, so
  solo and round-tripped runs agree by construction.

  **What this does not fix, and must not be confused with it:** when two
  events are simultaneously satisfiable, *some* order still decides the
  outcome, and the model has not specified one. Sorting makes our answer
  reproducible; it does not make it right. A model in that position is
  rejected at intake by P0.38.

- [x] **P0.42 — The inert-sink heuristic freezes a model's only output.**
  *Fixed 2026-09-05, wiring Kallenberger into the multi-hallmark composite.*
  Three parts: `SBMLProcess.with_unfrozen(*species)` is the explicit opt-out
  (`_frozen_indices` is static, so this rebuilds rather than `tree_at`);
  `Composite` calls `_unfreeze_coupled_sinks` after event expansion, which
  restores any frozen sink another process *reads* — being wired to a reader is
  unambiguous evidence the heuristic misfired, so no call site has to know; and
  the import warning now says the frozen species are unusable as coupling
  sources or reporter observables and names the opt-out, instead of implying a
  tidy-up. Verified on Kallenberger: `tBid` integrates 0 → 186.98 while `Bid`
  falls 224 → 37.02 over the deposit's own 240-minute window, conserving mass
  exactly. Filed 2026-09-04.
  `_frozen_sink_indices` (`sbml_import.py:1218`) freezes
  species that reactions write and nothing reads, so they cannot accumulate and
  wreck the state scaling. The heuristic is self-defeating for exactly the case
  the framework exists to serve: a **source process whose output is a terminal
  product** has, by construction, a downstream-facing species that nothing
  *inside that model* reads.

  Measured on Kallenberger 2014 (BIOMD0000000524): **six** states are frozen
  to zero — `tBid`, `mCherry`, `mGFP`, `PrNES`, `PrER`, `p18inactive` — of
  which `tBid` is the apoptosis-commitment readout and the only thing a
  downstream process would sensibly consume, and four of the others are the
  paper's measured fluorescent reporters. `Bid` falls 224 → 37.02 over 240 min
  while `tBid` stays at exactly 0.0. The mass is recoverable by balance and
  the freeze does warn, but a user composing this to read `tBid` gets a flat
  zero and a log line.

  **There is no opt-out**: `_frozen_indices` is a static field, so `eqx.tree_at`
  cannot reach it and a caller cannot un-freeze a species it means to export.

  *Fix:* the freeze needs an opt-out — a species named as a composition output
  must not be frozen — and the warning should say which species are being
  frozen *and* that this makes them unusable as coupling sources. Better: infer
  from the topology, since a frozen sink wired to another process's INPUT port
  is unambiguous evidence the heuristic is wrong for that model.

- [x] **P0.44 — `native_time_seconds` cannot be set at import, so correcting a
  guessed clock needs `eqx.tree_at` on a private field.**
  *Fixed 2026-09-05.* `process_from_sbml(..., native_time_seconds=60.0)` is the
  front door. The boolean `native_time_declared` is replaced by a three-valued
  `native_time_source` — `"declared"` (the file asserts it), `"supplied"` (the
  caller knows what the file omits), `"assumed"` (the tool default) — which is
  the distinction P0.40 asks for, so a supplied clock is not laundered into a
  declared one. A supplied value that contradicts a declared one warns. Verified
  on Kallenberger: `reconciled_to(86400)` returns 1440, where it returned 86400.
  Filed 2026-09-04.
  `process_from_sbml` takes `timescale` and `parameters` but has no argument
  for the model's native clock, so when a file declares no time unit — which
  is most of them — the only way to supply the right value is to reach into
  the process afterwards. Kallenberger 2014 declares zero `unitDefinition`
  elements; its rate laws are in minutes, so `reconciled_to(86400)` returns
  `time_scale = 86400` where the correct value is **1440**.

  CLAUDE.md records that a wrong clock has cost this project three times.
  A failure with that history should be correctable at the front door.

  *Fix:* `process_from_sbml(..., native_time_seconds=60.0)`, which also gives
  the natural place to record that the value was supplied rather than
  declared — the distinction P0.40 asks for.

- [x] **P0.47 — Timescale splitting cuts feedback loops, silently, with no
  diagnostic and no way to make the split accurate.** Filed 2026-09-05, found
  wiring Kallenberger in.

  DP14 (86400 s) and GZ06 (3600 s) are within `max_ratio`, so the composite was
  **one group** and the Scheduler took the single-solve fast path — the
  `damage_bridge -> dp14 -> gz06 -> p53_cdkn1a` feedback loop was integrated
  exactly. Adding Kallenberger (60 s) pushed the ratio to 1440 and split the
  composite straight through that loop.

  A cycle has no topological order, so whichever group runs first reads the
  other's previous-step value. `_order_by_coupling` **already detected this** —
  `if not ready: # cycle: keep the remaining timescale order` — and proceeded.
  Measured against the exact single-group solve:

  | config | macro_dt 3.5 | 1.75 | 1.0 | 0.5 |
  |---|---|---|---|---|
  | Lie, fast group first (current) | **10.00%** | 3.07% | 1.96% | 1.30% |
  | Lie, slow group first | 14.03% | 2.85% | 1.58% | **0.77%** |
  | Strang | 72.0% | — | 29.2% | 8.8% |
  | Lie + `adaptive_dt` | 65.2% | — | 38.6% | — |

  CDKN1A@14 is the casualty at the demo's own step: 23.44 against 26.03
  converged — the same reporter P0.2 names, for the same reason.

  **Nothing available today fixes it.** Group order changes it by <2x and
  reverses sign with `macro_dt`. Strang is 7x worse (see P0.2). `adaptive_dt`
  is 6-20x worse (P0.49). `coupling_mode="interpolated"` is bit-identical to
  `frozen` — **and that reading was wrong on both counts, corrected
  2026-09-05.**

  It was P0.23 after all. The bit-identical result came from a *three*-group
  composite whose driving edge was non-adjacent, so the interpolant (which
  carried only the previous group) never reached it. Cyclicity was a
  coincidence of that composite, not the cause. With P0.23 fixed, interpolated
  and frozen differ on a cycle and interpolation is worth roughly half the
  error — 11x at the coarsest step:

  ```
    macro_dt   frozen err   interp err
       2.000       7.127%       0.651%
       0.500       3.811%       1.911%
  ```

  The reasoning was wrong in principle too: a cycle only denies an interpolant
  to the group that runs **first**. Every later group has the earlier groups'
  freshly-computed trajectories available over the window. A cycle disables
  half the interpolation, not all of it.

  **Strang is not 7x worse either** — it was measured outside its asymptotic
  regime. Observed order from successive halvings over `dt` 0.5 -> 0.25 ->
  0.125 is **p = 2.33 then 2.07**, second order as designed. It is useless
  above `dt ~ 1.0` (8.9%, 8.0%), then falls to 0.199% at 0.5; at `dt = 0.125`
  it is **0.0094% against Lie's 0.986%, 105x better**. The driver's period is
  ~1.05, so at `dt >= 1` no splitting scheme is asymptotic — and the table
  above samples 3.5/1.0/0.5, almost entirely that band. P0.2 needs the same
  correction.

  So "nothing available today fixes it" is false. What remains true is that the
  cycle's backward edge holds the scheme at first order (p ~ 0.9 measured), and
  that is what waveform relaxation removes.

  *Not fixed by merging the groups.* That was the first attempt and it is an
  avoidance: multi-rate splitting is the capability being demonstrated, and a
  composite whose biology is feedback-coupled is exactly the case it has to
  serve. Backed out; `auto_groups` keeps the split and now warns, and
  `Composite.cyclic_group_sets()` reports which group sets a cycle runs
  through so the condition is queryable rather than folklore.

  **Fixed 2026-09-05: `Scheduler(waveform_sweeps=k)`.** Suggestion #4 built.
  Each macro step re-solves the window k times; within a sweep, group `gi`
  reads groups before it from *this* sweep and groups after it from the *last*
  one. That second half is what a one-pass Lie split never has, and it is
  exactly the backward edge of the loop.

  Measured on a two-group cycle, error vs a converged reference, and the
  observed order from successive halvings:

  ```
              scheme       2.0       1.0       0.5      0.25     0.125
  lie / interpolated   1.5223%   1.5186%   2.7933%   1.9326%   1.0210%
    observed order p      0.00     -0.88      0.53      0.92
         waveform x2   0.0556%   0.0215%   0.0057%   0.0011%   0.0003%
    observed order p      1.37      1.93      2.41      2.03
  ```

  **Error drops 27x at `macro_dt` 2.0 and 3400x at 0.125**; eager path 17x / 71x.

  *What the sweeps do and do not buy — corrected after checking convergence
  properly.* The iteration **does** reach a fixed point: at `macro_dt` 0.25 the
  sweep-to-sweep change is exactly 0 by k=8, and k=2 is already within 2% of
  the converged answer.

  ```
    sweeps        error   change vs prev sweep
         1    1.932600%
         2    0.001063%             1.9315364%
         4    0.001041%             0.0000226%
         8    0.001041%             0.0000000%
  ```

  But the fixed point is **not** the exact solution — it sits at 0.001041%
  against a ~1e-4% solver floor. The residual is the *interpolant's* sample
  count, not the splitting: at the converged k=8, varying
  `coupling_interp_points` moves it directly, while more sweeps do not.

  ```
   interp_pts        error (k=8, macro_dt 0.25)
            8    0.006022%
           16    0.001041%   <- default
           64    0.000439%
          128    0.000384%
  ```

  So the two knobs are separable and both are needed: **sweeps remove the
  stale-edge error, interpolation points set the floor they converge to.**
  Reporting this as "second order" was a mis-read — the observed p ~ 2 is the
  floor's own scaling, not a convergence rate of the iteration.

  *Not verified:* at `macro_dt` 1.0 the sweep-to-sweep change **grows**
  (0.0013% -> 0.0026% -> 0.0056% over k=4/8/16) rather than contracting, so the
  iteration does not cleanly converge at coarse steps. Whether that is
  contraction failure or interpolation noise is untested.

  On a **feed-forward** composite it is bit-identical to one pass, correctly:
  with no cycle there is no stale edge to converge, so the extra passes cost
  k x and buy nothing. Default stays `waveform_sweeps=1`; use
  `Composite.cyclic_group_sets()` to decide where it is worth paying.

  A fixed sweep count rather than the suggestion's `until residual < epsilon`:
  a data-dependent `while` inside `lax.scan` is not reverse-differentiable, and
  end-to-end differentiability is a hard invariant. `coupling_mode="frozen"`
  with `sweeps>1` **raises** — a frozen fill is the same constant every sweep,
  so it would cost k x and change nothing.

  *Fix:* **waveform relaxation — already designed, as suggestion #4 in
  [crossgen-suggestions.md](crossgen-suggestions.md)** (~40 lines wrapping the
  existing Lie loop, Anderson acceleration ~30 more). This entry is what that
  suggestion was waiting for: a measured trigger and cost. The condition is
  narrower than "one-pass Lie loses cross-coupling" — splitting is fine on an
  acyclic group DAG, and only a **cycle spanning groups** forces a permanently
  stale edge, so `cyclic_group_sets()` says where to spend the iteration and
  where to skip it.

  Two constraints recorded there and not in the original pseudocode: a
  residual-tested trip count needs `lax.while_loop`, which is **not
  reverse-differentiable**, so it would foreclose Calibrator unless #6 (IFT at
  sync boundaries) lands with it; a *static* sweep count unrolls, differentiates
  and keeps one compiled executable. Land with P0.23 — same two functions.

  Until then, size `macro_dt` from the table above whenever
  `cyclic_group_sets()` is non-empty.

- [x] **P0.58 — FIXED 2026-09-06. `build_initial_store` compared port initial
  values with exact equality, so a 1e-26 difference blocks the import outright.**
  Filed 2026-09-06. **Stop rule fired**: the triage reported EXPLODING and
  `sbmltoodejax` integrated the same model bounded.

  Dwivedi2014 (BIOMD0000000534-537) is a curated IL-6 QSP model — 41 species,
  71 reactions, 51 parameters, 100% ontology coverage — and the only deposit
  found in a 119-model screen that emits IL6 kinetically. All four arms
  **reject**. The reason is not numerical:

      ValueError: dwi/mwf345ed7a_... is claimed by 5 ports of the same role
      with differing initial values [0.0, 1e-26]

  Four SBML `<event>`s assign the same species; each expanded event port
  declares an initial value, and `store.py:159` rejects on
  `np.all(a == b)` (`_same_default`, `store.py:183`). The values differ by
  **1e-26** against a trajectory whose `max|y|` is 234 — 1e-28 relative. That
  is a CellDesigner export artefact, not "a modelling decision", which is what
  the error message calls it.

  Declaring the value by hand makes all of them run, bounded and finite:

  | model | with `Composite(initial={...: 0.0})` |
  |---|---|
  | BIOMD0000000534 | RAN, finite, `max|y|` = 234.6 |
  | BIOMD0000000537 | RAN, finite, `max|y|` = 703.8 |
  | BIOMD0000000873 | RAN, finite, `max|y|` = 1.4e5 |

  *Two candidate fixes, and the second is the one:*
  (a) compare within a tolerance — a patch, and the tolerance is unit-dependent
  (1e-26 M is a legitimate concentration);
  (b) **an event-assignment port should not declare an initial value at all.**
  An event writes its target when it fires; it does not own where the target
  starts. The owning species does. Under (b) there is no disagreement to
  tie-break and no tolerance to choose.

  Note `_same_default`'s docstring says it "drives a warning and nothing else",
  which the raising call site contradicts.

  **Fixed** by (b): `SBMLEvent.ports_schema` gives a species target
  `default=None` — the abstain the `Port` docstring already documents for
  exactly this case — and keeps the published value for a parameter target,
  which has no other owner. All four Dwivedi arms now **PASS** with
  `rest_residual` ~1e-15. Covered by `tests/unit/test_sbml_events.py`, which
  did not exist: `sbml_events` had no tests at all.

- [x] **P0.59 — FIXED 2026-09-06. The numerical screen reported a construction
  failure as EXPLODING with `max|y| = inf`.** Filed 2026-09-06, found alongside P0.58.

  Dwivedi2014 never reached a solver: `Composite.initial_state_vec` raised, so
  no trajectory exists. The screen recorded
  `EXPLODING + TOLERANCE-SENSITIVE max|y|=inf tol-rel-diff=inf` and rejected on
  it. A model that never ran cannot have exploded, and "exploding" sends the
  reader to solver tolerances instead of to the one-line construction error.

  The FRAMEWORK-SUSPECT annotation did fire and did say `sbmltoodejax
  integrates it bounded` — that annotation is the only reason this was caught,
  and it is doing more work than the verdict it qualifies.

  **Fixed:** `ScreenReport.did_not_construct`, set by building the composite
  in its own guarded step before the tight/loose solves, carrying the
  construction exception instead of sentinel infinities. Same family as the
  three `screen_produced_species` defects fixed the same day: a confident
  verdict about something that was never examined.

- [x] **P0.64 — FIXED 2026-09-06. A combinatorial propensity used as an ODE
  rate law goes negative, and it is machine-checkable.** Third occurrence.

  A rate law of the form `k*x*(x-1)*0.5` is a **Gillespie propensity**: the
  number of distinct pairs among `x` molecules. Integrated as an ODE it is
  negative for `0 < x < 1`, and small pools sit there.

  | deposit | rate law | consequence |
  |---|---|---|
  | Hui 2016 | `kdimerAlk5 * Alk5 * (Alk5 - 1) * 0.5` | mean-field wrong by `1/Alk5`, 3.3% at its own 30.5 molecules |
  | Proctor 2013 | `kdimercJun * cJun_P * (cJun_P - 1) * 0.5` | `cJun_dimer` reaches **-3.302e-4**, invariant to the sixth significant figure across rtol 1e-3…1e-10 — structural, not numerical. Persists 14 days and inverts seven transcription rate laws that read it linearly. |

  Both papers say why: they wanted stochastic *and* deterministic runs from one
  file. So the deposit is faithful and the defect belongs to the modelling
  choice — which is exactly what P0.57's stochastic-intent check is about,
  reached from the rate laws instead of the units.

  **Fixed:** `intake.combinatorial_propensities` scans the kinetic laws for
  an `x*(x-1)` factor where `x` is a species, and `triage_process` reports it
  with the mean-field correction. One pass over the MathML, no solve.
  Measured: Hui 2016 `Alk5Dimerisation` 1 hit, Proctor 2013 `cJunDimerisation`
  1 hit, Dwivedi 2014 0 hits. Reports rather than blocks — a deposit written
  for Gillespie is a legitimate object, and P0.57 is the check for whether it
  should be an ODE at all. Covered by `tests/unit/test_propensity_scan.py`.

- [x] **P0.66 — FIXED 2026-09-07 (`<log/>` with a base rewrites to `ln/ln` in `_preprocess_sbml`; Konrath 2020 imports). `log10()` in a rate law fails the import outright.** Filed
  2026-09-06. `MODEL2004300002` (Konrath 2020, p53/ATM/Wip1, 7 species)
  rejects with `calls function(s) outside sbmltoodejax's mathFuncs table:
  log10()`. `log10(x)` is `log(x)/log(10)`; the pre-processing that already
  flattens function definitions can rewrite it. Same class as P0.63 — a
  curated-quality deposit lost to a one-line translation gap.

- [x] **P0.67 — FIXED 2026-09-07. A batched `y0` was refused whenever the
  composite held a DISCRETE process.** The eager loop gathered a process's
  port view with `state[i]` and scattered its delta with `state.at[idxs]`,
  both leading-axis on a `(batch, n_vars)` state, so the guard refused the
  batch outright. Both now index the trailing axis; a delta entry is a scalar
  (every member) or `(batch,)` (one per member). The regression exposed a
  second one on the continuous side: `_FlatRHS` stacked a state-independent
  derivative (a constant source returns a scalar) next to a per-member one and
  failed on the stack, so any batched run with a constant source was already
  broken. Every piece is now broadcast to the batch shape before the single
  scatter, a no-op unbatched. EVENT processes were refused for the same
  reason one layer up: `condition` was reduced through Python `bool`, one
  verdict for the whole batch. The edge is now computed per member
  (`cond & ~was_active` as arrays), the handler's delta is masked to the
  members that fired, and `EventRecord.members` carries the mask.
  Regressions: `TestSchedulerBatchedGuards::test_batched_y0_with_discrete_matches_solo`
  and `::test_batched_y0_with_event_matches_solo`.

- [x] **P0.69 — FIXED 2026-09-07. Any `<root/>` in a rule failed the import: sbmltoodejax's function table maps `sqrt` to the misspelt `no.sqrt`, so the generated code raised `NameError` at the first assignment rule. `_preprocess_sbml` now rewrites roots as powers (and logs with a base as `ln/ln`), and the cache is versioned so old translations are redone. Erguler 2013 imports (38 species, 87 parameters, FLAG: not at rest) and its Goldbeter–Koshland `piecewise` evaluates to the hand computation at three PERK levels.** Originally filed as a piecewise failure:
  Filed 2026-09-07. BIOMD0000000446 (Erguler 2013, unfolded protein response,
  27 species) is the only kinetic ER-stress deposit in any repository and its
  readouts move in the benchmark (DDIT3 +0.23/+0.54, ATF3 +0.37/+0.74). Its
  eIF2α rule uses a Goldbeter–Koshland function written as a piecewise that
  sbmltoodejax cannot translate. Same class as P0.63/P0.66: a translation
  gap, not a modelling one.

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

- [x] **P0.17 — FIXED 2026-09-06. `atol_scale` froze the tolerance on a
  decaying state, and the solve returned 10⁵⁷ with `ok=True`.** ✓✓ From an off-attractor IC on GZ06,
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
  **The "not currently active" reading was wrong.** Re-found 2026-09-06 from the
  other end — a Dwivedi 2014 review measured `jax.jacfwd` through
  `Scheduler.run` as 0.40% off central differences **from the deposit's own
  IC**, invariant to `rtol` 1e-6…1e-12 and `newton_atol` 1e-6…1e-14. That is
  this entry: not a broken AD path but a correct gradient of a trajectory
  integrated to the wrong tolerance. On a 2-state stiff probe
  (`scratch/2026-09-06-gradient/`) the default cost **4.5% on the value and
  2.6% on the gradient**, resolving a state that had decayed to 1.66e-6
  against an `atol` frozen at 1e-6 — 60% of the answer. It reaches every
  calibration through a stiff group, which is most of them.

  **Fixed by removing the state scaling outright**, not by lowering the
  constant. `rtol` already scales the error allowance by the *current* state,
  which is what `atol_scale·|y₀|` was doing with stale data; `atol` is now a
  true floor near zero, shared by every group. `DEFAULT_ATOL_SCALE`, the
  `atol_scale` argument and `Scheduler._scaled_tolerances` are gone.

  Measured cost of the removal: DP14 34.2 → 38.8 ms, GZ06 26.2 → 27.5 ms, both
  trajectories moving ~1e-5 relative. The DP14 13.3 s → 1.7 s speedup this
  scaling was credited with came from **stiffness routing** (Kvaerno5 over
  Tsit5), which is untouched. After: value error 4.5% → 0.019%, gradient
  2.6% → 0.0054%. Regression test in
  `tests/unit/test_stiffness_routing.py::TestToleranceIsNotStateDerived`.

  *Still open from this entry:* `_guard_result` inspects only diffrax's RESULTS
  code, never the values, so a diverged solve can still report `ok=True`.

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

- [x] **P0.51 — the stop rule fired a second time, on the *continuous* path:
  a parameter sweep re-resolved the solver on every arm.** Measured and
  **fixed 2026-09-05**. Distinct from P0.35, which is the event machinery.

  Hand-rolled `dfx.diffeqsolve` beat `Scheduler.run` by **4.1x and 5.9x** per
  warm arm on BIOMD703 and BIOMD318, with **92 and 166 XLA compiles against 2**,
  at identical solver step counts (37/38.5 and 155/155) and endpoints agreeing
  to 1.2e-7 and 1.6e-9. Same maths, same work.

  **Cause, measured by neutralising each candidate** (stiff 8-state composite,
  6 arms, `HALLSIM_COMPILATION_CACHE_DIR=off`):

  ```
  hand-rolled (one filter_jit)   0.0016 s/arm    7 compiles first / 0 warm
  Scheduler, before              0.0327 s/arm   78 compiles first / 0 warm
  Scheduler, digest neutralised  0.0038 s/arm    1 compile  first / 0 warm
  ```

  The P0.29 parameter digest is **~90%** of it. Keying the stiffness verdict on
  concrete parameter values is correct — the verdict *is* a function of them —
  but a sweep changes a value every arm by construction, so every arm missed
  and re-resolved. The remaining **2.4x** is eager orchestration around the
  compiled core.

  **Calibration was never affected**, contrary to the obvious worry: under
  `jax.grad` the parameters are tracers, the digest abstains, and the warm-up
  verdict is reused. Verified by counting `analyze_groups` calls — **6 optimiser
  steps, 0 analyses**, against 6 analyses for 6 eager arms. (The first attempt
  at that instrument patched `hallsim.stiffness.analyze_groups`, which the
  scheduler binds at import, so it counted zero everywhere and read as "nothing
  happened". Patch `hallsim.scheduler.analyze_groups`.)

  *Fixed:* `run(plan, params_from=composite)` substitutes parameter values into
  a plan's existing resolution, guarded by `structural_fingerprint()` so only
  values may differ. **0.0327 -> 0.0038 s/arm, 8.6x, compiles 78 -> 1** —
  identical to ignoring parameter values, except the caller now *asserts* the
  verdict holds instead of a cache silently assuming it.
  `Scheduler.verify_plan(plan, composite)` measures that assertion and returns
  the groups whose verdict moved; run it against fitted parameters at the end of
  a fit. That closes review open question 3.
  Bit-exactness: 28 reference arrays, **0.000e+00** against HEAD without the
  change.
  **Still open:** the 2.4x orchestration residual, now the whole remaining gap.

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

- [x] **P3.0 — SBML events that assign to a parameter are silently skipped,
  so a constituent cannot run its own published experiment.** *Fixed
  2026-09-04.* `translate_events` keeps a parameter target and records it in
  `_param_targets`; `expand_events` promotes it on the owning process through
  the existing `ImportedODEProcess.with_param_input`, so the assignment
  reaches the rate laws through a store path. The event gets an INPUT read
  port for the target as well, because the handler applies an assignment as a
  delta and needs the current value, and the LATCHED write port starts at the
  parameter's published value rather than zero.
  `expand_events` now returns the promoted owner alongside the event
  processes, and its topology row carries only the promoted-parameter
  entries for the caller to merge.
  Two further defects surfaced on the same path and are fixed with it:
  **a zero delay was read as a delay** (COPASI writes `<delay>0</delay>` on
  every event it exports, so every COPASI model with events was refused for a
  delay it does not have — only a nonzero delay raises now, and a
  state-dependent one still does), and **rule-defined ModelValues would not
  resolve** (COPASI exports a constant as a non-constant parameter plus an
  assignment rule, e.g. `DNAdamagefoci_0 = Gy * FociPerGy`, which was absent
  from the constant table; `fold_constant_rules` folds those to a fixpoint and
  leaves genuinely dynamic rules alone).
  Verified on both models this blocked. Yao 2008 (BIOMD0000000318): `e1`/`e2`
  translate, targets `['S']`. Kollarovic 2016 (BIOMD0000000632): imports
  `[PASS]` with ‖f(y₀)‖/‖y₀‖ = 1.3e-16, and the dose now lands — at 0/5/20 Gy
  `TAF` goes 0.506/2.684/4.861, p21 1.00/3.66/10.31 and CycE-Cdk2 activity
  2.28/0.008/0.00006.

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
  **Priority note (written 2026-08-29, superseded):** array ports buy throughput
  but do not change what gets allocated, so they no longer head the queue —
  P3.10's dense Jacobian is what blocks N=10,000 outright. That held until the
  matrix-free stiffness path landed on 2026-08-31; see P3.10.
  Refuted alternative: keeping scalar ports and grouping contiguous index runs
  inside `_port_view` measures 9 equations *worse* than the free fix of eliding
  the identity unit multiply, with an identical slope and `slice` unchanged at
  2,203 — it cannot work while `derivative` receives `dict[str, scalar]`.

- [x] **P0.31 — `derivative()` or `assign()` returning an undeclared port is
  silently dropped.** Found 2026-08-31 (external systems review).
  `_FlatRHS.__call__` (`composite.py:325-331`) iterates `write_map.ports` and
  does `if port not in raw: continue`; nothing checks the converse.
  `assign()` has the same shape at `composite.py:262-264`. Measured — a process
  declaring only `x` and returning `{"x": -s["x"], "typo_port": 99.0}` gives
  `rhs(0, y0) == [-1.]`, the 99.0 contribution gone with no warning. A renamed
  or mistyped port name is the single most likely authoring error and the one a
  generated `Process` will make.
  **Free to fix.** `_FlatRHS.__call__` runs in Python at *trace* time, so a set
  comparison there never enters the jaxpr and costs nothing at runtime:
  `extra = raw.keys() - set(write_map.ports)` → raise naming the undeclared
  ports and the declared set. ~10 lines across `derivative` and `assign`, 1 hour.
  Same rule as P0.4 (`dose_window=None` silently deleting a hallmark dial):
  **an operation that resolves to nothing must say so.** Worth fixing as one
  rule rather than two instances.
  *Related but not a raise:* omitting a *declared* EVOLVED port from
  `derivative()` silently freezes that state (a process declaring `x` and `z`
  but returning only `x` gives `[-1., 0.]`). That is legitimately allowed — a
  process may contribute conditionally — so the answer is the per-path
  contributor report in P2.7, not an error.
  **Fixed 2026-09-07.** `_reject_undeclared` raises from both `_FlatRHS`
  and `_apply_assignments`, naming the process, the method, the undeclared
  ports and what it does declare. `_PortMap` carries the owner name and the
  declared set, so the check is one `frozenset` membership per returned key
  at trace time and never enters the jaxpr. Regressions in
  `test_composition.py::TestUndeclaredPortsRaise`, including the converse:
  omitting a declared port still freezes that state rather than raising.

- [x] **P0.4 — `dose_window=None` silently deletes a hallmark dial.**
  Documented as "sustained drive". `drive_pulse` is skipped, the pulse process
  never exists, and `HallmarkHandle.apply` skips mappings whose target is
  absent. Sweeping severity 0→50 returns the identical attractor to 4 s.f.
  **The exposed surface doubled on 2026-08-29:** Deregulated Nutrient Sensing
  now targets `nutrient_drive.after` the same way, so a composite built without
  that source silently loses the mTOR dial too.
  *Fix:* raise when every mapping of an applied hallmark misses its target.
  **Fixed 2026-09-08.** `HallmarkHandle.apply` raises when *every* mapping
  misses its target, naming the targets it wanted and the processes present.
  A partial miss stays legal: one hallmark spans composites that hold
  different subsets. Severity 0 raises too — the composite is misconfigured
  either way, and a dial that cannot turn is not made acceptable by sitting
  at its centre. `test_composition.py::TestHallmarkWithNoTarget`, and
  `test_models.py` had a test asserting the old ignore-silently behaviour,
  now inverted.

- [x] **P0.27 — An affine unit yields a garbage multiplier, silently.**
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
  **Fixed 2026-09-08 (the minimum fix).** `conversion_factor` probes
  linearity with `f(2) == 2 f(1)` and raises on an affine pair; pint refuses
  the doubling outright for an offset unit, which is the same answer. Ratio
  scales are unaffected (`day -> second` 86400, `uM -> mol/L` 1e-6).
  Carrying `(scale, offset)` per port is still not done, so a model that
  genuinely needs a temperature port raises rather than converting — the
  role-dependent offset in the entry above is what that would take.
  `test_validation.py::TestAffineUnitsAreRejected`.

- [x] **P0.32 — `semantic_validation={}` silently disables the entire
  validation layer.** Found 2026-08-31 (external systems review).
  `composite.py:400` is `if semantic_validation:`, and `{}` is falsy. Measured
  on a composite with a genuine `uM` vs `mol` conflict at a shared path:

  ```
  semantic_validation=True (default)    -> ValueError: Semantic validation failed
  semantic_validation={'strict': True}  -> ValueError: Semantic validation failed
  semantic_validation={}                -> CONSTRUCTED (no error)
  semantic_validation=False             -> CONSTRUCTED (no error)
  ```

  `docs/architecture.md` teaches the dict form ("opt out per subsystem with
  `semantic_validation={...}`"), so `{}` reads as "dict form, no overrides, i.e.
  defaults" and means the opposite.
  *Fix:* `if semantic_validation is not False and semantic_validation is not
  None:`. 15 minutes.
  **Fixed 2026-09-08.** The gate is `is not False and is not None`, so only
  those two opt out and `{}` means what the docs teach.
  `test_validation.py::TestEmptyValidationConfig`.

- [x] **P0.55 — `steady_state` returns NaNs silently: the guard is `res > tol`,
  and `NaN > tol` is False.** Filed 2026-09-05, found by the mathematician
  refereeing Hui 2016. `steady_state.py:548` warns only when the Newton
  residual exceeds tolerance. A diverged solve produces `res = nan`, the
  comparison is False, and the caller gets a state vector of NaNs with no
  warning at all — the one case where the warning matters most.

  Hit live: Hui 2016 has `d(AGEprod)/dt` identically 1e-6 at every state, so
  no fixed point exists, and `steady_state` returned **62 NaNs** without
  comment.

  *Fix:* guard on `not (res <= tol)`, which catches NaN, and say in the warning
  that a non-finite residual means no fixed point was approached rather than
  one was missed.
  **Fixed 2026-09-08.** The guard is `not (res <= tol)`, which catches NaN,
  and the warning now distinguishes a non-finite residual (the solve
  diverged, no fixed point was approached) from a merely loose one. The
  regression uses a system whose residual runs to infinity; a toy that
  reaches NaN through this Newton was not found, the reproducer being
  Hui 2016. `test_steady_state.py::test_a_non_finite_residual_warns_and_says_what_it_means`.

- [x] **P0.63 — A large integer literal in an SBML file overflows on import,
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
  **Fixed 2026-09-08.** The import pre-pass re-emits every literal with an
  integral value in exponent form (`1500e0`), so a constant power folds in
  float. An *exponent* is left as an integer: `x ** 6` is defined at negative
  `x` and `x ** 6.0` is not. Coercing the AST node alone was not enough —
  libsbml prints a real of integral value without a decimal point, so the
  translator re-parsed it as a Python int and the coercion was undone at the
  only place it mattered. BIOMD0000000293 goes REJECT to FLAG with a real
  rest residual (0.00225, was NaN), and DallePezze's published chi-squared is
  unchanged. `test_sbml_math_rewrite.py`.

- [x] **P1.19 — A composite is not bit-reproducible across a JAX pytree
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
  **Fixed, verified 2026-09-08.** Already closed by the constructor sorting
  added for the EVENT-ordering defect: `Composite.__init__` stores
  `{n: ... for n in sorted(flat_processes)}`, so the composite is canonically
  ordered before any round-trip and JAX's key sorting is a no-op. Re-ran the
  entry's own reproducer — six processes inserted unsorted, all writing one
  EVOLVED path — and the RHS is now bit-identical across `filter_jit`. The
  four unguarded sites it names iterate an already-sorted dict, so no change
  was needed; the property is pinned by
  `test_composition.py::test_a_pytree_round_trip_leaves_the_rhs_bit_identical`
  so it cannot regress silently.

- [x] **P2.8 — The CLI configures no logging, so every `log.info` in the
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
  **Fixed 2026-09-08.** The `simulate` group callback configures logging and
  takes `-v` / `-vv` / `-q`. The root stays at warnings while `-v` lifts the
  `hallsim` logger only, so verbosity means "what the framework decided" and
  not JAX's backend probing; `-q` quietens both. `force=True` so the level a
  user asked for wins over a demo module's own `basicConfig`, and that call
  is now guarded to the script path. `test_cli.py::TestVerbosity`.

- [x] **P0.53 — `composite_schematic` is drawn by hand and has been depicting a
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

  **Fixed 2026-09-08.** Membership, readouts and edge captions now derive from
  the composite: the blocks are drawn from whichever SBML processes are live,
  the readout lines from the run's own reporter set, and each arrow's caption
  from that edge process's `description`. What stays in a table is geometry —
  where a block or a label goes — plus the deposit accession, which cannot be
  derived because an imported process does not retain it (filed separately as
  P3.17). The guard now names the processes that have no slot rather than
  reporting a set difference, so the failure says what to add.

- [x] **P3.18 — DallePezze's ROS and Proctor's ROS are the same species held as
  two unwired pools.** Filed 2026-09-08 from the semantic validator's own
  warning. `ros_misfolding` drove Proctor's rate constant `k2` from
  DallePezze's ROS while `ups/ROS` stayed pinned at 10 inside the same rate
  law, `k2·NatP·ROS`, so the composite multiplied a rescaled ROS by an
  unrescaled one and hid the factor in the gain — Kounis's steady-state
  ratio, in a rate constant.
  *Fixed 2026-09-09.* `SBMLProcess.with_species_input` hands a species over:
  the port keeps its name and ontology and becomes INPUT, the model's own
  reactions stop moving it, and every rate law reads the external value.
  `ups/ROS` is now written by `ros_identity`, a level edge from `dp14/ROS`
  whose gain is the ratio of the two deposits' declared reference levels
  (10/10 = 1.0); `k2` stays at its published value and the validator no
  longer reports the pair. The primitive is SBML comp's replaced element
  with a conversion factor, in-process.

- [x] **P0.71 — The Scheduler's own advice for a batched stochastic run yields
  a population with one noise realisation.** Filed 2026-09-11. A batched `y0`
  on a composite with a stochastic reaction process is refused by
  `_reject_unsupported_batch` with "Run unbatched, drop the blocking feature,
  or vmap Scheduler.run from outside." Doing exactly that —
  `jax.vmap(lambda y: sched.run(comp, y0=y, seed=1).ys)` over three members
  of a 50-molecule decay model — returns three trajectories that are equal in
  every element: the integer seed becomes a single `jax.random.PRNGKey` inside
  the compiled core (`scheduler.py:1056`) and the key is broadcast across
  members. Vmapping the seed alongside `y0` gives distinct replicates, so the
  replicate axis is expressible today; nothing says so, and the documented
  route produces exactly the fake population the batch guard exists to
  prevent, with no warning. Probe:
  `scratch/2026-09-11-jkm-adoption/batched_ssa_probe.py`.
  *Fix:* accept a batched state with stochastic processes by splitting the
  key per member inside `_per_member` (its `in_axes` currently broadcasts the
  key), and let `run` take a `key` alongside `seed`. Until then the refusal
  message must say to vmap the seed too. `docs/design-stochastic-lane.md`
  already states the design: the replicate axis is the key axis.
  **Fixed 2026-09-11.** `Scheduler.run` takes ``key=`` alongside ``seed``;
  the compiled lane splits the key per batch member (`_per_member`), the
  eager lane derives one key per macro window with `jax.random.split` and
  `simulate_ssa` takes ``key=``, and the batch guard refuses a stochastic
  batch only on eager configurations. The probe's batched run now returns
  ``ys`` of shape (201, 2, 2) with members that differ; the outer-vmap route
  with a fixed seed still returns one realisation, and the refusal text now
  says to vmap the seed. Tests: `test_stochastic.py` (independent members,
  seed reproducibility, ``key`` ≡ ``seed``, eager lane seeded and refusing a
  batch).

- [x] **P3.20 — Initial values set by `<initialAssignment>` or by an
  assignment rule are not evaluated at import, so such a deposit does not
  load.** Found 2026-09-11 by the conformance suite. Nazaret 2009
  (BIOMD0000000232) sets ADP and NADH by assignment rules and DeltaPsi and
  six flux parameters by initial assignment, with no `initialConcentration`
  or `value` attribute; `sbmltoodejax` emits `None` for them and the import
  dies in `jnp.array`. The repository has carried a hand-edited copy,
  `nazaret2009_BIOMD0000000232_initialised.xml`, with the values typed in —
  which is the P0.63 pattern of patching the deposit instead of the
  importer. libRoadRunner and COPASI load the original. The original is a
  strict expected failure in `scripts/conformance.py`
  (`CANNOT_IMPORT`) until this is fixed.
  *Fix:* evaluate initial assignments and assignment rules at `t = 0` from
  the sympy form of each rule (`hallsim.sbml_math`) when the importer owns
  the translation, then delete the `_initialised` copy.
  **Fixed 2026-09-11** by the native importer (`hallsim.sbml_core`): initial
  assignments and assignment rules are evaluated at `t = 0` from their sympy
  form. The deposited Nazaret 2009 file imports unchanged and matches
  libRoadRunner to 1e-7 on both conformance windows; the `_initialised` copy
  is deleted and `demos/models/mitochondrial_aging.py` reads the deposit.

- [x] **P2.9 — A composite cannot be exported, so every cross-engine check
  needs a hand-written SBML merge.** Filed 2026-09-10. HallSim imports SBML and
  never emits it. To run the multi-hallmark composite in Tellurium and COPASI
  (P1.17) the three deposits had to be merged offline with libSBML into one
  file, and that merge is 300+ lines of reference surgery: `renameSIdRefs` does
  not descend into kinetic laws or update `speciesReference` species attributes,
  metaids are document-global and collide across documents, cross-document adds
  fail with a bare `-8` unless levels are normalized first (DallePezze is L2V4,
  the other two L2V1), assignment-rule targets need `setConstant(False)`, and
  display-name collisions are not checked by SBML validation at all — DallePezze's
  ROS and Proctor's ROS collided silently and were separated only by prefixing
  every element name by hand. All of that is a re-derivation of wiring the
  composite already holds as data, and it has to be redone by hand every time
  the wiring changes, which makes the conformance check expensive to keep
  current rather than expensive once.
  *Fix:* `Composite.to_sbml()` emitting the flattened model with the coupling
  edges as assignment rules and the clock reconciliation as rate-law scaling —
  the same translation the merge does by hand, from the topology that already
  describes it. It is also the interchange artifact for using COPASI as a
  stochastic and inverse-task engine on sub-models (P3.6), and the thing that
  makes the conformance test cheap to run on any composite rather than on one
  hand-merged file.
  **Fixed 2026-09-11.** `Composite.to_sbml()` (`hallsim.sbml_export`)
  writes the composite as one SBML L3V2 document: store paths as species
  (amounts, one unit compartment), a shared path one species, model
  constants and compartments prefixed parameters, edges as reactions and
  assignment rules from their declared symbolic forms, clock reconciliation
  compiled into rates and event triggers, SBML events, parameter drivers,
  parameter steps and input drivers included. The multi-hallmark composite
  exports and matches libRoadRunner and COPASI to 1.2e-6 over two days on
  42 species (`scripts/conformance.py --multi-hallmark`). Not covered:
  DISCRETE processes, and any hand-written process that declares no
  `reaction_channels()` / `assignment_rules()` — the mitochondrial demo's
  own modules today.

- [x] **P1.17 — Nothing checks HallSim against an established simulator; the
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
  *Evidence, 2026-09-10, and it is good news.* The whole multi-hallmark
  composite was rebuilt independently in Tellurium 2.2.13 / roadrunner 2.10 and
  in COPASI 4.46 via basico, by merging the three deposits into one 82-species /
  142-reaction / 20-rule SBML with the coupling edges compiled in as assignment
  rules and rate-law rescalings, and simulated over 50 days. Both engines
  reproduce the HallSim trajectory to integration tolerance: max relative
  deviation over 1001 points is 1e-6 to 9e-4 on DNA_damage, CDKN1A, ROS,
  phospho-mTORC1 and AggP in both arms and both engines. So the SBML semantics
  HallSim re-implements are, on this composite, right. Two caveats keep this
  entry open. The harness and its artifacts live outside this repo, so nothing
  here re-runs it. And the one outlier — COPASI's etoposide-arm p53 at 3.5e-2 —
  is not a discrepancy but a phase offset in a limit cycle scored with a
  pointwise metric, which is its own defect (see the oscillation-aware
  comparison entry below). Whether the merge was built against the current
  wiring or the pre-2026-09-09 one, in which DallePezze's ROS drove Proctor's
  `k2` rather than sharing its pool, is not recorded and needs establishing
  before the numbers are quoted.
  *Fix:* promote it to `tests/conformance/`, marked `slow` and gated on
  `pytest.importorskip("roadrunner")`, asserting a per-species relative-deviation
  bound on the bundled offline SBML. 2-3 days to make deterministic and bounded;
  it is ~80% written. Highest-value test asset available.
  *Progress 2026-09-11.* The single-model half is in-repo:
  `scripts/conformance.py` runs every vendored deposit and three
  synthetic models against libRoadRunner and COPASI on two windows, 31/31
  with the native importer. The composite half still needs `Composite.to_sbml()`
  (P2.9); until then the multi-hallmark cross-engine rebuild stays external.

  *Progress 2026-09-11, later.* The composite half has its instrument too:
  `Composite.to_sbml()` plus the composite case in the conformance suite.
  The multi-hallmark composite itself needs event export before it can go
  through it.
  **Closed 2026-09-11.** The instrument is `scripts/conformance.py` — by
  decision a development script, not a test, and the reference engines are
  not a dependency. Every vendored deposit, three synthetic models, a
  two-model three-edge composite, an event composite and the multi-hallmark
  composite itself match libRoadRunner and COPASI to integration tolerance.
  The document is generated from the composite's current wiring, so the
  "which wiring was the merge built against" caveat no longer arises.

- [x] **P0.72 — A composite nested inside another composite fired each SBML
  event twice.** Found 2026-09-11 by the export test. `expand_events` left
  the owner's `_events` in place after turning them into EVENT processes, so
  when the composite was flattened into an outer one `_compose_events`
  expanded them again under a second name; the collision check compares
  names, and the second copy had a different one. Both copies fired.
  **Fixed 2026-09-11:** expansion returns the owner through
  `without_events()`, so a nested composite carries nothing to expand twice
  (`test_a_nested_composite_expands_each_event_once`).

- [x] **P0.73 — An event on a clock-reconciled model fired at the wrong
  composite time.** Found 2026-09-11 while moving `sbml_events` onto
  `sbml_math`. Event math is written in the model's native time, and
  `SBMLEvent.condition` compared the composite's `t` against it directly, so
  a model passed through `reconciled_to` had its timed events fire at
  native-time instants read as composite time — off by the clock ratio. No
  vendored demo composes an event-bearing model, which is why it went
  unseen. **Fixed 2026-09-11:** an event carries its owner's `time_scale`,
  set by `expand_events`, and evaluates trigger and assignments at
  `t · time_scale` (`test_a_reconciled_model_fires_its_event_on_the_composite_clock`).

- [x] **P1.9 — Conservation laws are still inferred numerically for any
  composite containing a hand-written process.** The exact stoichiometric path
  needs every process to declare `stoichiometry()`; one undeclared edge disables
  it. **Fixed 2026-09-11:** `stoichiometry()` now derives from
  `reaction_channels()`, which the seven edge primitives and both forcing
  sources declare, and `conservation_laws` no longer needs everyone. The
  exact integer left null space of `N` over the paths nothing moves outside
  it is taken as it is; only a candidate touching a path an undeclared
  process or a rate rule also moves goes to the sampled Jacobians, and then
  within the span `N` allows rather than the whole space. A composite of
  declared processes samples nothing, one with none declared samples as
  before (`test_exact_laws_survive_an_undeclared_neighbour`,
  `test_an_undeclared_writer_on_a_declared_moiety_breaks_it`).
  `hallsim.structure.composite_moieties` states the exact laws as integer
  coefficients over store paths.

- [x] **P1.13 — Structurally redundant parameters are invisible before a fit.**
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
  **Fixed 2026-09-11:** `hallsim.identifiability.structural_redundancy`
  differentiates the composite's field, assembled as sympy from the declared
  forms (`hallsim.structure.symbolic_field`), with respect to each parameter
  and groups those whose sensitivities are proportional by a factor free of
  state and time — the same rate law on the same stoichiometry, or two
  constants that only ever multiply. On the DallePezze deposit it names
  `mito_biogenesis_by_mTORC1_pS2448` and `mito_biogenesis_by_AMPK_pT172`
  with ratio 1, and two clamps on one target the same way, since the edges'
  parameters are symbols in their laws too. `CalibrationProblem` runs it on
  the fitted set at construction and warns naming the group
  (`TestStructuralRedundancy`). A parameter reaching a process with no
  symbolic form is reported unassessed rather than cleared.

- [x] **P0.74 — The stochastic lanes read the store raw: a member's
  non-species ports were missing and its ASSIGNED inputs held stale
  values.** Found 2026-09-11 by taking the calibration gradient of the
  multi-hallmark composite with Proctor 2007 at reaction level. The
  compiled lane handed `reaction_propensities` the species vector only, so
  a driven constant (`p07/k1_in`) raised `KeyError`; and both lanes read a
  species input from the flat state, where an ASSIGNED path holds its
  initial value between windows because the assignment pass runs inside
  the ODE right-hand side. Proctor saw ROS = 0 for fourteen days: no
  misfolding, 6 350 events, native protein climbing to 6 550 while the mean
  field settles at 252 — plausible-looking and wrong. **Fixed 2026-09-11:**
  both lanes apply the composite's assignment pass to the window state
  before the member reads it, and every non-species port is read there and
  held over the window, as any Lie-coupled input is; the direct runner
  accepts such ports in `y0`
  (`test_a_driven_constant_reaches_the_stochastic_member`). At the
  published vector the SSA now tracks the mean field: 605 549 events over
  fourteen days, MisP 4 against 5.5, NatP 243 against 252, free proteasome
  86 against 85.

- [x] **P0.75 — The compiled stochastic lane would stop a window at 100 000
  reaction events and say nothing.** Found 2026-09-11 in the same
  experiment, by reading rather than by being bitten: `ssa_step_jax` bounds
  its `while_loop` by `max_events`, which the scan lane fixed at 100 000 per
  macro window, and a window that reached it continued from the truncated
  state with no record of it. Measured on Proctor 2007 in the multi-hallmark
  composite, no half-day window reaches it (605 549 events in 28 windows,
  the same trajectory under either bound), so nothing here was cut; the
  defect is the silence. The bound is a loop bound with no buffer behind it,
  so there was nothing to save by keeping it small. **Fixed 2026-09-11:**
  the bound is 10 million per window (`SSA_MAX_EVENTS_PER_WINDOW`), the run
  records whether any window reached it (`stats[member]["event_cap_hit"]`)
  and warns eagerly when one did.

- [x] **P0.76 — The sympy code generation shared nothing: every rate law
  recomputed what the others had, and each assignment rule was its own
  function.** Found 2026-09-11 by another session profiling the native
  importer against the `sbmltoodejax` path it replaced: XLA's cost analysis
  of the compiled DallePezze fast path showed 1.8× the arithmetic for fewer
  instructions, with the per-call RHS benchmark pointing the other way
  because an isolated batched call is launch-bound. `sbml_math.to_jax`
  printed each tree straight through `lambdify` with no common-subexpression
  elimination, `compile_sbml` lambdified the 41 laws as one tuple but the 14
  assignment rules as 14 functions, and `materialize_assigned` paid for
  every one of them per saved point. **Fixed 2026-09-11:** every
  non-boundary assignment is substituted into what reads it, in dependency
  order, so laws, rate rules and assignments form one expression list,
  lambdified once with sympy's `cse`; a boundary species' rule keeps its
  read of `w`, which is where a driver overrides it. Measured on
  DallePezze, batch 256, CPU: RHS bytes accessed 3.39e6 → 1.90e6,
  `materialize_assigned` over 141×256×37 2.7 → 1.1 ms, the compiled 14-day
  solve 3.23 → 2.88 s. Every cross-engine conformance case passes on the new
  code. The other session filed this defect as P0.72 in its own tree; that
  id is taken here, so it is P0.76 on merge.

- [x] **P1.25 — A pinned implicit solver whose root finder follows the
  controller's tolerance stalls seventeen-fold under diffrax 0.7, and
  nothing says so.** Found 2026-09-11 benchmarking against jaxkineticmodel.
  On the DallePezze field, ``Kvaerno5()`` as shipped (a chord iteration
  whose ``rtol``/``atol`` are taken from the step controller) at
  ``rtol=1e-10, atol=1e-12`` takes 18 260 steps and rejects 9 137 on this
  environment's diffrax 0.7.2, optimistix 0.1.0 and lineax 0.1.0; the
  identical generated field under diffrax 0.6.1 takes 1 142 steps and
  rejects 2, and a full Newton at ``atol=1e-12`` on the new stack hits the
  300 000-step cap. The Scheduler's own default,
  ``Kvaerno5(root_finder=optx.Newton(rtol=rtol, atol=1e-6))``, takes the
  1 142 steps here too, so the default is right and the trap is
  ``Scheduler(solver=...)`` or ``implicit_solver=...`` with a plain diffrax
  implicit solver at a tight controller tolerance. A user who matches
  another tool's tolerances gets a solve seventeen times slower than the
  default with no warning. *Fix:* when a pinned implicit solver's root
  finder inherits the controller tolerances and ``atol < 1e-8``, warn at
  construction naming the default's root finder, or substitute it.
  **Fixed 2026-09-12:** the Scheduler installs its own root finder into any
  implicit solver it is handed whose root finder would copy the
  controller's tolerances — `solver=`, `implicit_solver=`, or the default —
  and keeps one the caller set explicitly
  (`TestPinnedImplicitSolverRootFinder`). `Scheduler(solver=dfx.Kvaerno5())`
  now means what it says, at any tolerance.

- [x] **P0.77 — The compiled stochastic lane held a member's window-start
  value across every intermediate save point, so a saved trajectory of a
  fast stochastic species was a staircase of stale values.** Found
  2026-09-12 drawing Proctor 2007's reporters as a population on the
  calibrated trajectory: UBB's pooled fold-change came out at +1.3 log2 in
  the etoposide arm where the mean field and the data both sit near +0.05,
  with every cell agreeing — not sampling noise. Free ubiquitin starts at
  its published 500 and binds down to about 40 within minutes, and the lane
  wrote the member's state only at the end of each macro window, so the
  first window's four intermediate save points still read 500 and the
  reporter's two-day zero-phase mean at day 0 averaged them in. The
  species pools themselves were right (pooled free ubiquitin 40, 36, 47,
  38 against the mean field's 33, 33, 39, 35 over days 0, 3, 7, 14); only
  the recorded trajectory between window ends was wrong, and everything
  that reads a trajectory — reporters, figures, `materialize_assigned` —
  read it. **Fixed 2026-09-12:** `ssa_window_jax` records the sampled path
  at every save point of the window and the lane writes all of them
  (`test_saved_trajectory_carries_the_sampled_path_inside_a_window`). With
  the fix the 64-cell population mean sits within 0.1 log2 of the mean
  field on both Proctor reporters, both arms, both days.
- [x] **P0.78 — With the chord root finder as the Scheduler's default, every
  reverse-mode gradient through the multi-hallmark solve raised, and with
  the error demoted every gradient was NaN; the forward solve was fine and
  nothing said so.** Found 2026-09-12, four hours after the chord landed
  as the default, by the first gradient anyone took through it: the
  September 10 calibration's own three-parameter loss. The same loss with
  an `optx.Newton` root finder gives 0.27903 and finite gradients. The RHS
  and its tangents are finite everywhere on the trajectory. Instrumenting
  the root finder: the very first step, from the published launch state at
  the Scheduler's fixed `dt0 = 1e-3` days (300 times Proctor 2007's
  0.265 s relaxation), stage 2 of Kvaerno5 runs optimistix's chord for its
  full ten iterations growing by 1e8 per iteration to 2.4e143, and stage 3
  starts from that and returns NaN. The forward solve rejects the step and
  shrinks; the reverse pass factorises the Jacobian at the returned
  iterate, and a NaN factor times the rejected branch's zero cotangent is
  NaN, so one such step poisons every gradient. Optimistix's chord with
  Cauchy termination has no divergence stop, and the rate-based stop its
  other mode and diffrax's `VeryChord` use cannot see this case: the
  relative update size saturates at `1/rtol` while the iterate grows, so
  the rate is exactly 1. A guard on the finiteness of the iterate does not
  help either (1e143 is finite). Newton never leaves O(1), which is why
  the earlier default never showed it. **Fixed 2026-09-12:**
  `hallsim.root_finders.Chord`, the Scheduler's default, carries the
  residual of its current iterate, refuses an update whose residual is
  non-finite or more than `growth_limit` (10) times larger, and so returns
  only iterates whose residual it evaluated. One extra residual per solve.
  Gradient equals Newton's to six figures; on the multi-hallmark control
  arm 1 048 ms against the unguarded chord's 1 002 ms and Newton's
  1 434 ms, the same step count as the unguarded chord (a limit of 2 cost
  11 % more steps; 10, 100 and 1 000 the same steps). Conformance: 0 failing on every case. The launch step
  itself is P1.26. `test_the_scheduler_chord_stops_diverging_with_a_finite_iterate`.

- [x] **P1.26 — Every group starts its first step at the Scheduler's fixed
  `dt0 = 1e-3`, whatever the group's fastest rate, so a stiff launch begins
  with a guaranteed rejection and a nonlinear solve that has to diverge
  first.** Found 2026-09-12 tracing P0.78. On the multi-hallmark composite
  the Proctor 2007 group's fastest relaxation is 0.265 s and the first
  attempted step is 86 s; the chord inside it grows to 1e143 before the
  controller ever sees an error estimate, and the diary's population runs
  record 31–153 rejected steps per member at launch for the same reason.
  jaxkineticmodel never meets this: it starts every solve at `dt0 = 1e-12`
  and lets the controller grow the step. The Scheduler already measures
  each group's spectral abscissa for stiffness routing and carries a
  per-group `dt0_hint` across macro steps; only the first macro step
  ignores what it knows. *Fix:* start each routed group at
  `min(dt0, c / spectral_abscissa)` (or diffrax's own initial-step
  estimate when routing is unavailable), and measure the launch rejections
  before and after on the multi-hallmark and the 1024-member population. **Fixed 2026-09-12:** `DEFAULT_DT0` is `None`; the Scheduler estimates each group's first step from its field at the launch state (Hairer's rule, diffrax's own for `dt0=None`, in `Scheduler._initial_step`), carries it into the compiled lanes as the group's first hint, and passes `dt0=None` through on the eager lane and the fast path; a float still pins it. Measured on the multi-hallmark control arm the gain is small — 44 rejected launch-group steps against 46, 402 against 408 in the other group, trajectories within 1e-4 relative — so the 31–153 launch rejections are the controller working through the transient, not the first step; the claim above overstated the step's share. On a stiff cubic launch (`test_initial_step.py`) the pinned step rejects and the estimated one does not, and the ten-iteration chord divergence of P0.78 no longer has a step to happen in.
