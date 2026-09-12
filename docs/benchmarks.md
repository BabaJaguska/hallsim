# Benchmarks

Measured on one machine: Apple Silicon CPU, JAX 0.7.2, diffrax 0.7.2,
equinox 0.13.5, float64, 340-test suite green. Absolute numbers are
machine-specific; the ratios are the point.

Reproduce with `python scripts/bench.py` (see [Reproducing](#reproducing)).

---

## 1. `Scheduler.run` per-call trace cost

The scan path builds a `lax.scan` whose body contains one `diffeqsolve` per
group. Rebuilding that jaxpr on every call cost **~340 ms per group**, linear
in group count:

| groups | before | after | |
|---|---|---|---|
| 2 | 719 ms | 1.26 ms | **571×** |
| 4 | 1300 ms | 1.69 ms | **769×** |
| 8 | 2623 ms | 3.79 ms | **692×** |
| 16 | 6187 ms | 6.67 ms | **928×** |

Trivial 2-state models, so this is *entirely* framework overhead — the same
computation issued directly to diffrax takes 0.2 ms. On the real composite,
where the ODE work is substantial, the same fixed cost is a smaller fraction:

| multi-hallmark, 14 days, 52 vars, 2 stiff groups | |
|---|---|
| before | 10.6 s |
| after | 6.3 s |
| of which was re-tracing | 3.3 s (31%) |

Solver step counts are bit-identical across the change (group_0 1933 steps /
405 rejected, group_1 3834 / 1146), so the numerics are untouched.

**Why it went unnoticed.** Tracing and compilation are distinct. The XLA
compile cache is keyed *on the trace*, so an unjitted function reuses the
compiled executable but must re-trace to look it up. `jax.log_compiles`
therefore reported **zero recompiles** the whole time it was costing seconds
per call. `test_performance.py` asserted the right property and measured the
wrong half — and its fixture was a single process, so it only exercised the
fast path, where `diffeqsolve` is traced once and tracing is cheap.

**Fix.** `Scheduler.run` splits into an eager prologue (groups, coupling,
stiffness routing, save grid — all structural) and a compiled core cached by
`_core_signature`. Two constraints made it possible:

- `SBMLProcess._species_y0` and friends had to become `eqx.field(static=True)`.
  They are port defaults and index maps — structure, not fitted values — but as
  traced leaves they made `ports_schema()` raise `ConcretizationTypeError`
  under a trace, which blocked `initial_state_vec`, `build_rhs`, and
  `_effective_coupling`, i.e. all of `run()`.
- `_save_grid` and `_collect_jump_ts` return **numpy**, not `jnp`. A `jnp`
  array built while an outer trace is live is a tracer; captured by a cached
  closure it escapes that trace, and the cache is skipped entirely when the
  core is built under one.

---

## 2. What Lie splitting actually buys

| multi-hallmark, both via `Scheduler.run` | wall | solver steps |
|---|---|---|
| split — 2 groups, as shipped | 7.5 s | 5767 |
| merged — 1 group, fast path | 14.1 s | 4466 |

**Splitting wins 1.8×** even though both groups route to the *same* solver
(Kvaerno5) and both run at the full 52 dimensions — and the merged case had
the advantage of the fast path, one `diffeqsolve` over the whole span with no
macro-step restarts.

The mechanism is step size, not dimension, and total step count is the wrong
metric. Merged, one controller serves all 50 evolving states, so every step is
dictated by the stiffest mode: 4466 steps that *every* state must satisfy.
Split, group_0 takes 1933 at its own pace and group_1 3834 at its own, and
neither drags the other.

## 3. Restrict each group's solve to its own states

Splitting decouples the dynamics correctly — off-group derivatives are
*exactly* zero during a group's solve — but it does not shrink the system
handed to the solver:

| group | evolving states | dimension solved | Jacobian density |
|---|---|---|---|
| group_0 (nfkb + 2 edges) | 24 | 52×52 | 4.6% |
| group_1 (gz06 + dp14 + 3 edges) | 26 | 52×52 | 3.1% |

Newton factorises 52×52 for 24 real unknowns and `jacfwd` runs 52 JVPs instead
of 24 — 8–10× the linear algebra, 2× the JVPs. Restricting each group to its
own evolved indices, splicing the frozen off-group values inside the RHS,
measures **1.86×** and **2.78×**, and compounds with the 1.8× above.

This is the half that scales badly for a composition framework: group size is
fixed, total dimension is not, so the waste grows with every model added.

**Shipped** as `_ReducedRHS` (`scheduler.py`), measuring **2.30×** at
`rtol=1e-6` and **2.67×** at `1e-8` against a worktree at the prior HEAD. The
frozen-coupling caveat was wrong: `fill(t)` supplies the off-group states under
interpolated coupling too, so the solved dimension is the same either way.

Build the restricted RHS *once*, outside the timing loop. A closure rebuilt per
call is a static leaf that rehashes, so every solve misses diffrax's cache and
the measurement inverts — the restricted arm came out 3× *slower* until that
was fixed. `_FlatRHS` is an `eqx.Module` for exactly this reason; hand-rolling
a solve around the Scheduler reintroduces the bug the framework already fixed.

---

## 4. Solver choice: `optx.Newton` vs diffrax's default

GPU batching has a different tradeoff: the optional
`hallsim.root_finders.StepChord` reuses a Jacobian across implicit stages
while retaining Cauchy convergence. It improves the tested 256-cell GPU
workload but is slower for small batches, so Newton remains the default.
See [the Jacobian reuse profile](jacobian-reuse.md) for timings, accuracy,
and CUDA dispatch costs. The historical measurements below are not a
general statement that Jacobian reuse can never help.

`Kvaerno5`'s default `VeryChord` root finder reuses a stale Jacobian. On real
biochemical RHSs that rejects a third of all steps:

| root finder | steps | rejected | wall |
|---|---|---|---|
| `optx.Newton` (shipped) | 5,767 | 21% | 14.9 s |
| `VeryChord` (diffrax default) | 103,468 | 36% | 45.9 s |

**18× fewer steps**, which is why `Scheduler.implicit_solver` overrides the
default. The remaining 21% rejection rate is *not* a Newton-tolerance problem:
loosening its `atol` from 1e-9 to 1e-3 moved the step count by 0.5% (5767 →
5735) and the rejection rate not at all. Max |y₀| is 25, so a 1e-9 absolute
tolerance is not fighting large-magnitude states.

**Explained, and there is nothing to win.** `dfx.PIDController(rtol, atol)`
takes diffrax's defaults `pcoeff=0, icoeff=1, dcoeff=0` — the class implements
PID, the instance is a pure I-controller. PI control cuts rejection to 12.5%
and leaves total work flat (~4150–4200 steps either way), so the wall time does
not move. The name is a trap; the rejection rate is not the cost.

---

## 5. RHS graph composition

One derivative evaluation of DallePezze 2014 (23 species, 56 constants) traces
to 334 jaxpr equations:

| stage | eqns |
|---|---|
| constants-vector rebuild | 123 |
| port view, flat → dict | 69 |
| species re-stack, dict → array | 26 |
| scatter back | 58 |
| **the actual rate laws** | **~58** |

Two things were tried here and **both were rejected on measurement**:

- **Hoisting the constants vector** out of the RHS cut the full composite from
  863 to 747 equations with a `0.0` trajectory difference — and *no* runtime
  change, because XLA's loop-invariant code motion already lifts it out of the
  solver loop. The clean implementation (memoising on the instance) would also
  have made the pytree structure change after first use, losing the JIT cache.
- **Skipping the `× 1.0` unit conversion** when port and path units agree cut
  810 → 749 jaxpr equations, and the optimised HLO was *identical*: 161
  multiplies either way, 41.8 vs 40.2 µs. XLA folds `x * 1.0 → x` before
  codegen.

Both are recorded here so the next person doesn't re-derive them. The
remaining plumbing is a real cost only for *tracing* — which is no longer a
footnote, see item 6.

---

## 6. Where the time goes now: the one-time cost dominates

Full audit in [performance-audit.md](performance-audit.md). On the multi-hallmark demo:

| stage | wall |
|---|---|
| `import hallsim` | 0.6–0.7 s |
| composite build | ~1.1 s |
| trace + lower + compile | **~11 s** |
| the solve | **~3.2 s** |

Only the last line is arithmetic. Python dispatch is *not* a factor — 2.298 s
of a 2.308 s warm run is inside the compiled executable.

**A persistent compilation cache takes compile 11.70 s → 7.76 s (1.5×)**, on by
default at `~/.cache/hallsim/jax`. The threshold is the whole trick: at JAX's
default `min_compile_time_secs=1.0`, and at 0.05, it stores **four entries and
saves nothing**. One run emits ~250 individually-fast executables and the cost
is their sum, so the floor is 0.0 with `min_entry_size_bytes` at 0 as well.

The residue — ~7.8 s of tracing and MLIR lowering — is Python, scales with
jaxpr size, and no cache can skip it. That makes item 5's plumbing count the
live target rather than a curiosity.

**Rejected on measurement:**

| what | result |
|---|---|
| `--xla_cpu_multi_thread_eigen=false` | 2.247 → 2.276 s. Nil — the solve is single-threaded and has no parallelism to exploit |
| `--xla_cpu_enable_fast_math=true` | 2.247 → 2.100 s (6.5%). Not worth FTZ and no-NaN reassociation at `atol=1e-9` with curated oscillators |
| Lowering `max_steps` to shrink the reverse-mode checkpoint count | 32.1 s vs 27.7 s at the 4M default. `DEFAULT_MAX_STEPS` is not the lever |
| Sharding the *existing vmapped* batch axis | 0.83× — slower than doing nothing. One vmapped `while_loop` has one trip-count predicate, so SPMD adds a cross-device reduce instead of splitting the loop. Needs `shard_map` (2.4×) — **not reproducible on JAX 0.10.2 (2026-09-10): `shard_map` around `Scheduler.run` fails inside lineax's LU solve; see §6 and P0.70** |
| Parallelising forward-mode parameter directions | 1.09×. The vmapped JVP already shares one primal solve |

**A caveat on every absolute number above.** Re-running the same measurement
hours later on this machine gave warm 3.2 s against 2.25 s and first-call ~14 s
against 9.1 s — ~40% drift, no code change. Ratios held. Trust a *difference*
only when its arms were interleaved serially in one session on an idle machine;
treat a *level* as needing a re-measure.

---

## 6. Batched populations on CPU: measured, and the claim does not hold past 64

Multi-hallmark composite with Proctor 2007 attached — 79 states, 60 of them
integrated, 9 processes, 2 stiff groups — control arm over 14 days,
`macro_dt` 0.5, `save_dt` 0.5, initial conditions jittered log-normally
(σ = 0.1) on the integrated states, one `Scheduler.run` per size, second call
timed as warm. Apple Silicon, 11 cores, no GPU; the batch used 3.7 cores.
`scratch/2026-09-10-batch/batch_1024.py`.

| batch | cold | warm | per member | against B single runs |
|---|---|---|---|---|
| 1 | 10.2 s | 0.5 s | 528 ms | — |
| 64 | 30.5 s | 21.0 s | 329 ms | 1.6× faster |
| 256 | 118 s | 113 s | 442 ms | 1.2× faster |
| 1024 | 693 s | 724 s | 707 ms | **0.7× — slower** |

Every member solved at every size. The same 1024 population as 16 chunks of
64 takes 418 s, 1.7× less than the one-shot batch and 1.3× less than 1024
single runs, so a Python loop over chunks beats the framework's batch path
(P0.70).

**Mechanism, measured.** Inside one chunk of 64 the per-member solver step
counts spread from ~165 to 258–590 in the slow group (rejections 7 to 68),
and the chunk's wall time tracks its *maximum*: 20.8 s at max 258, 30.9 s
at max 590. One vmapped `while_loop` has one trip count, so every member
steps as many times as the slowest; the one-shot 1024 stepped ≥ 590 times
for a median member that needs ~170. A 10% jitter on the initial condition
is enough to open that spread. The three members that left the basin
(Proctor's runaway-aggregation state, reached deterministically) sat in
chunks that were not the slowest, so this is the ordinary spread of an
adaptive solver over a population, not a pathological member.

**What does work on this machine: worker processes over chunks.** The same
1024 as 16 chunks of 64 over 3 spawned processes, each with its own
Scheduler and its own compile: **266 s, 260 ms per member**, 2.7× faster
than the one batched call. Nothing is traced across members. `shard_map`
and `pmap` over host devices both fail inside lineax's LU solve on this
stack (P0.70). The GPU claim ("near-flat") is unmeasured on this machine.

| how the 1024 were run | wall time | per member |
|---|---|---|
| one batched `Scheduler.run` | 724 s | 707 ms |
| 1024 single runs | 540 s | 528 ms |
| 16 chunks of 64, sequentially | 418 s | 408 ms |
| 16 chunks of 64, 3 worker processes | **266 s** | **260 ms** |

## 7. What the Scheduler is for, measured against a bare solve

`scripts/bench_scheduler.py`, 2026-09-11. Every row solves the same reduced
field with the same Kvaerno5, the same chord root finder, the same PID
controller and `dt0=None`; only the orchestration differs. The reference is
a bare Kvaerno5 at `rtol 1e-10, atol 1e-13`; *error* is the largest
deviation over the saved points divided by the largest reference value.
Warm is the median of three calls, cold the first call with the compilation
cache off. Two systems: a synthetic diffusion chain (τ = 1, sinusoidal drive
of period 10) closed through an 8-state stiff relaxation block (k = 10⁴) in
a loop, span 40; and the multi-hallmark composite (60 evolved states, dosing
jump at day 2, span 14). The save grid is `macro_dt / 15`.

**Same solver, whole system.** The Scheduler with one group against the
bare call — the cost of the wrapper.

| system | bare Kvaerno5 | Scheduler, one group |
|---|---|---|
| chain, 72 states | 75 ms, 139 steps | 81 ms (+8 %) |
| chain, 264 states | 827–924 ms, 122 steps | 957–969 ms (+5 to +16 %) |
| chain, 1 032 states | 10.1 s, 106 steps | 10.1 s (0 %) |
| multi-hallmark, 60 states | 2.65–3.6 s, 1 707 steps | 2.57–3.3 s (−3 to −7 %) |
| gz06 alone, 3 species | 6.7 ms | 8.6 ms (+28 %) |

Identical steps, identical error. The wrapper is a fixed few milliseconds
per call, visible on a 7 ms solve and gone on a 3 s one.

**The split, chain with 264 states.** Routing puts the stiff block on
Kvaerno5 and the chain on Tsit5. Time, then error.

| macro step | Lie, interpolated (auto) | frozen | Strang | interpolated, 2 sweeps | bare Kvaerno5 | bare Tsit5 |
|---|---|---|---|---|---|---|
| 1.0 | 76 ms, 0.50 | 78 ms, 0.50 | 78 ms, 0.078 | 189 ms, 0.021 | 827 ms, 4.0e-5 | 1.95 s, 7.0e-6 |
| 0.25 | 218 ms, 0.13 | 200 ms, 0.13 | 225 ms, 0.018 | 431 ms, 5.6e-4 | 924 ms, 4.1e-5 | 1.95 s, 8.1e-6 |

**Chain with 72 states**, where the bare solve is 75 ms at every macro step
and bare Tsit5 1.25–1.35 s:

| macro step | auto | Strang | 2 sweeps |
|---|---|---|---|
| 1.0 | 55 ms, 0.50 | 57 ms, 0.078 | 130 ms, 0.021 |
| 0.25 | 167 ms, 0.13 | 178 ms, 0.018 | 325 ms, 5.6e-4 |
| 0.1 | 356 ms, 0.054 | 382 ms, 0.0069 | 798 ms, 1.9e-4 |

**Chain with 1 032 states, macro step 0.25:**

| auto | frozen | Strang | 2 sweeps | bare Kvaerno5 | bare Tsit5 |
|---|---|---|---|---|---|
| 271 ms, 0.13 | 252 ms, 0.13 | 292 ms, 0.018 | 516 ms, 5.6e-4 | 10.1 s, 7.0e-5 | 3.69 s, 1.6e-5 |

**Multi-hallmark composite**, both groups implicit:

| macro step | auto | frozen | Strang | 2 sweeps | bare Kvaerno5 | bare Tsit5 |
|---|---|---|---|---|---|---|
| 0.5 | 1.37 s, 1.2e-3 | 1.59 s, 5.8e-2 | 1.68 s, 2.6e-2 | 2.68 s, 1.2e-3 | 3.60 s, 4.3e-6 | 61 s, 1.1e-9 |
| 0.1 | 1.24 s, 2.2e-4 | 1.90 s, 1.3e-2 | 2.23 s, 3.9e-3 | 2.51 s, 2.2e-4 | 2.65 s, 4.3e-6 | 46 s, 1.1e-9 |

(The two multi-hallmark rows were measured hours apart; the bare column
drifted by a third with no code change, as §6 warns. Ratios within a row
hold.)

After §8's two changes to the RHS the same night — the assignment pass
pruned to what the loop reads, and the stoichiometry product folded into
per-species expressions — an interleaved A/B against the commit before
them (98c3665), two rounds each, same session, macro step 0.5:

| tree | bare Kvaerno5 | Scheduler, auto |
|---|---|---|
| 98c3665 | 2 562 / 2 566 ms | 1 020 / 1 008 ms |
| with §8's changes | 1 035 / 1 038 ms | 382 / 382 ms |

2.5× on the bare solve and 2.65× on the default lane, identical error
(1.17e-3), steps 2 133 → 2 136. A 14-day multi-hallmark run through the
Scheduler is 0.38 s.

**What it says.**

- The split pays where it is for. At 1 032 states it is 20× faster than the
  best bare solve at an error of 5.6e-4 (two sweeps), or 37× at 0.13 (the
  default); the bare implicit solve factorises a 1 032² Jacobian per stage.
  At 72 states it never pays: the bare solve is 75 ms.
- On the multi-hallmark composite the default is the best point on the
  table: 2.6× and 2.1× faster than the bare implicit solve at 1.2e-3 and
  2.2e-4. A second sweep buys nothing there (forward edges dominate), and
  Strang is 20× worse because it forbids the interpolant.
- On a feedback loop the default is first order in the macro step
  (0.50 → 0.13 → 0.054) and interpolated equals frozen, because the group
  solved first sees the other frozen. A second sweep buys 200× at 2× the
  cost; Strang 7× at 5 %. The plan now warns when its coupling graph has a
  cycle and one sweep (P1.28).
- Explicit on everything crosses over: bare Tsit5 loses to bare Kvaerno5 by
  17× at 72 states, by 2× at 264, and wins by 2.7× at 1 032. Routing decides
  per group, so the split never has to pick.
- Found on the way: routing crashed above 512 states on a clustered
  spectrum (P0.82, fixed); interpolated coupling returns its own save grid
  (P1.27); the routing verdict is measured at `y0` only — gz06's abscissa
  there put it on the implicit solver where bare Tsit5 was 2.7× faster and
  more accurate.

## 8. Where the derivative's time goes

`scripts/bench_field.py`, 2026-09-11. DallePezze 2014 (23 species, 37 store
slots), one Kvaerno5, one chord, one PID controller, `dt0=None`, 141 saved
points; five presentations of the same field. *program*: the member's
compiled program called bare. *generated*: the field re-emitted from
`symbolic_field` as one CSE'd function — **a different problem** (P1.29: it
holds the irradiation pulse on for ever; 1 140 steps against 663), kept only
as the per-call floor for the code generation. *flat*: the composite's RHS
reduced to the evolved states, as the Scheduler integrates it. *no assign
pass*: the same with the assignment-rule pass removed. *Scheduler*:
`Scheduler.run`. Batched call is 256 states in one `vmap`.

Before the assignment pass was pruned:

| form | jaxpr eqns | batched call | solve, 1e-10/1e-12 | solve, defaults |
|---|---|---|---|---|
| program | 469 | 67 µs | 103 ms, 663 steps | 18.5 ms, 111 steps |
| generated (other problem) | 204 | 16 µs | 68 ms, 1 140 steps | 6.9 ms, 105 steps |
| flat | 1 256 | 92 µs | 115.5 ms | 20.1 ms |
| flat, no assign pass | 610 | 64 µs | 78.2 ms | 13.9 ms |
| Scheduler | | | 117.2 ms, cold 5.1 s | 20.3 ms |

After — the RHS keeps only the assignments something in the loop reads:

| form | jaxpr eqns | batched call | solve, 1e-10/1e-12 | solve, defaults |
|---|---|---|---|---|
| flat | 610 | 65 µs | 78.6 ms | 13.6 ms |
| Scheduler | | | 80.2 ms, cold 2.9 s | 14.1 ms |

Results identical to 6e-15 at day 14 in every row but *generated*.

After the second change of the night — the member program emits one rate
expression per species instead of a dense stoichiometry product over every
reaction per call:

| form | jaxpr eqns | batched call | solve, 1e-10/1e-12 | solve, defaults |
|---|---|---|---|---|
| program | 542 | 31 µs | 42.5 ms, 669 steps | 7.7 ms, 111 steps |
| flat | 678 | 39 µs | 56.4 ms, 672 steps | 10.1 ms |
| Scheduler | | | 58.6 ms, cold 3.0 s | 10.1 ms |

Summation order changed, so the rows now agree at rounding level rather
than bit level (1.1e-9 at day 14 at the tight tolerance, 3.8e-11 at
defaults) and the step count moved by nine. Over the night: DallePezze
through the Scheduler 117.2 → 58.6 ms at 1e-10/1e-12, 20.3 → 10.1 ms at
defaults, first call 5.1 → 3.0 s.
On the multi-hallmark composite the two changes together are 2.65× on the
default lane (§7's A/B table).

- The composite re-ran every member's assignment rules — the member's
  whole program — before each derivative, to fill ASSIGNED store slots the
  derivative never reads: an SBML member computes its own rules inside its
  program. On a model with 14 rules that was 32 % of the solve and 43 % of
  the compile. `build_rhs` now traces each derivative once on an abstract
  state to see which ports it reads and keeps an assignment only if a
  derivative, or an assignment a derivative reads, consumes it. The saved
  trajectory gets every assigned value through `materialize_assigned` as
  before.
- The Scheduler is now 2 % above the bare flat solve, and below the member's
  program called bare in its own species order; that last gap was not
  chased.
- Per call the composite's form cost 4× the standalone generated function
  (65 against 15 µs): element reads of `y`, `w` and `c`, and a dense
  stoichiometry product the generated form folds into its expressions. The
  product is now folded the same way (39 µs, 2.4×); the reads and the port
  view's gather, slice and restack are what is left of P3.21, and the solve
  bounds them by the implicit step's share of RHS work.

## 9. Per-group stiffness routing against a pinned solver

Measured 2026-09-12, CPU (`JAX_PLATFORMS=cpu`), x64, compile cache off, load
10.7–13.6. Median of repeats, `[min–max]` beside it. Three composites: a stiff
group alone, a non-stiff group alone, and a mixed pair the timescale clustering
splits into two groups. `rel err` is against a reference at tighter tolerance.

| composite | arm | solver(s) chosen | warm | steps (rejected) | rel err |
|---|---|---|---|---|---|
| **stiff** k=1e4, 1 group | `auto_stiffness=True` | Kvaerno5 | **194.2 ms** | 122 (20) | 3.74e-05 |
| | `auto_stiffness=False` | Tsit5 | 2 694.2 ms | 145 083 (31 293) | 2.82e-06 |
| | pinned Tsit5 | Tsit5 | 2 719.3 ms | 145 083 (31 293) | 2.82e-06 |
| | pinned Kvaerno5 | Kvaerno5 | 152.1 ms | 122 (20) | 3.74e-05 |
| **non-stiff** k=1, 1 group | `auto_stiffness=True` | Tsit5 | **6.4 ms** | 175 (26) | 1.96e-06 |
| | pinned Tsit5 | Tsit5 | 5.7 ms | 175 (26) | 1.96e-06 |
| | pinned Kvaerno5 | Kvaerno5 | 138.9 ms | 114 (17) | 3.38e-05 |
| **mixed** k=1e4, 2 auto groups | `auto_stiffness=True` | Kvaerno5 + **Tsit5** | **345.4 ms** | 3 974 (653) | **2.23e-08** |
| | `auto_stiffness=False` | Tsit5 + Tsit5 | 1 572.5 ms | 146 637 (31 194) | 1.01e-06 |
| | pinned Tsit5 | Tsit5 + Tsit5 | 1 536.1 ms | 146 637 (31 194) | 1.01e-06 |
| | pinned Kvaerno5 | Kvaerno5 + Kvaerno5 | 671.1 ms | 3 973 (653) | 6.07e-07 |

**On the mixed composite the split lane wins on both axes at once** — the only
row where that happens:

| against | speed | accuracy |
|---|---|---|
| all-implicit (pinned Kvaerno5) | **1.9× faster** | **27× more accurate** |
| all-explicit (pinned Tsit5) | **4.6× faster** | **45× more accurate** |

**What a wrong pin costs, both directions:**

| mistake | wall clock | solver steps |
|---|---|---|
| explicit on a stiff group | **13.9×** (2 694 ms vs 194 ms) | **1 189×** (145 083 vs 122) |
| implicit on a non-stiff group | **21.7×** (138.9 ms vs 6.4 ms) | 0.65× (114 vs 175) |

Note the explicit-on-stiff row is *more accurate* (2.8e-06 against 3.7e-05):
the stability-limited step over-resolves. Accuracy is not the symptom of a
wrong solver choice — wall clock and step count are.

This is the per-group routing half of what §7 measures for the orchestration
as a whole, and it pays on the shape it was built for.

## Reproducing

```bash
HALLSIM_COMPILATION_CACHE_DIR=off python scripts/bench.py            # items 1-3
HALLSIM_COMPILATION_CACHE_DIR=off python scripts/bench.py --solver   # item 4 (slow: ~2 min)
HALLSIM_COMPILATION_CACHE_DIR=off python scripts/bench.py --graph    # item 5
HALLSIM_COMPILATION_CACHE_DIR=off python scripts/bench_scheduler.py chain:256 --macro-dt 0.25   # item 7, one case
HALLSIM_COMPILATION_CACHE_DIR=off python scripts/bench_field.py       # item 8
```

Disable the compile cache when timing, or the second run of a pair is a cache
hit and the comparison is meaningless.
