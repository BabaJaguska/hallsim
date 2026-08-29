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
| Sharding the *existing vmapped* batch axis | 0.83× — slower than doing nothing. One vmapped `while_loop` has one trip-count predicate, so SPMD adds a cross-device reduce instead of splitting the loop. Needs `shard_map` (2.4×) |
| Parallelising forward-mode parameter directions | 1.09×. The vmapped JVP already shares one primal solve |

**A caveat on every absolute number above.** Re-running the same measurement
hours later on this machine gave warm 3.2 s against 2.25 s and first-call ~14 s
against 9.1 s — ~40% drift, no code change. Ratios held. Trust a *difference*
only when its arms were interleaved serially in one session on an idle machine;
treat a *level* as needing a re-measure.

---

## Reproducing

```bash
HALLSIM_COMPILATION_CACHE_DIR=off python scripts/bench.py            # items 1-3
HALLSIM_COMPILATION_CACHE_DIR=off python scripts/bench.py --solver   # item 4 (slow: ~2 min)
HALLSIM_COMPILATION_CACHE_DIR=off python scripts/bench.py --graph    # item 5
```

Disable the compile cache when timing, or the second run of a pair is a cache
hit and the comparison is meaningless.
