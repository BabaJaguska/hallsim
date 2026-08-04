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

## 3. Proposed: restrict each group's solve to its own states

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
**Not implemented** — only valid under frozen coupling, and the stiff-group
vector `atol` needs restricting per group too.

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
tolerance is not fighting large-magnitude states. Cause unidentified.

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
remaining plumbing is a real cost only for *tracing*, which item 1 now pays
once per structure rather than once per call.

---

## Reproducing

```bash
python scripts/bench.py            # items 1-3
python scripts/bench.py --solver   # item 4 (slow: ~2 min)
python scripts/bench.py --graph    # item 5
```
