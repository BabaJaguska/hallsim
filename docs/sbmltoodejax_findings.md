# sbmltoodejax findings (for an upstream PR / issue tracker)

Limitations and bugs in [`sbmltoodejax`](https://developmentalsystems.org/sbmltoodejax/)
that HallSim has hit and worked around. Collected so an upstream contribution
is one writeup away. Each entry: **symptom → cause → HallSim workaround → upstream
fix sketch.** Workarounds live in
[`src/hallsim/sbml_import.py`](../src/hallsim/sbml_import.py) (source-patching the
generated module in `_generate_sbml_model`, plus `_preprocess_sbml`).

---

## 1. Hardcoded `float32` defeats `jax_enable_x64` — silent precision floor (HIGH)

**Symptom.** A model imported under `jax.config.update("jax_enable_x64", True)`
still computes its RHS at float32 precision (~1e-7). With an adaptive
step-size controller at `rtol=1e-6` (≤ float32 machine epsilon), the embedded
error estimate is dominated by float32 roundoff, so the controller can never
meet tolerance: **~50–57% step rejection, `max_steps` blowups**, and NaN
sensitivities — *masquerading as stiffness*. Explicit solvers limp through the
primal; implicit solvers (Kvaerno/KenCarp) reject ~half their steps regardless
of root finder, atol, scaling, or order. We spent hours chasing "stiffness"
before finding it was precision.

Concretely, the generated `RateofSpeciesChange.__call__` emits:
```python
rateRuleVector = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector
```
and the `stoichiometricMatrix` / `c` / `w0` / `y0` are float32 too. An *explicit*
`dtype=jnp.float32` overrides the global x64 flag.

**Cause.** `modulegeneration.py` hardcodes `dtype=jnp.float32` for generated
constants and rate vectors instead of respecting the active JAX default dtype.

**HallSim workaround.** Source-patch `float32` → `float64` in the generated
module before import (`sbml_import.py`, in the generation-patch block). Verified:
DallePezze 2014 (BIOMD0000000582) under `Kvaerno5(root_finder=optx.Newton)` goes
from 80000 steps / 57% rejection (max_steps, no gradient) to **112 steps / 5%
rejection with a finite, forward==reverse-consistent gradient**. (No-op when x64
is off — JAX downcasts float64 → float32.)

**Upstream fix.** Emit no explicit dtype (let JAX pick per the x64 flag), or
honor `jax.numpy.float_`/the canonical dtype, rather than literal `jnp.float32`.

---

## 2. MathML namespace prefix `no` emitted as undefined name (MEDIUM)

**Symptom.** Generated code contains bare `no.sqrt(...)` (or similar) — an
`ImportError`/`NameError` for an undefined `no`. Seen on Sivakumar 2011 crosstalk
(BIOMD0000000398).

**Cause.** A MathML namespace prefix `no` is emitted verbatim instead of being
mapped to `jax.numpy`.

**HallSim workaround.** Patch `import no` → `import jax.numpy as no` (or prepend
the alias) in the generated source.

**Upstream fix.** Resolve MathML namespace prefixes to the numerics module during
codegen.

---

## 3. `eqx.static_field()` — removed Equinox API (LOW)

**Symptom.** Generated code calls `eqx.static_field()`, removed in modern Equinox.

**HallSim workaround.** Patch `eqx.static_field()` → `eqx.field(static=True)`.

**Upstream fix.** Update codegen to the current Equinox field API.

---

## 4. Custom `<functionDefinition>`s rejected (MEDIUM)

**Symptom.** `NotImplementedError: Custom functions are not handled` in
`ParseRHS` for any model using SBML function definitions (e.g. DallePezze 2014).

**Cause.** `ParseRHS` does not inline user-defined functions.

**HallSim workaround.** Run libsbml `expandFunctionDefinitions` first
(`_preprocess_sbml`), which inlines every `<functionDefinition>` body at its call
sites, then hand the flattened SBML to sbmltoodejax.

**Upstream fix.** Either inline function definitions during parse, or document the
required libsbml pre-pass.

---

## 5. Missing math functions in the `mathFuncs` whitelist (MEDIUM)

**Symptom.** Models using e.g. `log10()` fail to translate (blocks Konrath 2020,
MODEL2004300002).

**Cause.** `modulegeneration.mathFuncs` whitelist omits some basic math
functions.

**HallSim workaround.** Pre-flight check (`sbml_import._precheck`) surfaces the
offending function names with a clear error rather than a deep stack trace.

**Upstream fix.** Extend the `mathFuncs` table (`log10` → `jnp.log10`, etc.).

---

## 6. Other gaps (LOW / documentation)

- **MIRIAM species ontology annotations dropped** — HallSim re-extracts them from
  the SBML via libsbml for semantic validation.
- **`<event>` unsupported** — `NotImplementedError` (blocks Proctor 2008
  BIOMD0000000188); a roadmap item is an SBML-event → `ProcessKind.EVENT`
  translator.
- **Boundary species buried in `w`** — not surfaced as settable inputs; HallSim
  exposes them through its `parameters` surface (`_collect_boundary_inputs`).
- **Inert-sink species** (write-only degradation collectors, e.g. a `Nil` species)
  are integrated as ordinary states rather than treated as boundary; HallSim
  detects and freezes them (`_detect_inert_sinks`).
