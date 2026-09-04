# Design — the spontaneous-endpoint screen

Planned, not built. This is the implementable specification; a working
reference implementation exists in the SASP-branch bundle
(`hallsim_sasp_branch/hallsim_sasp_branch.diff`, `diagnostics.py` hunk, with
`untracked/tests/unit/test_spontaneous_endpoint.py`) and should be read
alongside this before starting.

## What it is for

Every model in a perturbation study carries an implicit claim: that the
contrast between the perturbed arm and the control is caused by the
perturbation. Nothing in the framework checks it. DallePezze 2014 fails it
completely — dosed and undosed converge to the same attractor
(‖Δ‖ = 1.8×10⁻⁵ at day 400, identical to six significant figures across six
decades of dose), so its "control" is the same trajectory arriving slightly
later. That is [known-problems.md](known-problems.md) P0.14, and it was found
only after a composite had been built on it, calibrated, and scored.

The screen asks one question at import time: **of the motion the perturbed arm
made, what share does the perturbation explain?** DallePezze returns ~0. It
would have returned ~0 in 2014, before any composite existed.

This belongs in the framework, not in a demo or a document. It has to fire for
the next model as well, whoever imports it.

## The metric

Run the composite as configured, and a control copy with every detectable
perturbation source zeroed. Compare per state, normalised by that state's peak
excursion in the perturbed run so the ratio is unit-free and no state
dominates by having larger units:

```
fraction      = RMS(ŷ_pert[−1] − ŷ_ctrl[−1]) / RMS(ŷ_pert[−1] − ŷ₀)
fraction_path = RMS(ŷ_pert    − ŷ_ctrl   ) / RMS(ŷ_pert    − ŷ₀)
```

where `ŷ` is per-state normalised. `spontaneous` fires when
`fraction < threshold`.

**Both numbers, not one.** A model can be legitimately responsive over the data
window (`fraction_path` high) while its long-run endpoint is spontaneous. That
is DallePezze exactly, and it is why the defect survived peer review: the fit
is to the transient, and the transient is real. Reporting only the endpoint
would call a good pulse-response model broken; reporting only the path would
have passed DallePezze.

**The endpoint is the one that decides**, because the endpoint is what a demo
reads off and what a fold-change contrast is computed from.

### Degenerate cases, and what they must return

| situation | `fraction` | verdict |
|---|---|---|
| both arms return to y₀ (healthy homeostatic pulse response) | NaN | not spontaneous; the denominator is the degeneracy, not the model |
| run does not move at all | NaN | vacuous |
| no perturbation source detected | — | `n_sources == 0` → **VACUOUS**, never "passed" |

A NaN is not a pass. The report has to say which of these it is, and
`screen_composite` must not fold a NaN into an `ok` flag.

## Source detection — the part that generalises

Zeroing "the perturbation" requires knowing what the perturbation is, without
being told. Two source kinds:

1. **`PulseSource` processes** — set amplitude to 0. The process stays wired
   and keeps its topology; it emits zero. Nothing about the composite's shape
   changes, so the control run traces identically.

2. **SBML boundary inputs whose native rule varies over the run window.**
   Extracted at import as `time_varying_rules` (a rule whose MathML references
   `AST_NAME_TIME`), then overridden to 0 by routing the input to an unwired
   INPUT port via `with_input_driver`.

**The trap that makes the syntactic test alone useless.** COPASI and
PottersWheel exports wrap *constants* in time-referencing piecewises.
DallePezze's `Insulin` is `piecewise(1, time < −1, 1, time < 0, 1)` — constant
1 over any real time span, but syntactically time-varying. Zeroing it builds a
starvation model and calls it a control, which is a worse error than the one
being screened for.

So `time_varying_rules` is a *candidate* list. Each candidate is then evaluated
through the model's own assignment function at n points across the actual
`[t0, t1]` of this run, and zeroed only if its value actually changes:

```
span = max(v) − min(v)  over n samples;  vary if span > rtol · max(|v|, 1)
```

A pulse that fires entirely outside the run window reads as constant, and is
correctly left alone — it does not perturb *this* run.

That two-stage test is the difference between a check that works on one model
and one that works on the corpus. Build it that way from the start.

**Already externally driven inputs are left alone** — their driver process is
itself a source and gets zeroed on its own. Zeroing both double-counts.

**Excluded from the comparison:** a zeroed source's own store paths. The drive
signal *is* the perturbation, so comparing it counts the input as output.

## Report

```python
@dataclass
class SpontaneousReport:
    n_sources: int
    zeroed: tuple[str, ...]        # human-readable, one per source
    motion: float                  # RMS(ŷ_pert[−1] − ŷ₀), the denominator
    divergence: float              # RMS(ŷ_pert[−1] − ŷ_ctrl[−1])
    fraction: float
    fraction_path: float
    threshold: float
    spontaneous: bool
    detail: str = ""

    @property
    def ok(self) -> bool: ...
```

`__str__` must distinguish VACUOUS from ok from SPONTANEOUS-ENDPOINT in its
first token — this string is what lands in a triage table.

## Integration points

| surface | change |
|---|---|
| `hallsim.diagnostics.spontaneous_endpoint(composite, t_span, ...)` | the function |
| `diagnostics.screen_composite` | run it alongside the existing screens; NaN must not read as pass |
| `intake.triage_process` / `triage_sbml` | report it in the verdict, so a single-model import is gated before composition |
| `sbml_import._extract_coupling_metadata` | emit `time_varying_rules` (the `AST_NAME_TIME` walk) |

The `_extract_coupling_metadata` change is shared with the rate-rule fix
already applied — same function, same walk pattern.

## Advisory, not a gate

A legitimate relaxation experiment flags here by design: washout recovery, a
fitted non-rest initial condition, any protocol whose interesting motion is the
system returning to baseline. The correct response is to state that, not to
suppress the screen. It reports a property of the *design*, and only the person
who wrote the design knows whether that property is intended.

## Tests

1. **DallePezze 2014 regression** — the load-bearing case. `fraction` ≈ 0 at a
   day-400 horizon (arms differ by ~1e-9 there). This is the test that would
   have caught P0.14.
2. **Synthetic positive control** — `dx/dt = u·(1 − x)` from `x(0) = 0`: a
   sustained drive moves the state for good, so `fraction` ≈ 1.
3. **Synthetic pulse response** — a damped return to y₀ in both arms:
   `fraction` is NaN and `spontaneous` is False.
4. **Vacuous case** — a composite with no source: `n_sources == 0`.
5. **The COPASI-piecewise trap** — DallePezze's `Insulin` must NOT be zeroed
   while `Irradiation` must be. Assert on `zeroed` by name. Without this test
   the two-stage detection will be "simplified" back to the syntactic check by
   the next person who reads it.

## Known limits, to be stated wherever the screen is reported

- Only the two source kinds above are recognised. A perturbation expressed as a
  **parameter change between arms** is invisible — and that is exactly what
  DallePezze needs as a control per critique §9 (`AMPK_T172_phos × 10`). A
  model whose arms differ by a parameter returns `n_sources == 0`.
- The screen measures whether the perturbation explains the endpoint. It does
  not measure whether the endpoint is *right*.
- Threshold choice is a convention. Report the number, not only the verdict.

## Estimate

~300 lines in `diagnostics.py` plus the metadata walk in `sbml_import.py`, and
the five tests above. The reference implementation in the SASP-branch bundle
covers the function and its report; the triage/screen integration and tests 4
and 5 are new work.
