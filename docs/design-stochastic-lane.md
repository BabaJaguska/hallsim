# A stochastic lane for the Scheduler

Written 2026-09-06, after Hui 2016 (BIOMD0000000560) was rejected as an ODE
import because the deposit is a Gillespie model (P0.57). The rejection is what
makes the case: a deposit whose published results are population statistics
over 500 stochastic runs is not importable at all today, and reading it
deterministically deletes one of its two mechanisms in silence.

Conclusion up front: **most of the machinery exists, the missing piece is a
source of randomness, and the design is a per-group integrator rather than a
new kind of Process.** The measured event budget puts the 14-day composite
window inside a coffee break and the paper's own 30-month horizon overnight.

## What is already built

`sbmltoodejax` generates the reaction stoichiometric matrix and evaluates the
reaction velocity vector inside its ODE RHS, while HallSim originally kept
only the collapsed sum:

```
r.stoichiometricMatrix            # (62, 117)  N
r.calc_reaction_velocities(y,w,c,t)  # (117,)  v
rateOfSpeciesChange = N @ v + rateRuleVector
```

- `N` is already surfaced — `SBMLProcess.stoichiometry()`
  (`src/hallsim/sbml_import.py:288`), used for conserved moieties.
- `v` is generated and discarded. `SBMLProcess.derivative`
  (`sbml_import.py:466`) calls `ratefunc`, which forms `N @ v` internally.
- The Scheduler already routes each timescale group to its own solver through
  `GroupIntegrator` (`scheduler.py:102`), and already has a non-Diffrax lane
  for DISCRETE processes fired on a `dt_step` grid (`scheduler.py:1191`).

HallSim now preserves source reaction IDs and rate-law strings and exposes an
executable `reaction_propensities(t, state)` view when every reaction law can
be translated. Models with unsupported custom functions still import
deterministically, but their stochastic view raises explicitly when used.
`simulate_ssa` is currently a direct-method single-process runner; the
Scheduler now also recognizes an explicitly selected
`SBMLProcess.as_stochastic()` in an eager single-stochastic-process lane.
The direct runner remains useful for one-way hybrid inputs; batched stochastic
groups, multiple stochastic processes, and fully coupled hybrid splitting
remain unfinished.

## The gap

**Nothing in the solve path has a PRNG.** `Scheduler.run` (`scheduler.py:591`)
takes no key, no process is handed one, and `SchedulerResult` has no replicate
axis. `jax.random` appears only in `steady_state.py` (basin sampling) and
`neuralode.py` (init and batching). That is the one structural change; the rest
is assembly.

## Design: a stochastic group integrator

Not a fourth `ProcessKind`. The Scheduler already splits a composite into
timescale groups and gives each its own stepper; a group whose members declare
reaction channels gets an SSA stepper instead of `diffeqsolve`. Three
consequences follow for free:

- Grouping, macro-step splitting, `save_dt` anti-aliasing and the coupling mode
  are inherited rather than reimplemented.
- **The coupling approximation is the one the splitting already makes.** An SSA
  member reading a continuous input needs that input held constant between
  jumps; Lie/Strang splitting already freezes non-own states across a macro
  window. No new approximation is introduced, and `macro_dt` is already the
  knob that controls its error.
- A stochastic member and an ODE member sit in one composite on one clock,
  which is the framework claim rather than a side feature.

### Surface

```python
class Process:
    def reaction_channels(self) -> dict | None:
        """``{"stoichiometry": N, "propensities": fn(t, y) -> a}``.
        ``None`` (the default) means the process is deterministic."""
```

Sibling to `stoichiometry()`, same convention: `None` is undeclared, not
"no channels". `SBMLProcess` implements it by returning `N` and
`calc_reaction_velocities`; every hand-written Process keeps returning `None`
and nothing about existing composites changes.

### PRNG threading

`Scheduler.run(..., key=None)`. A key splits once per macro step per stochastic
group, so the stream is a pure function of `(key, step index, group name)` —
reproducible under `jit`, and under `vmap` the replicate axis is the key axis,
which is what a population run needs anyway.

## The propensity gate

The one place this corrupts results silently. **A rate law is not a
propensity.** Hui's deposit writes

```
Alk5Dimerisation :: kdimerAlk5 * Alk5 * (Alk5 - 1) * 0.5
```

— the combinatorial form. Its rate laws already *are* propensities, because the
model was written for Gillespie. A concentration model's are not: converting
needs `a_j = v_j · V · N_A` plus the combinatorial correction on every
higher-order channel, and getting it wrong produces plausible trajectories.

So the SSA lane **refuses** any process that fails the P0.57 stochastic-intent
check (`substance` unit `item`, `hasOnlySubstanceUnits`, integral initial
amounts, unit compartments) unless the caller declares the volume explicitly.
P0.57 stops being a warning and becomes the admission gate.

Read the other way: `Alk5 * (Alk5 - 1) * 0.5` integrated as an ODE is wrong by
`1/Alk5` — 3.3% at the model's own 30.5 molecules. A third independent sign
that this file was never an ODE.

## Composition rules to enforce

- A stochastic group's species must not be EVOLVED by any continuous process.
  A continuous writer adds a non-integral delta to a count, and the state stops
  being a count. Composition-time check, not a runtime one.
- Species counts stay in the float64 store; `y + N[:, j]` preserves
  integrality exactly. A negative count means the propensities are wrong, so
  that is an assertion, never a `maximum(y, 0)` clamp.

## Measured budget

Prototype (`scratch/2026-09-06-ssa-prototype/ssa_probe.py`): direct-method SSA
over the generated propensity vector, `lax.scan` over jumps, `vmap` over cells.
Measured on CPU: 6.6e4 events/s at 1 cell, 3.7e5 at 64, **6.25e5 at 256**.

Hui's own event budget, from `∫a₀ dt` along the deterministic trajectory
(`a₀` = 38.2/s at `y₀`, 3.94/s at 3 months, 2.35/s at 30 months):

| horizon | jumps/cell | 256 cells at 6.25e5 events/s |
|---|---|---|
| paper's 30 months | 2.11e8 | ~18 h (×200 replicates: 4.2e10 events) |
| 14 days from `y₀` | 5.52e6 | ~38 min |
| **14 days from a 3-month state** | **4.67e6** | **~32 min** |

The composite's own window is affordable. Reproducing the paper's figure 4B is
an overnight job.

## Tau-leaping does not rescue the long horizon

Cao–Gillespie–Petzold selection (ε = 0.03, `n_c` = 10) along the same
trajectory: **6.13e7 leap steps plus 6.42e7 exact jumps** for the critical
channels — 3.4× against 2.11e8, not the two orders of magnitude leaping is
worth on a well-mixed high-copy system.

The reason is structural and cannot be tuned away: a median **58 of 117
channels are critical**, because **24 of Hui's 62 dynamic species never exceed
10 molecules** over the whole run (43 never exceed 100). That is the same fact
that makes the mean-field limit invalid. A model that needs SSA is a model
where leaping has nothing to leap over.

Exact direct method first. Leaping is an optimisation to revisit on a model
that earns it.

## What this does not give

- **Calibration gradients.** Nothing differentiates through a categorical
  sample. Fit on the mean field and validate stochastically, or go
  gradient-free for the stochastic member. Not a blocker for import or for the
  population demo; it is a hard boundary for `Calibrator`.
- **Population observables.** The paper's only readout is "% of cells above a
  threshold". `vmap` over keys supplies the axis; the reporter layer has no
  population statistic to put on it (P3.8, P3.9), and
  `demos/composite_population.py` is where it would land.
- **Hybrid jump/continuous members.** Out of scope here, and Hui does not need
  it: 0 rules, 0 events, 0 function definitions, 4 compartments all of size
  1.0. A pure reaction network — 115 of 117 channels mass-action, 2
  Michaelis–Menten, 72 first-order and 45 second-order. Deliberately the
  easiest possible first target.

## Order of work

1. `Process.reaction_channels()` + the `SBMLProcess` implementation.
2. P0.57 as an executable gate in `triage_sbml`, since it admits the model.
3. `key=` through `Scheduler.run` and into group stepping.
4. Direct-method SSA as a `GroupIntegrator` variant.
5. Composition checks (no continuous writer on a count).
6. Population observable + the Hui demo.

Steps 1–3 are independently useful: 1 is a free accessor, 2 closes a filed
defect, and 3 is what every future stochastic or sampling feature needs.
