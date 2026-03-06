# Design: Multi-Timescale Scheduler & Hybrid Discrete-Continuous Architecture

**Status:** Proposed
**Date:** 2026-03-05
**Scope:** Replaces monolithic `Simulator` with a multi-rate `Scheduler`; extends `Process` to support discrete and event-driven processes.

---

## Motivation

Aging biology spans timescales from seconds (ROS kinetics, protein folding) to years (epigenetic drift, telomere shortening), and involves fundamentally discrete events (cell division, apoptosis, mutation acquisition, senescence entry). The current architecture forces everything into a single continuous ODE solve:

```
Process.derivative(t, state) -> dy/dt   # one Diffrax solve, one adaptive timestep
```

This creates two problems:

1. **Stiffness waste** — the adaptive solver slaves to the fastest process. Slow processes (epigenetic drift) get evaluated at every tiny timestep dictated by fast kinetics, wasting compute.
2. **Discrete events are impossible** — cell division, threshold-triggered state transitions, and other discontinuities cannot be expressed as smooth derivatives.

## Prior Art & Attribution

This design borrows explicitly from two systems:

**Vivarium** (Agmon et al., 2022) — HallSim already uses Vivarium's composition semantics (Process, Port, Topology, Store). Vivarium's Engine is a discrete-event simulation engine where each Process declares its own timestep and the engine orchestrates multi-rate stepping. We borrow this scheduling concept.

- Vivarium repo: https://github.com/vivarium-collective/vivarium-core
- Paper: https://doi.org/10.1093/bioinformatics/btac049

**Ptolemy II** (UC Berkeley) — Ptolemy's "Director" abstraction defines *how* processes execute (continuous-time, discrete-event, dataflow, etc.). Directors can be composed hierarchically: a continuous-time director can contain a discrete-event sub-director. We borrow the concept of heterogeneous models of computation under a single orchestrator, though we implement a simpler flat scheduler rather than Ptolemy's full hierarchical director stack.

- Ptolemy II: http://ptolemy.berkeley.edu/ptolemyII/

**What HallSim adds on top:**

| Capability | Vivarium | Ptolemy II | HallSim |
|------------|----------|------------|---------|
| GPU-accelerated continuous solves | No (CPython) | No (Java) | Yes (JAX/Diffrax) |
| Differentiability through ODE solves | No | No | Yes (`jax.grad`) |
| Population parallelism | multiprocessing | Threads | `jax.vmap` |
| Composition-time validation | Basic | No | 4-subsystem semantic layer |
| Heterogeneous process types | Yes (Engine) | Yes (Directors) | Yes (Scheduler) |

HallSim's value proposition is **Vivarium's composition semantics + Ptolemy's heterogeneous execution model, implemented natively on JAX for performance and differentiability.**

---

## Design

### 1. Process Taxonomy

Extend `Process` with a `kind` field. Current processes are implicitly `CONTINUOUS` and require no changes.

```python
class ProcessKind(Enum):
    CONTINUOUS = "continuous"   # derivative(t, state) -> dy/dt
    DISCRETE = "discrete"      # update(t, state) -> delta_state, called at intervals
    EVENT = "event"            # condition + handler, fires on threshold crossing
```

#### Continuous (existing behavior, unchanged)

```python
class ROSProduction(Process):
    kind = ProcessKind.CONTINUOUS
    timescale: float = 1.0          # characteristic time in seconds

    def ports_schema(self):
        return {"ros": Port(role=PortRole.EVOLVED, default=0.0, units="uM")}

    def derivative(self, t, state):
        return {"ros": self.rate * state["ros"]}
```

#### Discrete

Called every `dt_step` seconds. Returns a **delta** (additive), consistent with how kicks and EVOLVED ports already work.

```python
class CellDivision(Process):
    kind = ProcessKind.DISCRETE
    dt_step: float = 86400.0        # evaluate once per day

    def ports_schema(self):
        return {
            "cell_count": Port(role=PortRole.EXCLUSIVE, default=1.0, units="cells"),
            "damage": Port(role=PortRole.INPUT, default=0.0, units="dimensionless"),
        }

    def update(self, t, state):
        """Returns delta to add to current state."""
        can_divide = state["damage"] < 0.8
        return {"cell_count": jnp.where(can_divide, state["cell_count"], 0.0)}
```

#### Event

Fires once when `condition` crosses from False to True. Handler returns a delta.

```python
class SenescenceEntry(Process):
    kind = ProcessKind.EVENT

    def ports_schema(self):
        return {
            "p53": Port(role=PortRole.INPUT, default=0.0, units="uM"),
            "senescent": Port(role=PortRole.LATCHED, default=0.0, units="dimensionless"),
        }

    def condition(self, t, state):
        """Returns True when event should fire."""
        return state["p53"] > 0.9

    def handler(self, t, state):
        """Returns delta. Called once on crossing."""
        return {"senescent": 1.0 - state["senescent"]}
```

### 2. New Port Role: LATCHED

One new port role for values set by discrete/event processes and read by continuous processes as constants within a macro step.

```python
class PortRole(Enum):
    INPUT = "input"          # read-only (unchanged)
    EVOLVED = "evolved"      # additive continuous derivatives (unchanged)
    EXCLUSIVE = "exclusive"  # sole-writer continuous derivatives (unchanged)
    LATCHED = "latched"      # written by discrete/event processes,
                             # read as constant by continuous processes
                             # within a macro step
```

**Semantics:**
- A continuous process may read a LATCHED port (it sees the current value, constant within the macro step).
- Only discrete or event processes may write to a LATCHED port.
- Multiple discrete processes writing to the same LATCHED path follow the same additive semantics as EVOLVED (deltas are summed).
- Validation enforces: no continuous process declares a LATCHED port as EVOLVED or EXCLUSIVE.

### 3. Timescale Declaration & Auto-Grouping

Each continuous process declares a characteristic `timescale` (in seconds). The scheduler uses this to partition processes into groups.

```python
class FastKinetics(Process):
    kind = ProcessKind.CONTINUOUS
    timescale: float = 1.0            # seconds

class SlowDrift(Process):
    kind = ProcessKind.CONTINUOUS
    timescale: float = 86400.0 * 30   # ~month
```

**Auto-grouping heuristic:** Processes within ~2 orders of magnitude (100x) of each other share a group. The user can override with explicit group assignments:

```python
scheduler = Scheduler(
    composite,
    groups={                          # manual override
        "fast": ["ros_prod", "antioxidant"],
        "slow": ["epigenetic_drift", "telomere_loss"],
    },
)
```

If no groups are specified and no timescales are declared, all continuous processes go into a single group (equivalent to current behavior).

### 4. Scheduler Architecture

The `Scheduler` replaces `Simulator` as the top-level execution engine.

```
                        Scheduler
                           |
              macro_dt (communication interval)
                           |
         +-----------------+-----------------+
         |                 |                 |
    Continuous         Discrete          Event
    Groups             Processes         Processes
         |                 |                 |
    +---------+       called at          conditions
    |         |       their dt_step      checked at
  Fast      Slow                         sync points
  Group     Group
    |         |
  Diffrax   Diffrax
  solve     solve
  (JIT)     (JIT)
```

#### Macro Step Loop

```python
def run(self, composite, t_span, macro_dt, y0=None):
    t = t_span[0]
    state = y0 or composite.initial_state()

    while t < t_span[1]:
        t_next = min(t + macro_dt, t_span[1])

        # 1. Solve each continuous group independently
        for group_name, group_procs in self.continuous_groups.items():
            group_rhs = composite.build_group_rhs(group_procs)
            group_result = self._solve_continuous(group_rhs, state, (t, t_next))
            state = merge_continuous_result(state, group_result)

        # 2. Fire discrete processes that are due
        for proc_name, proc in self.discrete_processes.items():
            if is_due(t, t_next, proc.dt_step):
                view = extract_port_view(state, topology[proc_name])
                delta = proc.update(t_next, view)
                state = apply_delta(state, delta, topology[proc_name])

        # 3. Check event conditions
        for proc_name, proc in self.event_processes.items():
            view = extract_port_view(state, topology[proc_name])
            if proc.condition(t_next, view) and not self._was_active[proc_name]:
                delta = proc.handler(t_next, view)
                state = apply_delta(state, delta, topology[proc_name])
                self._was_active[proc_name] = True
            elif not proc.condition(t_next, view):
                self._was_active[proc_name] = False

        t = t_next

    return SchedulerResult(...)
```

#### Operator Splitting

Within a macro step, continuous groups are solved sequentially using **Lie splitting** (solve group A, then group B with updated state). This is first-order accurate in the splitting error.

Future option: **Strang splitting** (solve A for half-step, B for full step, A for half-step) for second-order accuracy. Not needed initially.

#### Choosing macro_dt

`macro_dt` is the synchronization interval — how often continuous groups exchange state and discrete/event processes run. Guidelines:

- Must be <= the smallest `dt_step` of any discrete process
- Should be small enough that LATCHED values don't go stale in ways that matter
- Should be large enough that the scheduling overhead is negligible vs compute
- Default heuristic: `macro_dt = min(discrete dt_steps) / 2`

### 5. Composite Extensions

`Composite` gains methods to support the scheduler:

```python
class Composite(eqx.Module):
    # ... existing fields ...

    def continuous_processes(self) -> dict[str, Process]:
        """All CONTINUOUS kind processes."""

    def discrete_processes(self) -> dict[str, Process]:
        """All DISCRETE kind processes."""

    def event_processes(self) -> dict[str, Process]:
        """All EVENT kind processes."""

    def build_group_rhs(self, proc_names: list[str]):
        """Build a JAX-compatible RHS for a subset of continuous processes.
        Only these processes contribute derivatives; other store paths
        get zero derivatives."""

    def auto_groups(self, max_ratio: float = 100.0) -> dict[str, list[str]]:
        """Partition continuous processes into timescale groups.
        Processes within max_ratio of each other share a group."""
```

`build_group_rhs` is the key new method. It works exactly like current `build_rhs()` but only iterates over the named subset of processes. Store paths owned by other groups get zero derivatives within this solve.

### 6. Validation Extensions

New checks for the multi-timescale architecture:

| Check | Category | Severity |
|-------|----------|----------|
| Continuous process writes to LATCHED port | structure | ERROR |
| Discrete/event process writes to EVOLVED port | structure | ERROR |
| LATCHED port has no discrete/event writer | graph | WARNING |
| Timescale ratio within group > 100x | timescale | WARNING |
| `macro_dt` > smallest discrete `dt_step` | timescale | ERROR |
| Event condition references EVOLVED port (noisy) | graph | WARNING |
| Discrete `dt_step` not aligned with `macro_dt` | timescale | WARNING |

### 7. JAX Boundary

| Layer | JAX? | Differentiable? | Why |
|-------|------|-----------------|-----|
| Continuous group ODE solve | Yes (Diffrax, JIT) | Yes | This is where the FLOPs are |
| Discrete `update()` function | Yes (JAX arrays) | Through smooth relaxations only | Function body is JAX, but calling schedule is Python |
| Event `condition()` / `handler()` | Yes (JAX arrays) | No (discontinuous) | Physically correct: can't differentiate through cell division |
| Scheduler macro loop | No (Python) | No | Coordination logic, not compute |
| vmap over cell populations | Yes | Yes (per-group) | Key scaling mechanism for multi-cell |

**End-to-end differentiability is preserved within each continuous group.** This enables:
- Gradient-based parameter calibration within a timescale
- Sensitivity analysis of continuous dynamics
- Neural ODE training for unknown continuous processes

Gradients do NOT flow across discrete events or between timescale groups. This is a feature, not a limitation — these boundaries represent genuine discontinuities in the biology.

### 8. Backward Compatibility

A composite with only CONTINUOUS processes and no timescale declarations degrades exactly to current behavior:

```python
# This still works identically:
composite = Composite(
    processes={"decay": Decay(), "growth": Growth()},
    topology={"decay": {"x": "pool/x"}, "growth": {"x": "pool/x"}},
)

# Scheduler with a single group = current Simulator behavior
scheduler = Scheduler()
result = scheduler.run(composite, t_span=(0.0, 100.0), macro_dt=1.0)
```

The existing `Simulator` class can remain as a convenience wrapper that delegates to `Scheduler` with a single group.

### 9. SimResult Extension

```python
@dataclass
class SchedulerResult:
    ts: jnp.ndarray                        # macro step times
    ys: dict[str, jnp.ndarray]             # state trajectories
    events: list[EventRecord]              # fired events log
    stats: dict[str, Any]                  # per-group solver stats

@dataclass
class EventRecord:
    time: float
    process: str
    delta: dict[str, jnp.ndarray]
```

---

## Example: Multi-Timescale Aging Cell

Putting it all together — a single cell with fast ROS kinetics, slow epigenetic drift, discrete division, and event-driven senescence:

```python
# Fast continuous: ROS production + antioxidant defense (~seconds)
ros_prod = ROSProduction(rate=0.1, timescale=1.0)
antioxidant = AntioxidantDefense(rate=0.08, timescale=1.0)

# Slow continuous: epigenetic drift (~months)
epi_drift = EpigeneticDrift(rate=1e-7, timescale=86400 * 30)

# Discrete: cell division check (once per day)
division = CellDivision(dt_step=86400.0)

# Event: senescence entry (threshold on p53)
senescence = SenescenceEntry()

composite = Composite(
    processes={
        "ros_prod": ros_prod,
        "antioxidant": antioxidant,
        "epi_drift": epi_drift,
        "division": division,
        "senescence": senescence,
    },
    topology={
        "ros_prod":    {"ros": "cell/ROS"},
        "antioxidant": {"ros": "cell/ROS"},
        "epi_drift":   {"methylation": "cell/methylation", "ros": "cell/ROS"},
        "division":    {"cell_count": "pop/count", "damage": "cell/methylation"},
        "senescence":  {"p53": "cell/p53", "senescent": "cell/senescent"},
    },
    semantic_validation=True,
)

scheduler = Scheduler()
result = scheduler.run(
    composite,
    t_span=(0.0, 86400 * 365),   # one year
    macro_dt=3600.0,              # sync every hour
)
```

The scheduler will:
1. Auto-group `ros_prod` + `antioxidant` (fast) and `epi_drift` (slow)
2. Solve the fast group with small adaptive steps within each macro step
3. Solve the slow group with large adaptive steps
4. Check `division.update()` every 86400s (once per day)
5. Check `senescence.condition()` every macro step (hourly)
6. Sync all state through the shared store at each macro step boundary

---

## Implementation Plan

### Phase 1: Foundation — DONE
- [x] Add `ProcessKind` enum and `kind` field to `Process` (default: CONTINUOUS)
- [x] Add `timescale` field to `Process` (default: None)
- [x] Add `LATCHED` to `PortRole`
- [x] Add `update()` and `condition()`/`handler()` stub methods to `Process`
- [x] Extend topology validation for new port roles

### Phase 2: Composite Extensions — DONE
- [x] `Composite.continuous_processes()`, `discrete_processes()`, `event_processes()`
- [x] `Composite.build_group_rhs(proc_names)`
- [x] `Composite.auto_groups(max_ratio)`
- [x] Validation extensions (LATCHED/EVOLVED conflicts, timescale checks)

### Phase 3: Scheduler — DONE
- [x] `Scheduler` class with macro step loop
- [x] Continuous group solving (Lie splitting)
- [x] Discrete process dispatch
- [x] Event detection and handling
- [x] `SchedulerResult` with event log
- [x] Backward-compatible: single-group = current Simulator

### Phase 4: Validation & Testing — DONE
- [x] New validation checks (kind/role compatibility, LATCHED conflicts, dt_step)
- [x] Unit tests for each process kind (39 tests)
- [x] Integration test: continuous + discrete + event together
- [x] Backward compatibility tests (all 82 existing tests still pass)
- [x] Demo CLI commands

### Phase 5: Documentation & Examples — DONE
- [x] Update README (roadmap, architecture, API examples, test count)
- [x] CLI demo: `simulate multiscale`
- [x] Example processes for each kind (in test_multiscale.py)

---

## Open Questions (Deferred)

1. **Strang splitting** — upgrade to second-order splitting if Lie proves too inaccurate. Measure first.
2. **Diffrax event API** — for precise within-solve event detection (more accurate than macro-step checking). Adds complexity; defer until macro-step granularity proves insufficient.
3. **Multi-cell / spatial** — vmap the scheduler over cell populations. Needs design for inter-cell communication (paracrine signaling, diffusion). Separate design doc.
4. **Stochastic processes** — Gillespie-type discrete stochastic events. Natural extension of DISCRETE kind with stochastic firing. Defer.
5. **SBML import** — auto-detect timescales and process kinds from SBML model metadata. Defer.
