# HallSim: A Modular, Multi-Scale Simulator for Aging Biology
[![Basic CI/CD Workflow](https://github.com/BabaJaguska/HallSim/actions/workflows/basic_CI_linux.yaml/badge.svg)](https://github.com/BabaJaguska/HallSim/actions/workflows/basic_CI_linux.yaml)

## Background & Motivation

Aging is a complex, network-level phenomenon. Its hallmarks — such as mitochondrial dysfunction, genomic instability, and altered intercellular communication — do not act in isolation. Instead, they form dense webs of feedback and crosstalk.

Traditional approaches to modeling aging have been reductionist. Inspired by conceptual frameworks like Cohen et al. (2022) [1], HallSim aims to embrace complex systems theory and explore emergent properties of aging arising from loss of resilience across multiple sub-systems.

Existing bio simulation tools and model repositories provide standardized formats and simulation capabilities for individual dynamical models, but they generally lack support for composable model libraries — i.e., reusable modules with defined high-level interfaces that can be assembled into heterogeneous larger multi-scale dynamical systems.

HallSim's goal is to enable compositional co-simulation of reusable systems-biology modules, with a focus on aging biology.

---

## Project Goals

* Build a modular, multi-scale agent-based simulator
* Allow plug-and-play modules with exposed high level abstractions corresponding to 12 hallmarks of aging [2]
* Incorporate an LLM layer helping integrate new modules in a semi-automated way
* Enable simulation of aging trajectories, interventions, and emergent phenotypes
* Serve as an educational tool
* Somewhere down the road become an in-silico testbed for perturbations (radiation, caloric restriction, therapies, etc.)

---

## Architecture

HallSim has two layers: a **legacy cell-agent system** and a new **composable architecture** built on JAX/Equinox/Diffrax.

### Composable Architecture (v2)

The composable layer reimplements key ideas from Vivarium [3] (ports, topology, stores) natively in JAX for GPU acceleration, differentiability, and explicit composition semantics.

**Core concepts:**

| Concept       | Description |
|---------------|-------------|
| **Process**   | Equinox module (`eqx.Module`) — declares typed ports and a `kind` (CONTINUOUS / DISCRETE / EVENT). Parameters are JAX arrays: differentiable, JIT-compilable, vmappable. |
| **Port**      | Named connection point with a role (`INPUT` / `EVOLVED` / `EXCLUSIVE` / `LATCHED`), default value, units, description, and ontology annotation. |
| **Topology**  | Static wiring map `{proc_name: {port_name: store_path}}`. Defined at composition time, not inside processes. |
| **Composite** | Bundles processes + topology. `build_rhs()` / `build_group_rhs()` return JAX-compatible ODE right-hand sides. Auto-groups continuous processes by timescale. |
| **Simulator** | Diffrax wrapper for single-group continuous solves. Retained for simple cases. |
| **Scheduler** | Multi-rate orchestrator: groups continuous processes by timescale (Lie splitting), dispatches discrete processes at intervals, checks event conditions at sync points. See [design doc](docs/design-multiscale-scheduler.md). |
| **Store**     | Flat `dict[str, jnp.ndarray]` with path-like keys (e.g., `"cytoplasm/ROS"`). A valid JAX PyTree. |

**Process kinds:**

- `CONTINUOUS` (default) — computes `derivative(t, state) -> dy/dt`, solved by Diffrax ODE integrator.
- `DISCRETE` — computes `update(t, state) -> delta`, called every `dt_step` seconds by the Scheduler.
- `EVENT` — declares `condition(t, state) -> bool` and `handler(t, state) -> delta`, fires on False-to-True crossing.

**Port roles:**

- `INPUT` — read-only, process uses this value but doesn't write a derivative.
- `EVOLVED` — additive: multiple processes can contribute derivatives to the same store path (summed).
- `EXCLUSIVE` — sole ownership: only one process may write to this store path (validated at composition time).
- `LATCHED` — written by discrete/event processes, read as constant by continuous processes within a macro step.

**Validation layer** (optional, runs at composition time):

| Subsystem          | What it checks |
|--------------------|----------------|
| **UnitChecker**    | pint-based dimensional analysis across shared store paths |
| **SemanticChecker**| Ontology ID comparison (ChEBI, GO, SBO) for species disambiguation |
| **GraphAnalyzer**  | Feedback cycle detection, fan-in analysis, coupling density |
| **CouplingAuditor**| Heuristic duplicate-reaction detection via description overlap |

Validation is warnings-by-default (not errors). Use `strict=True` to promote warnings to errors.

### Legacy Cell-Agent System

* Each Cell has a `CellState` representing internal biology
* Multiple Submodels (e.g., ERiQ, NeuralODE) plug into a Cell [4]
* Cells reside on a 2D grid (expandable to 3D)
* Models of unknown dynamics can be trained as a NeuralODE using `optax` and `equinox`
* Hallmarks are a high level input, allowing a directional interface through a 0-1 handle

---

## Getting Started

### Install

```bash
make install        # or make install-dev for development
```

### Run demos

```bash
# Composable architecture demos
simulate compose              # ROS production + antioxidant defense ODE
simulate compose-kick          # Same system + mid-run perturbation
simulate validate-demo         # Validation layer catching unit/semantic issues
simulate validate-demo --strict # Strict mode: warnings become errors
simulate info                  # Architecture overview

# Legacy demos
simulate basic                 # Mitochondrial damage simulation
simulate kick                  # Perturbation demo
```

### Run tests

```bash
make test
# or directly:
.venv_hallsim/bin/python -m pytest tests/ -v
```

148 tests: 24 legacy + 31 composition + 27 validation + 66 multi-timescale/scheduler.

### Python API

```python
import jax.numpy as jnp
from hallsim.process import Process, Port, PortRole
from hallsim.composite import Composite
from hallsim.simulator import Simulator

# 1. Define processes
class Decay(Process):
    rate: float = 0.1

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=1.0, units="uM")}

    def derivative(self, t, state):
        return {"x": -self.rate * state["x"]}

class Growth(Process):
    rate: float = 0.05

    def ports_schema(self):
        return {"x": Port(role=PortRole.EVOLVED, default=1.0, units="uM")}

    def derivative(self, t, state):
        return {"x": self.rate * state["x"]}

# 2. Wire via topology
composite = Composite(
    processes={"decay": Decay(), "growth": Growth()},
    topology={"decay": {"x": "pool/x"}, "growth": {"x": "pool/x"}},
    semantic_validation=True,  # optional: run unit/semantic checks
)

# 3. Solve
sim = Simulator()
result = sim.run(composite, t_span=(0.0, 100.0), dt=1.0)
print(result.ts.shape, result.ys["pool/x"].shape)

# 4. Perturbation
result = sim.run_with_perturbation(
    composite, t_span=(0.0, 100.0),
    kick_time=50.0, kick_dict={"pool/x": 5.0},
)
```

### Multi-timescale with Scheduler

```python
from hallsim.process import ProcessKind
from hallsim.scheduler import Scheduler

# Discrete process: fires every dt_step seconds
class CellDivision(Process):
    kind: ProcessKind = ProcessKind.DISCRETE
    dt_step: float = 86400.0  # once per day

    def ports_schema(self):
        return {
            "count": Port(role=PortRole.LATCHED, default=1.0, units="cells"),
            "damage": Port(role=PortRole.INPUT, default=0.0),
        }

    def update(self, t, state):
        can_divide = state["damage"] < 0.8
        return {"count": jnp.where(can_divide, state["count"], 0.0)}

# Event process: fires once when condition crosses True
class SenescenceEntry(Process):
    kind: ProcessKind = ProcessKind.EVENT

    def ports_schema(self):
        return {
            "p53": Port(role=PortRole.INPUT, default=0.0, units="uM"),
            "senescent": Port(role=PortRole.LATCHED, default=0.0),
        }

    def condition(self, t, state):
        return state["p53"] > 0.9

    def handler(self, t, state):
        return {"senescent": 1.0 - state["senescent"]}

# Mix all three kinds in one composite
composite = Composite(
    processes={
        "decay": Decay(), "growth": Growth(),
        "division": CellDivision(), "senescence": SenescenceEntry(),
    },
    topology={
        "decay": {"x": "pool/x"}, "growth": {"x": "pool/x"},
        "division": {"count": "pop/count", "damage": "pool/x"},
        "senescence": {"p53": "pool/x", "senescent": "cell/senescent"},
    },
)

# Scheduler handles multi-rate orchestration
scheduler = Scheduler()
result = scheduler.run(composite, t_span=(0.0, 86400.0), macro_dt=3600.0)
print(result.events)  # log of fired events
```

Parameters are JAX arrays, so you can differentiate through entire simulations:

```python
import jax

def loss(rate):
    proc = Decay(rate=rate)
    comp = Composite(
        processes={"decay": proc},
        topology={"decay": {"x": "pool/x"}},
        validate=False,
    )
    rhs = comp.build_rhs()
    y = {"pool/x": jnp.array(1.0)}
    dy = rhs(0.0, y)
    return dy["pool/x"] ** 2

grad = jax.grad(loss)(0.1)  # d(loss)/d(rate)
```

---

## Dev Instructions

- `pyproject.toml` is the single source of dependencies.
- To add a model, create a new `.py` file in `src/hallsim/models/` (legacy) or define a new `Process` subclass (composable).
- Models are aggregated on an additive basis in the ODE RHS via EVOLVED ports. If your model's effect is supposed to be multiplicative, consider using a separate store path and an INPUT port to read the other variable.

### Key files (composable architecture)

```
src/hallsim/
  process.py      — Process base class, Port, PortRole, ProcessKind
  store.py        — Store utilities (build, extract, route, validate)
  composite.py    — Composite: wires Processes via Topology, auto-grouping
  simulator.py    — Diffrax-based ODE solver wrapper (single-group)
  scheduler.py    — Multi-rate Scheduler (continuous groups + discrete + events)
  validation.py   — Semantic validation layer (4 subsystems)
  cli.py          — CLI entry points (simulate command group)
```

---

## Roadmap

### Done

* [x] JSON-based CellState initialization
* [x] Make a model factory and cell agent wrapper
* [x] Port ERiQ to JAX
* [x] Allow generic NeuralODE training
* [x] Expose higher level input (hallmarks)
* [x] Add more models as submodules from literature
* [x] Define composability formalisms (Process/Port/Topology/Composite)
* [x] Semantic validation layer (units, ontology, graph analysis, coupling audit)

### Multi-Timescale Scheduler (see [design doc](docs/design-multiscale-scheduler.md))

The multi-rate Scheduler replaces the monolithic single-ODE-solve architecture. It supports continuous, discrete, and event-driven processes at different timescales. Design borrows scheduling concepts from Vivarium's Engine [3] and Ptolemy II's Directors [5], implemented natively on JAX.

* [x] Process taxonomy: CONTINUOUS / DISCRETE / EVENT kinds
* [x] LATCHED port role for discrete-to-continuous communication
* [x] Timescale declaration & auto-grouping of continuous processes
* [x] Scheduler with macro-step loop, Lie operator splitting
* [x] Discrete process dispatch & event detection at sync points
* [x] Validation extensions (LATCHED/EVOLVED conflicts, timescale checks)
* [x] Backward compatibility (single-group Scheduler = current Simulator)

### Later

* [ ] Port ERiQ as a composable Process
* [ ] Enable SBML auto-import (stub exists)
* [ ] Multi-cell / spatial: vmap over populations, inter-cell communication
* [ ] Stochastic process support (Gillespie-type discrete events)
* [ ] Expose higher level output (phenotypes)
* [ ] Add 3D support for spatial diffusion & ECM modelling
* [ ] LLM-assisted model composition

---

## License

This project is licensed under the MIT License.

---

## References

1. Cohen, Alan A., et al. "A complex systems approach to aging biology." Nature aging 2.7 (2022): 580-591.
2. Lopez-Otin, Carlos, et al. "Hallmarks of aging: An expanding universe." Cell 186.2 (2023): 243-278.
3. Agmon, Eran, et al. "Vivarium: an interface and engine for integrative multiscale modeling in computational biology." Bioinformatics 38.7 (2022): 1972-1979.
4. Alfego, D., & Kriete, A. (2017). Simulation of cellular energy restriction in quiescence (ERiQ)—a theoretical model for aging. Biology, 6(4), 44.
5. Ptolemaeus, C. (Ed.). "System Design, Modeling, and Simulation using Ptolemy II." Ptolemy.org, 2014. (Heterogeneous models of computation, Director abstraction.)
