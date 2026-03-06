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

The composable layer reimplements key ideas from Vivarium (ports, topology, stores) natively in JAX for GPU acceleration, differentiability, and explicit composition semantics.

**Core concepts:**

| Concept       | Description |
|---------------|-------------|
| **Process**   | Equinox module (`eqx.Module`) — declares typed ports, computes `derivative(t, state)`. Parameters are JAX arrays: differentiable, JIT-compilable, vmappable. |
| **Port**      | Named connection point with a role (`INPUT` / `EVOLVED` / `EXCLUSIVE`), default value, units, description, and ontology annotation. |
| **Topology**  | Static wiring map `{proc_name: {port_name: store_path}}`. Defined at composition time, not inside processes. |
| **Composite** | Bundles processes + topology. `build_rhs()` returns a JAX-compatible `f(t, y) -> dy/dt` that sums additive (EVOLVED) contributions and enforces exclusive ownership. |
| **Simulator** | Diffrax wrapper with adaptive Tsit5 solver, perturbation support, and trajectory recording. |
| **Store**     | Flat `dict[str, jnp.ndarray]` with path-like keys (e.g., `"cytoplasm/ROS"`). A valid JAX PyTree. |

**Port roles:**

- `INPUT` — read-only, process uses this value but doesn't write a derivative.
- `EVOLVED` — additive: multiple processes can contribute derivatives to the same store path (summed).
- `EXCLUSIVE` — sole ownership: only one process may write to this store path (validated at composition time).

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
* Multiple Submodels (e.g., ERiQ, NeuralODE) plug into a Cell [3]
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

82 tests: 24 legacy + 31 composition + 27 validation.

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
  process.py      — Process base class, Port, PortRole
  store.py        — Store utilities (build, extract, route, validate)
  composite.py    — Composite: wires Processes via Topology
  simulator.py    — Diffrax-based ODE solver wrapper
  validation.py   — Semantic validation layer (4 subsystems)
  cli.py          — CLI entry points (simulate command group)
```

---

## Roadmap

* [x] JSON-based CellState initialization
* [x] Make a model factory and cell agent wrapper
* [x] Port ERiQ to JAX
* [x] Allow generic NeuralODE training
* [x] Expose higher level input (hallmarks)
* [x] Add more models as submodules from literature
* [x] Define composability formalisms (Process/Port/Topology/Composite)
* [x] Semantic validation layer (units, ontology, graph analysis, coupling audit)
* [ ] Port ERiQ as a composable Process
* [ ] Enable SBML auto-import (stub exists)
* [ ] Multi-timescale support (per-process timesteps)
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
3. Alfego, D., & Kriete, A. (2017). Simulation of cellular energy restriction in quiescence (ERiQ)—a theoretical model for aging. Biology, 6(4), 44.
