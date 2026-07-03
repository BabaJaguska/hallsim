# HallSim: A Differentiable, Composable Multi-Scale Simulator for Aging Biology
[![Basic CI/CD Workflow](https://github.com/BabaJaguska/HallSim/actions/workflows/basic_CI_linux.yaml/badge.svg)](https://github.com/BabaJaguska/HallSim/actions/workflows/basic_CI_linux.yaml)

**HallSim composes independently-published systems-biology models into one multi-scale dynamical system — and calibrates the whole thing by gradient descent through the ODE solve.** Built on JAX / Equinox / Diffrax, with a focus on aging biology, where no single model captures the crosstalk between hallmarks.

- **End-to-end differentiable.** The entire composite — multiple stiff SBML models, operator-split across timescales — is a single differentiable function. Mechanism parameters spread across separate publications are fit with the same reverse-mode autodiff that trains neural networks, *through* the stiff ODE solve. GPU-friendly, with held-out validation. See [docs/calibration.md](docs/calibration.md).
- **Agent-friendly by construction.** A model is discovered and imported from BioModels in two calls (`search_for_model` → `process_from_sbml`), wired by a plain `{process: {port: path}}` topology dict, and its fittable parameters self-document via `Composite.calibration_targets()`. Typed ports carry units and ontology; `analyze_composability` proposes how to merge overlapping models. Meant for an LLM agent to assemble and calibrate a digital twin without bespoke glue. See [docs/architecture.md](docs/architecture.md).

## Why

Aging is network-level: its hallmarks — mitochondrial dysfunction, genomic instability, altered intercellular communication, and more — form dense webs of feedback, not isolated axes. Inspired by complex-systems framings of aging (Cohen et al. 2022 [1]), HallSim explores emergent properties that arise from loss of resilience across sub-systems.

HallSim is a **composition framework** — you bring the modules (hand-written, SBML-imported, or learned). It is JAX-native end to end — the composition itself is differentiable, JIT-compiled, and GPU-batched — and **built for an LLM agent** to discover, assemble, and calibrate a 'digital twin' without bespoke glue.

## Goals

- A composable, differentiable, multi-scale simulator for aging biology — bring your own modules (hand-written, SBML-imported, or learned via `NeuralODE`).
- High-level severity handles for the 12 hallmarks of aging [2] (4 mapped today; each new one a single handle away).
- Calibrate interventions and emergent phenotypes against real data, with held-out validation.
- Make multi-model composition tractable for AI agents building digital twins.
- Serve as an educational in-silico testbed for perturbations (rapamycin, caloric restriction, …).

## Architecture

![HallSim Architecture](docs/assets/hallsim_architecture.png)

| Concept | Role |
|---|---|
| **Process** | `eqx.Module` with typed ports and a kind (CONTINUOUS / DISCRETE / EVENT); parameters are JAX arrays. |
| **Port** | Named connection point with a role (INPUT / EVOLVED / EXCLUSIVE / LATCHED), units, and ontology. |
| **Topology** | Static wiring `{proc: {port: store_path}}`, outside the processes. |
| **Composite** | Bundles processes + topology into a flat, JAX-compatible ODE RHS; auto-groups by timescale. |
| **Scheduler** | The one runner for every composite shape — timescale groups, discrete dispatch, events, batched populations. |
| **Store** | Flat `dict[str, jnp.ndarray]` with path-like keys; a JAX PyTree. |

A composition-time **validation layer** (units via pint, ontology IDs, feedback/fan-in graph analysis, duplicate-reaction heuristics) runs warnings-by-default and raises on hard conflicts.

**Full details** — process kinds & port roles, the validation layer, composing composites, the merge-or-couple protocol, SBML import, hallmark handles, and batched population studies — are in **[docs/architecture.md](docs/architecture.md)**. The multi-rate scheduler (splitting schemes, coupling modes, adaptive step control) is in **[docs/design-multiscale-scheduler.md](docs/design-multiscale-scheduler.md)**.

## Quickstart

```bash
make install                 # or: make install-dev
```

```python
import jax.numpy as jnp
from hallsim.process import Process, Port, PortRole
from hallsim.composite import Composite
from hallsim.scheduler import Scheduler

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

composite = Composite(
    processes={"decay": Decay(), "growth": Growth()},
    topology={"decay": {"x": "pool/x"}, "growth": {"x": "pool/x"}},
    semantic_validation=True,   # optional unit/semantic checks
)
result = Scheduler().run(composite, t_span=(0.0, 100.0), macro_dt=1.0, save_dt=1.0)
print(result.get("pool/x").shape)
```

Parameters are JAX arrays, so you can `jax.grad` through an entire simulation.
For the real, multi-model, calibrated version see
**[docs/calibration.md](docs/calibration.md)** and the runnable
[`demos/multi_hallmark_calibrate.py`](demos/multi_hallmark_calibrate.py).

### Demos & tests

```bash
simulate compose | compose-kick | multiscale | validate-demo | info
.venv_hallsim/bin/python demos/multiscale_coupling_demo.py
make test
```

## What you can do with it

- **Compose published SBML models.** Discover on BioModels and import in two calls; the full mechanism surface auto-populates and is discoverable. → [docs/architecture.md#sbml-import](docs/architecture.md#sbml-import)
- **Turn hallmark severities.** 0–1 differentiable handles that modulate the right parameters across models; interventions (rapamycin, CR) live on the hallmark layer they perturb. → [docs/architecture.md#hallmark-handles](docs/architecture.md#hallmark-handles)
- **Calibrate against data with held-out validation.** Gene-reporter concordance, log2-fold-change loss, MAP priors, differentiation through the stiff solve. → [docs/calibration.md](docs/calibration.md)
- **Run batched population studies.** A `(batch, n_vars)` `y0` flows through the solve as one computation — no `vmap` — near-flat on GPU. → [docs/architecture.md#population-studies-via-batched-y0](docs/architecture.md#population-studies-via-batched-y0)

## Roadmap

Scheduler (waveform relaxation, IMEX, Mori-Zwanzig coupling), models & validation (lipid-metabolism extension, stochastic/Gillespie support, multi-cell communication), and SBML import improvements (event translation). Calibration fits fold-change time courses (multi-timepoint), not just endpoints — see [docs/calibration.md](docs/calibration.md).

## License

MIT.

## References

1. Cohen, A. A., et al. "A complex systems approach to aging biology." *Nature Aging* 2.7 (2022): 580–591.
2. López-Otín, C., et al. "Hallmarks of aging: An expanding universe." *Cell* 186.2 (2023): 243–278.
3. Agmon, E., et al. "Vivarium: an interface and engine for integrative multiscale modeling in computational biology." *Bioinformatics* 38.7 (2022): 1972–1979.
4. Alfego, D., & Kriete, A. "Simulation of cellular energy restriction in quiescence (ERiQ)." *Biology* 6.4 (2017): 44.
5. Ptolemaeus, C. (Ed.). *System Design, Modeling, and Simulation using Ptolemy II.* Ptolemy.org, 2014.
