# HallSim: A Differentiable, Composable Multi-Scale Modelling Framework for Aging Biology
[![Basic CI/CD Workflow](https://github.com/BabaJaguska/HallSim/actions/workflows/basic_CI_linux.yaml/badge.svg)](https://github.com/BabaJaguska/HallSim/actions/workflows/basic_CI_linux.yaml)

**HallSim composes independently-published systems-biology models into one multi-scale dynamical system and calibrates the whole thing by gradient descent through the ODE solve.** Built on JAX / Equinox / Diffrax, with a focus on aging biology, where no single model captures the crosstalk between hallmarks.

- **End-to-end differentiable.** The entire composite of multiple stiff models, operator-split across timescales, is a single differentiable function. 
- **Agent-friendly by construction.** 
- **Scale is central, not an edge case.** 

## Why

Aging is network-level: its hallmarks — mitochondrial dysfunction, genomic instability, altered intercellular communication, and more — form dense webs of feedback, not isolated axes. Hallsim explores emergent properties that arise from loss of resilience across sub-systems.

- Calibrate interventions and emergent phenotypes against real data, with held-out validation.
- Make multi-model composition tractable for AI agents building at a scale no one assembles by hand.
- Serve as an in-silico testbed for perturbations (rapamycin, caloric restriction, …).

## Architecture

![HallSim Architecture](docs/assets/hallsim_architecture.png)

| Concept | Role |
|---|---|
| **Process** | `eqx.Module` with typed ports and a kind (CONTINUOUS / DISCRETE / EVENT); parameters are JAX arrays. |
| **Port** | Named connection point with a role (INPUT / EVOLVED / EXCLUSIVE / LATCHED / ASSIGNED), units, and ontology. |
| **Topology** | Static wiring `{proc: {port: store_path}}`, outside the processes. |
| **Composite** | Bundles processes + topology into a flat, JAX-compatible ODE RHS; auto-groups by timescale. |
| **Scheduler** | The one runner for every composite shape — timescale groups, discrete dispatch, events, batched populations. |
| **Store** | Flat `dict[str, jnp.ndarray]` with path-like keys; a JAX PyTree. |

A composition-time **validation layer** (units via pint, ontology IDs, feedback/fan-in graph analysis, duplicate-reaction heuristics) runs warnings-by-default and raises on hard conflicts.


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
Calibration through the solve is documented in
**[docs/calibration.md](docs/calibration.md)**.

### Workflow

1. **Find** — `simulate find <query>`: search the repositories for a deposit that *emits* what you need; `simulate find-data <query>` does the same for a GEO dataset to calibrate against.
2. **Screen** — `simulate screen <id-or-path>`: triage and the numerical screen of that one model on its own. Nothing joins a composite unscreened.
3. **Import** — `process_from_sbml` / `process_from_xpp`, then `reconciled_to` to put it on the composite's clock.
4. **Compose** — `Composite` with a topology; `analyze_composability` where two models overlap.
5. **Calibrate** — `CalibrationProblem` with held-out arms ([docs/calibration.md](docs/calibration.md)).

### Demos & tests

Framework mechanics, end to end:

```bash
simulate demo compose        # a minimal two-process composite
simulate demo compose-kick   # the same, with a mid-run perturbation
simulate demo multiscale     # continuous + discrete + event processes on one clock
simulate demo clamp          # chronic vs transient exposure: hold a consumed species
simulate demo stiffness      # per-group stiffness verdict + solver routing
simulate info           # what the architecture exposes
```

Finding a model to compose is part of the workflow, not a prerequisite for it:

```bash
simulate find NFkB inflammation --produces 'IL6|CXCL8'   # every repository, filtered by what a deposit emits
```

One worked case study composes three published SBML models and calibrates them
against a public dataset:

```bash
simulate demo multi-hallmark run        # score it out of the box, no fitting
simulate demo multi-hallmark calibrate  # fit, then evaluate on held-out arms
simulate demo multi-hallmark sweep      # two-hallmark severity sweep
simulate demo multi-hallmark-ssa        # one-way DP14/GZ06 + Proctor SSA hybrid
simulate demo hallmark-levers           # browser page: pull a hallmark, watch all three re-solve
```

The first `run` downloads the dataset (GEO GSE248823, about 200 MB unpacked)
into `data/`; `simulate demo multi-hallmark fetch-data` does only that.

The lever page needs the `app` extra (`pip install "hallsim[app]"`). Each
slider is a hallmark severity; a pull re-solves the composite against
control. Proctor 2007 runs as a
population of cells at reaction level, and the etoposide exposure window
is shaded, with longer windows on a switch.

## What you can do with it

- **Compose published models.** Search BioModels, JWS Online, ModelDB, BioSimulations, Physiome and Europe PMC supplements from one call, filtered by what a deposit *emits*; import SBML, COPASI `.cps` or XPPAUT `.ode`. 
- **Pull a perturbation handle.** A named, differentiable severity that moves the right parameters across models. The hallmarks of aging ship as one registry; a drug or a gene dosage is another entry, applied the same way.
- **Calibrate against data.** Gene-reporter concordance, log2-fold-change loss, MAP priors, differentiation through the stiff solve. 
- **Run batched population studies.** A `(batch, n_vars)` `y0` flows through the solve as one computation — no `vmap` to write. 

## License

MIT.

## References

Preprint available at: []
