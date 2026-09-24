# hallsim: A Differentiable, Composable Multi-Scale Modelling Framework for Aging Biology
[![Basic CI/CD Workflow](https://github.com/BabaJaguska/HallSim/actions/workflows/basic_CI_linux.yaml/badge.svg)](https://github.com/BabaJaguska/HallSim/actions/workflows/basic_CI_linux.yaml)

**hallsim composes independently-published systems-biology models into one multi-scale dynamical system and calibrates the whole thing by gradient descent through the ODE solve.** Built on JAX / Equinox / Diffrax, with a focus on aging biology, where no single model captures the crosstalk between hallmarks.

- **End-to-end differentiable.** The entire composite of multiple stiff models, operator-split across timescales, is a single differentiable function. 
- **Agent-friendly by construction.** 
- **Scale is central, not an edge case.** 

## Goals
- Serve as an in-silico testbed for perturbations (rapamycin, caloric restriction, …).
- Make large multi-model composition, tractable for AI agents building at a scale no one assembles by hand.
- Educational material letting students see what happens across a range of processes when a perturbation is applied 


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

### Workflow

1. **Find** — `simulate find <query>`: search the repositories for a deposit that *emits* what you need; `simulate find-data <query>` does the same for a time course to calibrate against, across GEO, ArrayExpress, PRIDE, MetaboLights, Metabolomics Workbench, the BioImage Archive and Zenodo, reading each hit's arms and timepoints from its sample titles; `--composite` keeps the hits that measure something your composite carries, and `--paper` lists a paper's own data. When nothing is deposited, `simulate discover <topic>` finds the papers and the code they link. `simulate census` measures how much of BioModels, curated and uncurated, clears the gate at all and keeps the stamped verdict table under `results/census/`; `simulate census-data` does the same for the data repositories: every deposited time course, and which screened models it could score.
2. **Screen** — `simulate screen <id-or-path>`: triage and the numerical screen of that one model on its own. Nothing joins a composite unscreened.
3. **Import** — `process_from_sbml` / `process_from_xpp`, then `reconciled_to` to put it on the composite's clock.
4. **Compose** — `Composite` with a topology; `analyze_composability` where two models overlap.
5. **Calibrate** — `CalibrationProblem` with held-out arms

### Demos & tests

`simulate demo --help` lists examples: framework mechanics on
toy processes, and one case study that composes three published SBML
models and calibrates them against GSE248823 (`simulate demo
multi-hallmark run`; the first run fetches the dataset). 
`simulate view module:name` serves any composite as a page: levers over
its handles and its wiring with a signal trace; `--bake DIR` writes the
levers as a static site instead.
`make test` runs the unit suite.

## What you can do with it

- **Compose published models.** Search BioModels, JWS Online, ModelDB, BioSimulations, Physiome and Europe PMC supplements from one call, filtered by what a deposit *emits*; import SBML, COPASI `.cps` or XPPAUT `.ode`. 
- **Pull a perturbation handle.** A named, differentiable severity that moves the right parameters across models. The demos ship the hallmarks of aging as one registry on their models; a drug or a gene dosage is another entry, applied the same way.
- **Calibrate against data.** Gene-reporter concordance, log2-fold-change loss, MAP priors, differentiation through the stiff solve. 
- **Run batched population studies.** A `(batch, n_vars)` `y0` flows through the solve as one computation — no `vmap` to write. 

## License

MIT.

## References

If you use hallsim, please cite our paper: https://doi.org/10.64898/2026.09.22.753641 
