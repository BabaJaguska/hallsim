# HallSim: A Modular, Multi-Scale Simulator for Aging Biology
[![Basic CI/CD Workflow](https://github.com/BabaJaguska/HallSim/actions/workflows/basic_CI_linux.yaml/badge.svg)](https://github.com/BabaJaguska/HallSim/actions/workflows/basic_CI_linux.yaml)

## 🧬 Background & Motivation

Aging is a complex, network-level phenomenon. Its hallmarks — such as mitochondrial dysfunction, genomic instability, and altered intercellular communication — do not act in isolation. Instead, they form dense webs of feedback and crosstalk.

Traditional approaches to modeling aging have been reductionist. Inspired by conceptual frameworks like Cohen et al. (2022) [1], HallSim aims to embrace complex systems theory and explore emergent properties of aging arising from loss of resilience across multiple sub-systems.

Existing bio simulation tools and model repositories provide standardized formats and simulation capabilities for individual dynamical models, but they generally lack support for composable model libraries — i.e., reusable modules with defined high-level interfaces that can be assembled into heterogeneous larger multi-scale dynamical systems.

HallSim’s goal is to enable compositional co-simulation of reusable systems-biology modules, with a focus on aging biology.

---

## 🌟 Project Goals

* Build a modular, multi-scale agent-based simulator
* Allow plug-and-play modules with exposed high level abstractions corresponding to 12 hallmarks of aging [2]
* Incorporate an LLM layer helping integrate new modules in a semi-automated way
* Enable simulation of aging trajectories, interventions, and emergent phenotypes
* Serve as an educational tool
* Somewhere down the road become an in-silico testbed for perturbations (radiation, caloric restriction, therapies, etc.)

---

## 🛠️ Architecture:

* Each Cell has a `CellState` representing internal biology
* Multiple Submodels (e.g., ERiQ, neuralode) plug into a Cell [3]
* Cells reside on a 2D grid (expandable to 3D, need to add spatial dynamics)
* Simulation operates through `Diffrax` ODE integration
* Models of unknown dynamics or surrogate models can be trained as a NeuralODE using `optax` and `equinox`
* Hallmarks are a high level input, allowing a directional interface through a 0-1 handle specifying the level to which a hallmark is activated

![HallSim Architecture](https://github.com/BabaJaguska/hallsim/blob/master/hallsim_architecture.png?raw=true)

---

## 🚀 Getting Started


`make install` or `make install-dev` for developers.

`make test` will run a series of pytest tests.

`make run` will run the default simulation, or you can do `simulate basic` (mitochondrial damage sim)
or `simulate kick` to add a step perturbation to the basic simulation.
cli.py is your entry point, where you control the CLI and what options it offers.

---

## 🧰 Dev Instructions

- `pyproject.toml` is the single source of dependencies. These are compiled when you run `make update`.

- To add a model, create a new `.py` file in `src/hallsim/models/`. Optionally, add its dedicated JSON config file under `configs/`.

- If the model introduces new state variables, add them both to `default_cell_config.json` and to the `Cell` class.

Note: Models are aggregated on an additive basis in the ODE rhs. If your model's effect is supposed to be multiplicative with other models, well, there's gonna be more work.  

---

## 🛋️ Roadmap

* [x] JSON-based CellState initialization
* [x] Make a model factory and cell agent wrapper
* [x] Port ERiQ to JAX
* [x] Allow generic NeuralODE training 
* [x] Expose higher level input (hallmarks) 
* [x] Add more models as submodules from literature
* [ ] Enable SBML and other standardized format support
* [ ] Expose higher level output (phenotypes)
* [ ] Define composability formalisms and improve integration layer
* [ ] Potentially repackage models to encourage reusability
* [ ] Add 3D support for spatial diffusion & ECM modelling (CAX or PhysiCell integration? Vertex models?)

---

## 📜 License

This project is licensed under the MIT License.

---

## References

1. Cohen, Alan A., et al. "A complex systems approach to aging biology." Nature aging 2.7 (2022): 580-591.
2. López-Otín, Carlos, et al. "Hallmarks of aging: An expanding universe." Cell 186.2 (2023): 243-278.
3. Alfego, D., & Kriete, A. (2017). Simulation of cellular energy restriction in quiescence (ERiQ)—a theoretical model for aging. Biology, 6(4), 44.






