# HallSim: A Modular, Multi-Scale Simulator for Aging Biology
[![Basic CI/CD Workflow](https://github.com/BabaJaguska/HallSim/actions/workflows/basic_CI_linux.yaml/badge.svg)](https://github.com/BabaJaguska/HallSim/actions/workflows/basic_CI_linux.yaml)

## 🧬 Background & Motivation

Aging is a complex, network-level phenomenon. Its "hallmarks" — such as mitochondrial dysfunction, genomic instability, and altered intercellular communication — do not act in isolation. Instead, they form dense webs of feedback and crosstalk.

Traditional approaches to modeling aging have been reductionist. Inspired by conceptual frameworks like [Cohen et al. (2022)](https://pubmed.ncbi.nlm.nih.gov/37117782/), HallSim aims to embrace complex systems theory and explore emergent properties of aging arising from loss of resilience across multiple sub-systems.

---

## 🌟 Project Goals

* Build a modular, multi-scale agent-based simulator
* Allow plug-and-play modules (ODE/SBML) with exposed high level abstractions corresponding to 12 hallmarks of aging
* Incorporate an LLM layer helping integrate new modules in a semi-automated way
* Enable simulation of aging trajectories, interventions, and emergent phenotypes
* Eventually serve as an in-silico testbed for perturbations (radiation, caloric restriction, therapies, etc.)

---

### 🛠️ Architecture:

* Each Cell has a `CellState` representing internal biology
* Multiple Submodels (e.g., ERiQ, neuralode) plug into a Cell
* Cells reside on a 2D grid (expandable to 3D, need to add spatial dynamics)
* Simulation operates through `Diffrax` ODE integration
* Models of unknown dynamics or surrogate models can be trained as a NeuralODE using `optax` and `equinox`

---

## 🚀 Getting Started


`make install` or `make install-dev` for developers.

`make test` will run a series of pytest tests.

`make run` will run the default simulation, or you can do `simulate basic` (mitochondrial damage sim)
or `simulate kick` to add a step perturbation to the basic simulation.
cli.py is your entry point, where you control the CLI and what options it offers.

---

## 🧰 Dev Instructions

pyproject.toml is the single source of dependendies, compiled when you run `make update`.

---

## 🛋️ Roadmap

* [x] JSON-based CellState initialization
* [x] Make a model factory and cell agent wrapper
* [x] Port ERiQ to JAX
* [x] Allow generic NeuralODE training 
* [ ] Add more models as submodules from literature
* [ ] Enable SBML and other standardized format support
* [ ] Expose higher level input (hallmarks) and output (phenotypes)
* [ ] Add SDE noise terms
* [ ] Add 3D support for spatial diffusion & ECM modelling (CAX or PhysiCell integration? Vertex models?)

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments




