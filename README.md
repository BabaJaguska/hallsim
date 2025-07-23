# HallSim: A Modular, Multi-Scale Simulator for Aging Biology
[![Basic CI/CD Workflow](https://github.com/BabaJaguska/HallSim/actions/workflows/basic_CI_linux.yaml/badge.svg)](https://github.com/BabaJaguska/HallSim/actions/workflows/basic_CI_linux.yaml)

## 🧬 Background & Motivation

Aging is a complex, network-level phenomenon. Its "hallmarks" — such as mitochondrial dysfunction, genomic instability, and altered intercellular communication — do not act in isolation. Instead, they form dense webs of feedback and crosstalk.

Traditional approaches to modeling aging have been reductionist. HallSim aims to go beyond that: to embrace complex systems theory, simulate **emergent behavior**, and enable researchers to explore how cellular-level pathways give rise to tissue-level phenotypes like frailty, fibrosis, and skin aging.

Inspired by conceptual frameworks like [Cohen et al. (2022)](https://pubmed.ncbi.nlm.nih.gov/37117782/), HallSim lets researchers investigate how resilience erodes, tipping points arise, and interventions (e.g., caloric restriction, rapamycin) shift system behavior.

---

## 🌟 Project Goals

* Build a modular, multi-scale agent-based simulator
* Allow plug-and-play hallmark modules (ODE/SBML)
* Enable simulation of aging trajectories, interventions, and emergent phenotypes
* Serve as an in-silico testbed for perturbations (radiation, caloric restriction, therapies, etc.)

---

### 🛠️ Architecture:

* Each Cell has a `CellState` representing internal biology
* Multiple Submodels (e.g., ERiQ) plug into a Cell, mh, well, not yet
* Cells reside on a 2D grid (expandable to 3D)
* Simulation operates through `Diffrax` ODE integration                           |

---

## 🚀 Getting Started



```make install``` or ```make install-dev``` for developers.

```make test``` will run a series of pytest tests.

---

## 🧰 Dev Instructions

pyproject.toml is the single source of dependendies, compiled when you run `make update`.

---

## 🛋️ Roadmap

* [x] JSON-based CellState initialization
* [ ] Structured output (phenotypes?)
* [ ] Port ERiQ to JAX
* [ ] Add more models as submodules from literature
* [ ] Spatial diffusion & ECM modelling
* [ ] SDE noise terms
* [ ] Output metrics to tissue phenotype (e.g. wrinkling)
* [ ] Add SBML support
* [ ] Add 3D support, perhaps through PysiCell integration

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments




