# Proctor 2007: independent stochastic comparison

Tellurium/RoadRunner and HallSim agree on the bundled SBML, but this does
not establish reproduction of the published figures. The inhibition
trajectory retains visible differences from Figure 3 in both engines.

## Experiment

Source: `demos/models/sbml/proctor2007/proctor2007_BIOMD0000000105.xml`.
SHA256: `9473f8cde9be3662ea56903330dabe296ced65a6947644f5ae4c5babc33f33b2`.
Each simulator imports this file independently; Tellurium does not consume
HallSim-generated equations. Tellurium 2.2.13.1 / RoadRunner 2.10.0 uses
Gillespie with fixed output times. HallSim uses `simulate_ssa` directly;
this comparison does not exercise Scheduler's stochastic dispatch.

- Normal: `k69=1e-3`, 10 hours.
- Inhibited: `k69=0`, 20 hours.
- All other parameters and initial amounts unchanged; seconds internally.
- 32 independent runs per engine/condition, 241 samples per run.
- Seeds 1000–1031 in Tellurium, 0–31 in HallSim. Different RNGs preclude
  trajectory-by-trajectory equality even with matching seed numbers.

The condition durations and grouped observables follow Figures 2b and 3a
of [Proctor et al. (2007)](https://link.springer.com/article/10.1186/1752-0509-1-17).
The paper's Figure 2a instead uses 100 runs. Our bands are pointwise
mean ± 1.96 standard errors, not intervals containing 95% of trajectories.

## Results

All saved states are finite, nonnegative integer counts. At 11 sampled
normal-condition states, all 94 imported propensities agree exactly with
RoadRunner (maximum absolute difference 0); the complete stoichiometric
matrices agree exactly after aligning reaction and species identifiers.
The evidence is saved in `reaction_probes.npz` and `reaction_check.json`.

For the six species in the ensemble plots, differences between the engines'
time-averaged means are at most 1.50 combined standard errors. Standard
errors treat each trajectory's temporal mean as one independent observation,
not its correlated time samples as independent replicates. This is a
consistency check, not a formal equivalence test or exhaustive SSA validation.

| Observable | Tellurium | HallSim |
|---|---:|---:|
| Normal NatP, mean over 0–10 h | 497.51 | 497.57 |
| Normal MisP, mean over 0–10 h | 5.801 | 5.683 |
| Inhibited MisP, ensemble mean at 20 h | 426.59 | 417.34 |
| Inhibited free Ub, ensemble mean at 20 h | 16.44 | 16.97 |
| Inhibited free proteasome, ensemble mean at 20 h | 61.13 | 61.72 |
| Inhibited SeqAggP, ensemble mean at 20 h | 17.09 | 16.78 |

Normal-condition behavior is qualitatively consistent with Figure 2.
Figure 3's illustrated inhibited trajectory shows free proteasomes recovering
toward about 80; both engines instead remain near 60 at the end. Free
ubiquitin also remains higher than the illustrated near-zero trace.
These are visual comparisons, not digitized numerical reference data.
Agreement between engines isolates the remaining question to the source
model/experimental setup or published realization; it does not identify
which source parameter or equation causes the difference. No parameters
were fitted to make the plots agree.

## Reproduce and inspect

From the repository root (external simulator dependencies already live in
`.venv-comparison`):

```bash
.venv-comparison/bin/python scripts/compare_proctor2007.py tellurium
JAX_PLATFORMS=cpu .venv/bin/python scripts/compare_proctor2007.py hallsim
.venv/bin/python scripts/compare_proctor2007.py plot
```

Artifacts in `outputs/proctor2007_comparison/`:

- `normal_paper_panels.png`, `inhibited_paper_panels.png`: first independent
  realization from each engine, with the six paper observable groups and
  matched vertical scales across engines.
- `normal_ensemble.png`, `inhibited_ensemble.png`: means and uncertainty.
- `{engine}_{condition}.npz`: all 35 species, times, and seeds.
- `{engine}_{condition}.json`: source hash, versions, and run settings.
- `summary.json`: means and standard errors used above.
- `paper_figure2.jpg`, `paper_figure3.jpg`: publisher images for visual reference.

Ubiquitin conjugates count ubiquitin molecules (chain-length-weighted sums),
including proteasome-bound substrates. Total bound Ub also includes E1_Ub
and E2_Ub. Proteasome-bound substrate counts use unweighted sums. Degradation
panels use the five deposited reaction counters, degUb4 through degUb8.

## Original supplement versus BioModels

The publisher's [original SBML supplement](https://media.springernature.com/original/springer-static/esm/art%3A10.1186%2F1752-0509-1-17/MediaObjects/12918_2006_17_MOESM1_ESM.XML)
was downloaded as `outputs/proctor2007_comparison/paper_original.xml`.
A libSBML comparison found identical species initial amounts and boundary
flags, and identical reactant/product stoichiometries and kinetic-law
formulas for all 94 reactions. Both contain 39 species. Of the global
parameters, only k69 differs: the original `UPSBaselineModel` uses 0.001,
whereas BIOMD0000000105 uses 0. This is precisely the condition override
already used in the simulations above. `source_diff.json` records the
parameter lists and reaction comparison. Switching to the original file
therefore does not supply a different cell-line model.

The paper's Figure 3 compares U87MG, NIH-3T3, IMR90, and ts20 experimentally.
The paper describes a shared model with experimental interventions, not four
separately calibrated cell-line models. It explicitly discusses failure to
capture ts20's ubiquitin response during proteasome inhibition and proposes
adding ubiquitin turnover to address cell-type differences. The E1 shutdown
experiment changes k62 at 0.5 h; it is a different protocol applied to the
same network. See the paper's Figure 3, Figure 5, and Discussion.
