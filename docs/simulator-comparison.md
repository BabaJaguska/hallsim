# Hallsim GPU versus Tellurium and Vivarium — 2026-09-10

This benchmark compares wall-clock simulation time for the same heterogeneous
DP14 and multihallmark populations on one Tesla T4 versus serial CPU execution
on an Intel Xeon Gold 5218R. It is a comparison of these particular workloads
and configurations, not a universal ranking of the frameworks.

Tellurium is fastest for both tested populations. At 256 cells it is **5.24 times faster than Hallsim GPU for DP14** and **17.35 times faster for the composite**. Hallsim is 2.82 times faster than this Vivarium configuration for DP14, but 5.60 times slower for the composite. All final full-population comparisons pass the common numerical criterion.

## DP14 timings (seconds)

| Cells | Hallsim GPU | Tellurium CPU | Vivarium + CVODE CPU | Tellurium stepped control |
|---:|---:|---:|---:|---:|
| 1 | 0.511768 | 0.001844 | 0.028204 | 0.014469 |
| 8 | 0.736066 | 0.014047 | 0.187396 | 0.114808 |
| 32 | 0.982386 | 0.055951 | 0.709512 | 0.458736 |
| 128 | 1.634814 | 0.221933 | 3.080335 | 1.851982 |
| 256 | 2.351770 | 0.448505 | 6.635897 | 3.697700 |

## Multihallmark composite timings (seconds)

| Cells | Hallsim GPU | Tellurium CPU | Vivarium + CVODE CPU | Tellurium stepped control |
|---:|---:|---:|---:|---:|
| 1 | 7.532316 | 0.016447 | 0.055162 | 0.041536 |
| 8 | 10.914189 | 0.114639 | 0.379330 | 0.312391 |
| 32 | 24.781896 | 0.519231 | 1.568940 | 1.326634 |
| 128 | 36.060292 | 2.173753 | 6.600046 | 5.492236 |
| 256 | 73.554592 | 4.238185 | 13.137067 | 10.640392 |

[Timing plot](../outputs/simulator_comparison/timing_comparison.png) · [Machine-readable results](../outputs/simulator_comparison/summary.json)

## What was compared

- **Hallsim:** independent float64 batch members, Kvaerno5, synchronized GPU
  execution. DP14 uses the verified earlier sweep; the composite is remeasured
  at tighter tolerances after cross-simulator validation.
- **Tellurium:** Tellurium 2.2.13.1 / libRoadRunner 2.10.0, stiff CVODE, one
  continuously integrated model per cell in a serial loop.
- **Vivarium:** vivarium-core 1.6.5 Engine, one CVODE-backed Process per cell,
  a 0.1-day process timestep, and the normal in-memory emitter. Each Process
  integrates the entire coupled cell model. Vivarium handles scheduling,
  state updates, and emission; CVODE supplies the ODE integrator.
- **Stepped Tellurium control:** the same 140 calls per cell used by the
  Vivarium adapter, without its Engine or emitter. This separates repeated
  solver-call costs from framework costs.

Vivarium is a composition engine, not a replacement ODE integrator. A custom
adapter is therefore required. The adapter follows the
[Vivarium Process protocol](https://github.com/vivarium-collective/vivarium-core/blob/master/doc/tutorials/write_process.rst),
keeps a private RoadRunner solver per cell, exposes the full state vector as
a store value, and applies external changes if that store differs from its
last output. It does not wrap Hallsim, precompute trajectories, or use a
mock Engine. The fully coupled representation matches Hallsim's single
continuous group; this is not a benchmark of splitting individual hallmark
modules into separately scheduled Vivarium processes.

The models run from day 0 to 14 and return 141 samples for every cell and
every saved state. Exactly the same initial arrays are loaded from the
earlier GPU artifacts: 5% log-space variation, seed 20260910, with zero
initial states remaining zero. Parameters are shared within each population.
The default six-process, 43-path composite excludes optional proteostasis.
No model fitting is performed.

## Timing boundaries and hardware

Times are medians of three wall-clock measurements after one untimed warm
execution for every batch size. CPU timers include per-cell resets, initial
state assignment, integration, and assembly of all output arrays. Vivarium
timers also include Engine scheduling and emitter extraction. Model loading,
LLVM compilation, RoadRunner cloning, and Engine construction are excluded
from simulation timing and reported separately. Python/package startup is
not included. GPU output stays on-device during timing; host transfer and
artifact writing are excluded, as in the earlier GPU report.

Tellurium/Vivarium use no worker pool. `OPENBLAS_NUM_THREADS`,
`OMP_NUM_THREADS`, and `MKL_NUM_THREADS` are set to 1. This is a serial CPU
comparison, not an 80-logical-CPU comparison. The GPU and CPU measurements
are from the same workstation, but this is not an equal-power comparison.
The isolated `.venv-comparison` installation leaves Hallsim's environment
unchanged. Machine-readable files record package versions, affinity,
individual timings, setup timings, and solver tolerances.

Loading/compiling the Tellurium model took 0.214 seconds for DP14 and 0.259
seconds for the composite. At 256 cells, median Vivarium engine construction
(including RoadRunner clones and initialized stores) added 3.903 and 4.881
seconds respectively. These are additional to the simulation times in the
tables. Hallsim's DP14 first call at batch 256 took 18.480 seconds including
tracing/compilation and execution, versus 2.352 seconds warm. First-call
timings are not isolated compiler benchmarks; persistent compilation caches
can affect them.
The stricter composite's first 256-cell GPU call took 97.942 seconds,
versus a 73.555-second warm median.

## Equation and accuracy checks

The benchmark adapter copies the vendored SBML definitions into one
namespaced SBML model. It applies the actual Hallsim parameter values and
time scales, freezes DP14's unused Nil counter like Hallsim, and encodes the
composite's pulse, nutrient drive, damage bridge, and p53-to-CDKN1A flux.
A no-op SBML event marks each forcing jump for the integrator. The event
does not change any biological state. This is an adapter for these two
workloads, not a general-purpose composite exporter.

Before timing, derivative probes at seven saved times are compared with
Hallsim: DP14 matches exactly; the composite's maximum absolute difference
is 5.33e-15. The first trajectory check covers cells 0, 128, and 255. Timed
outputs are then checked over the full 256-cell population and all 141
samples, not only endpoints. The common criterion is
`abs(error) <= 1e-5 + 1e-3 * abs(reference)` for every saved value.

DP14 uses `rtol=1e-6, atol=1e-9` in both backends. All DP14 comparisons pass;
the largest full-population error is 0.6826 comparison-tolerance units
(absolute difference 0.02190 in DNA damage near the initial irradiation
pulse).

The earlier composite GPU simulation used `rtol=1e-7, atol=1e-10` and was
checked against Radau for cell 0. Broader cross-simulator validation found
larger phase-sensitive errors in other cells: up to approximately four
comparison-tolerance units. Both CPU and GPU settings were refined for
this comparison. Final settings are Hallsim `rtol=1e-8, atol=1e-11` and
CVODE `rtol=1e-9, atol=1e-12`. CVODE uses the same settings inside and outside
Vivarium. Nominal local tolerances are not assumed to imply equal global
trajectory accuracy; the output check decides whether a comparison passes.
[Tellurium documents the CVODE tolerance controls here](https://tellurium.readthedocs.io/en/latest/notebooks.html).

Final composite results pass across all 256 cells and 141 samples:

| Comparison | Maximum absolute difference | Maximum tolerance units |
|---|---:|---:|
| Tellurium vs Hallsim GPU | 5.18e-5 | 0.307 |
| Stepped Tellurium vs Hallsim GPU | 6.64e-5 | 0.354 |
| Vivarium vs Hallsim GPU | 6.64e-5 | 0.354 |
| Vivarium vs Tellurium | 2.81e-5 | 0.124 |

The final composite arrays from all backends are finite and nonnegative.
Subtracting the stepped control from Vivarium gives approximately 2.94
seconds of additional Engine/emission work for DP14 and 2.50 seconds for
the composite at 256 cells. This decomposition is specific to the adapter
and output configuration, not a general overhead constant for Vivarium.

The JSON retains failures against the earlier reference so the accuracy
finding is not erased. Final full-population checks are stored under
`final_population_accuracy` against the new GPU reference. The saved CPU
populations are the actual timed outputs; rechecking does not replace them
with new trajectories. These checks establish numerical agreement for this
workload, not biological validity.

## Reproduction

From the repository root, with access to the GPU and Hallsim's SBML cache:

```bash
uv venv --python .venv/bin/python .venv-comparison
uv pip install --python .venv-comparison/bin/python tellurium==2.2.13.1 vivarium-core==1.6.5
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/verify_gpu_batch.py --models composite --sizes 1,8,32,128,256 --repeats 3 --rtol 1e-8 --atol 1e-11 --out outputs/gpu_comparison_accuracy
JAX_PLATFORMS=cpu .venv/bin/python scripts/prepare_simulator_comparison.py --composite-gpu-dir outputs/gpu_comparison_accuracy
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/hallsim-mpl .venv-comparison/bin/python scripts/bench_simulator_comparison.py --models dp14
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/hallsim-mpl .venv-comparison/bin/python scripts/bench_simulator_comparison.py --models composite --tolerance-factor 0.1
MPLCONFIGDIR=/tmp/hallsim-mpl .venv-comparison/bin/python scripts/bench_simulator_comparison.py --summarize
```

The comparator loads uv's matching `libpython` explicitly before importing
libRoadRunner. This avoids relying on a system-wide shared-library path.
The benchmark reads the DP14 artifacts described in
[the earlier GPU report](gpu-verification.md).

Artifacts are under `outputs/simulator_comparison/`: SBML workloads, initial
states and RHS probes, validation arrays, actual timed CPU populations,
timing JSON, generated tables, and a timing plot. The new GPU population
and its timing sweep are under `outputs/gpu_comparison_accuracy/`.
