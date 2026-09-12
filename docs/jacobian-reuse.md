# Jacobian reuse experiment — 2026-09-10

**Result:** opt-in StepChord reduces the 256-cell composite runtime from
74.178 s to **55.660 s** on the same Tesla T4 (25.0% less time, 1.33x
throughput), with full-trajectory accuracy passing. It is slower for small
batches and remains about 13.1x slower than the 4.238 s Tellurium baseline.

Integration base: `df4cbd4` (Compiled events), with the local per-member plan
shape fix retained. Rebuilding a compiled core for a different batch shape now
also forwards the upstream discrete and event handlers. A regression test
checks that both still fire in a batched run.

## Method

`scripts/profile_jacobian_reuse.py` runs the existing Scheduler API with
Kvaerno5 and four root-finder configurations: Newton (fresh Jacobian every
iteration), Chord (reuse within a stage), and VeryChord (reuse across stages
within a step), plus StepChord (step reuse with Cauchy termination).
No custom integrator or bypass of Hallsim is involved.

The inputs, 14-day horizon, 141 output samples, and CVODE reference are the
same as in `simulator-comparison.md`. Integration tolerances remain
`rtol=1e-8, atol=1e-11`; explicitly supplied root tolerances are
`rtol=1e-8, atol=1e-6` for every method, with ten root iterations allowed.
Different root finders have different convergence criteria even with these
same numbers. All saved values must satisfy
`abs(error) <= 1e-5 + 1e-3 * abs(CVODE reference)`.

Timing uses float64 on Tesla T4, one warm-up then three synchronized calls;
compilation and output transfer are excluded from warm timings. Compilation
cache is disabled. Nsight traces separately capture one warm call on another
T4 and are not used as the uninstrumented timing measurements.

## Updated Newton baseline

| Batch | Warm median (s) | Attempted steps per cell | Accuracy |
|---:|---:|---:|---|
| 1 | 7.504 | 1,440 | pass |
| 32 | 24.872 | 879–2,949 | pass |
| 256 | 74.178 | 879–2,956 | pass |

The worst full-population error is 0.3071 comparison-tolerance units.

## Newton profile

For one cell, Nsight records 19,083 calls to the main LU factorization kernel.
That kernel takes 0.824 s, 47.6% of summed GPU kernel duration. There are
43,722 `cuStreamSynchronize` calls, totalling 3.788 s in the CUDA API summary.
The instrumented wall time is 8.106 s versus 7.504 s without tracing.
Kernel durations, API durations, and wall time are overlapping measurements
and must not be added together. These results establish substantial repeated
linear algebra and synchronization costs; they do not attribute every
synchronization specifically to Jacobian construction.

## Reuse results

| Method | Batch | Warm seconds | Attempted steps | Rejected steps | Accuracy |
|---|---:|---:|---:|---:|---|
| Chord, stage reuse | 1 | 16.971 (median of 3) | 1,440 | 60 | pass |
| Chord, stage reuse | 32 | 45.787 (median of 3) | 879–2,955 per cell | 2,360 total | pass |
| Chord, stage reuse | 256 | 77.119 (median of 3) | 879–2,955 per cell | 19,793 total | pass |
| VeryChord, step reuse | 1 | 95.378 (one warm run) | 9,467 | 4,470 | pass |
| StepChord, step reuse with Cauchy termination | 1 | 15.165 (median of 3) | 1,444 | 62 | pass |
| StepChord, step reuse with Cauchy termination | 32 | 38.092 (median of 3) | 879–2,962 per cell | 2,347 total | pass |
| StepChord, step reuse with Cauchy termination | 256 | 55.719 (median of 3) | 879–2,962 per cell | 19,803 total | pass |

The VeryChord result above is uninstrumented. Its first Nsight run stalled
and was terminated; the incomplete trace is not used for comparisons.

The completed stage-Chord trace explains why fewer factorizations do not
translate to a speedup:

| One-cell trace metric | Newton | Stage Chord |
|---|---:|---:|
| Main LU kernel calls | 19,083 | 11,528 |
| Main LU kernel seconds | 0.824 | 0.495 |
| All GPU kernel calls | 401,278 | 230,808 |
| Summed GPU kernel seconds | 1.731 | 1.066 |
| `cuGraphLaunch` calls | 30,754 | 88,903 |
| `cuStreamSynchronize` calls | 43,722 | 52,447 |
| Seconds inside `cuStreamSynchronize` | 3.788 | 10.112 |
| Instrumented wall seconds | 8.106 | 17.694 |

Stage reuse preserves essentially the same integration workload and saves
factorization work, but increases graph dispatch and synchronization costs.
Whole-step VeryChord also increases the adaptive integration workload.
The `StepChord` adapter honors Diffrax's `init_state` hint while
retaining Optimistix Chord's Cauchy convergence test, to separate the effect
of reuse from VeryChord's different stopping criteria. The reusable class is
in `src/hallsim/root_finders.py`; no default solver change has been made.

The StepChord control restores nearly the Newton step count, but remains
slower at batches 1 and 32. At batch 256 it reduces runtime by about 25%.
This isolates two separate issues: VeryChord's convergence behavior
increases integration work on this workload, while the cached-Jacobian path
also increases GPU dispatch/synchronization overhead. A lower arithmetic
cost alone is insufficient to recommend a default change. StepChord is an
explicit option for workloads where population size amortizes that overhead.

```python
import diffrax as dfx
from hallsim import Scheduler
from hallsim.root_finders import StepChord

scheduler = Scheduler(
    rtol=1e-8, atol=1e-11,
    implicit_solver=dfx.Kvaerno5(
        root_finder=StepChord(rtol=1e-8, atol=1e-6),
    ),
)
```

The first StepChord sweep used GPU 0; Newton and stage Chord used GPU 1.
All are Tesla T4s. A separate same-GPU confirmation is saved under
`same_gpu_confirmation/`: three warm calls took 55.683, 55.653, and 55.660 s.
The median is 55.660 s, with maximum error 0.3138 comparison-tolerance units.
All trajectories are finite and nonnegative. Use this same-device result
for the headline speedup rather than the first sweep's 55.719 s.

## Validation

Integration tests: 135 passed, one deselected pre-existing SBML cache test,
and one expected failure. The new batch-shape/event/discrete regression
passed separately. The optional root finder has two analytic nonlinear-decay
tests covering heterogeneous batched trajectories and parameter gradients
in forward and reverse mode; both tests pass on CPU and GPU. All completed
composite comparisons use full saved trajectories against CVODE, not only
endpoints. The solver default remains Newton because small-batch timings
regress and the gain is workload-dependent.

Completed Nsight reports, CSV summaries, trajectories and machine-readable
timings are under `outputs/jacobian_reuse/`. Newton results are in
`results.json`, stage reuse in `stage_chord/results.json`, and the
uninstrumented VeryChord case in `whole_step/results.json`.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false HALLSIM_COMPILATION_CACHE_DIR=off .venv/bin/python scripts/profile_jacobian_reuse.py --methods newton,chord,step_chord
```

For a warm-only Nsight trace, use `nsys profile --trace=cuda --sample=none
--cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop` and
pass `--profile --methods newton --sizes 1 --repeats 1` to the script. Use a
different output directory for each experiment. `nsys stats --report
cuda_gpu_kern_sum,cuda_api_sum` summarizes the resulting report.
