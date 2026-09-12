# DP14 and multihallmark GPU verification — 2026-09-10

Follow-up: the [Tellurium/Vivarium comparison](simulator-comparison.md)
extends validation to every saved value in all 256 cells. It finds larger
composite phase errors outside the single-cell Radau check below and uses
`rtol=1e-8, atol=1e-11` for the final composite GPU benchmark. The timings
and checks below remain the record of the earlier tolerance settings.

DP14 runs as an independent heterogeneous batch on a Tesla T4. Runtime is
strongly sublinear, but **not flat** over the measured 1–256-cell range.
Verification found and fixed a `RunPlan` reuse bug in `Scheduler._execute`:
planning with an unbatched initial state and then supplying a batch retained
the scalar core, making the implicit solver factorize one population-sized
system. The executor now selects a core appropriate to the supplied shape,
retaining the plan's numerical routing. Both directions of rank change are
covered by an analytic-solution and compiled-work regression test.

## Synchronized warm timings

One Tesla T4 (physical GPU 1), float64, JAX 0.10.2, Diffrax 0.7.2,
Equinox 0.13.8, NVIDIA driver 575.57.08. Repository base commit
`297d0e7836442542323de6a0991226a9771e6ae9` plus the fix in this change.
Each entry is the median of three synchronized calls after a separate cold
call for that batch shape; compilation and host transfer are excluded.

| Cells | DP14 seconds | Composite seconds |
|---:|---:|---:|
| 1 | 0.512 | 3.422 |
| 8 | 0.736 | 4.964 |
| 32 | 0.982 | 11.905 |
| 128 | 1.635 | 19.097 |
| 256 | 2.352 | 39.901 |

DP14: 256 times the population costs 4.60 times the runtime, about 55.7
times greater throughput than the one-cell batch. From 32 to 256 cells,
runtime grows 2.39 times. This supports efficient batching, not a claim of
constant runtime. The composite grows 11.66 times from 1 to 256 cells.
These are measured scaling ratios, not comparisons against CPU performance.

Before the fix, the scalar-plan path measured 0.634, 1.828, and 15.264 seconds
for 1, 8, and 32 cells. The 128-cell case was stopped after more than 12
minutes without completing its cold call. Do not interpret these numbers
as the performance of the existing direct `run(comp, y0=batch)` path: the
defect specifically affected reuse of a plan built with a different shape.

## Workload and numerical checks

Both workloads span day 0–14, saving 141 points at 0.1-day intervals.
Initial values receive seeded multiplicative lognormal variation
(`seed=20260910`, log-space sigma 0.05); zero states remain zero and assigned
states are recomputed by the framework. Parameters are shared across cells.
The benchmark uses Kvaerno5 with `rtol=1e-6`, `atol=1e-9` and a 10,000-step
limit. All final benchmark calls succeeded and all saved population states
were finite and nonnegative.

DP14 is the vendored DallePezze 2014 model (37 store paths). The composite is
the current default `build_multi_hallmark_composite()` with Genomic
Instability 0.5, no nutrient-sensing intervention: DP14, GZ06, damage bridge,
p53-to-CDKN1A flux, irradiation pulse, and rapamycin drive (six processes,
43 store paths). It excludes the optional Proctor proteostasis extension.
Both resolve to one fully coupled continuous group, so the macro interval
does not introduce operator-splitting error in these runs.

For each workload, cells 0, 128, and 255 from the 256-cell batch were checked
against individual GPU and CPU solves at every saved time and store path.
The comparison uses `abs(error) <= 1e-6 + 1e-4 * abs(reference)`.

| Workload | Batch vs individual GPU: max absolute error | GPU vs CPU: max absolute error |
|---|---:|---:|
| DP14 | 3.41e-8 | 3.28e-8 |
| Composite | 8.58e-8 | 2.78e-8 |

An independent SciPy 1.18.0 Radau solve uses `rtol=1e-8`, `atol=1e-11`,
the fully coupled RHS, and explicit restarts at declared forcing jumps.
This validates the integrator against a different numerical method, but
shares the imported equations and therefore does not independently validate
SBML translation. Assigned columns are materialized before comparison.
The first selected cell is also rerun with tenfold tighter tolerances.
These checks use `abs(error) <= 1e-5 + 1e-3 * abs(reference)` on all saved
states, not just endpoints.

DP14 passes both checks: maximum absolute differences are 4.31e-4 against
the tighter solve and 4.09e-4 against Radau. The default-tolerance composite
fails this stricter criterion: maximum normalized errors are 2.08 and 2.34
times the allowed error. One example is p53 (`gz06/x`) at day 5.5:
0.210740 versus Radau's 0.210225. A tenfold tighter composite solve passes
against Radau (maximum 0.255 tolerance units). The final GPU composite is
therefore rerun with `rtol=1e-7`, `atol=1e-10`; its results are recorded
separately from the default-tolerance timing table.

**Final refined GPU simulation: passed.** All 256 cells completed in 51.057
seconds for one synchronized warm call (76.598 seconds for the cold call,
including compilation). Saved states are finite and nonnegative. Maximum
absolute differences: 5.36e-12 versus individual GPU solves, 1.05e-11 versus
CPU, 2.67e-4 versus the further-refined solve, and 2.99e-4 versus Radau.
The latter two comparisons reach only 0.227 and 0.255 tolerance units.
The first two comparisons cover three selected cells; the independent
integrator and refinement checks cover cell 0 over its full saved trajectory.

Local artifacts: [batch scaling](../outputs/gpu_verification/batch_scaling.png),
[final composite trajectories](../outputs/gpu_verification_refined/composite_trajectories.png),
[final population data](../outputs/gpu_verification_refined/composite_gpu.npz),
and [final validation results](../outputs/gpu_verification_refined/cpu.json).

These are numerical checks of a computational workload, not evidence of
biological validity. The deposited DP14 fitting workbook is not available
in this checkout, so its published chi-square fit was not revalidated.
The common 0.1-day output grid is not a validation of pulse extrema or
high-frequency spectral statistics.

## Regression suite and reproduction

The performance, multiscale, and stiffness-routing suites yielded 144
passes, one expected failure, and one independent failure in
`test_sbml_reimport_reuses_generated_class`. The latter reproduces in
isolation: repeated SBML imports have unequal pytree structures, despite
reusing the generated model class. It does not execute the scheduler.
The newly added plan-rank regression passes. `git diff --check` passes.

Run from the repository with GPU and SBML cache access:

```bash
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/verify_gpu_batch.py
JAX_PLATFORMS=cpu .venv/bin/python scripts/verify_gpu_batch.py --reference
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/verify_gpu_batch.py --models composite --sizes 256 --repeats 1 --rtol 1e-7 --atol 1e-10 --out outputs/gpu_verification_refined
JAX_PLATFORMS=cpu .venv/bin/python scripts/verify_gpu_batch.py --reference --models composite --rtol 1e-7 --atol 1e-10 --out outputs/gpu_verification_refined
```

The default-tolerance CPU command intentionally exits nonzero for the
composite accuracy finding above. JSON includes individual timings, solver
step counts, devices, shapes and comparison tolerances. Compressed NPZ
files include all 256 GPU trajectories, initial states, times, store keys,
and selected CPU/reference trajectories. Artifacts live under
`outputs/gpu_verification/` and `outputs/gpu_verification_refined/`.
