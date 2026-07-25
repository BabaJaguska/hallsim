# One-off checks

Auditable scripts behind claims in `docs/diary.md`. Each answers one question
and prints the numbers it was run for, so a claim can be re-derived rather
than taken on trust. They are not library code and nothing imports them.

Run from the repo root with the project venv:

    .venv/bin/python scripts/checks/<name>.py

| script | question | answer on 2026-07-24 |
|---|---|---|
| `check_p53_period.py` | is the composite's p53 period 0.30 d, or is that the save grid? | 0.2842 d = 6.82 h; 0.30 d was 15 x `save_dt` snapping |
| `check_equilibrate_seeding.py` | does Newton need a forward pre-solve to find the physical fixed point? | yes — unseeded gives 6 negative NF-kB species; a 20 d seed gives 0 |
| `check_ctrl_stability.py` | is the ctrl fixed point stable, or does NF-kB limit-cycle? | stable: 0 eigenvalues with Re > 0, and 5 d from the fixed point moves 8.5e-5 |
| `check_cell_baseline.py` | should the population's IC jitter be applied before or after equilibration? | before — after leaves cells at residual 2.6e4 instead of 1e-11 |
| `check_tolerance.py` | are the DDIS reporter readouts solver-dependent? | no: 100x tighter tolerance moves them <= 7.1e-5 |
| `check_precision.py` | is the population run really float64, and what does the GPU give? | float64 end to end; T4 measures 0.250 TFLOP/s f64 vs 4.94 f32 |
| `check_float32.py` | can the *forward* solve run in float32? | run `f64` then `f32`; unresolved (see diary) |

`check_float32.py` overrides `jax_enable_x64` **after** importing hallsim,
because `hallsim/__init__.py` force-enables it at import — that override is
the only way to run the framework in single precision today.
