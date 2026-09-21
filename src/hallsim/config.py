"""Integration defaults — the single source of truth.

Every entry point that integrates an ODE (the :class:`~hallsim.scheduler.Scheduler`
and the :mod:`~hallsim.diagnostics` screen) reads its tolerances, step
budget, and initial step from here. Keeping them in one module is what
guarantees the screen tests exactly what production runs — divergent
hardcoded defaults are how a model can pass production yet fail (or
falsely fail) the screen.

Override per run by passing the argument explicitly
(``Scheduler(max_steps=...)``); change the global default by editing the
value here.
"""

# Step-controller tolerances; tight, because oscillatory biology is
# accuracy-limited (loosen only after screening every oscillator).
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-9

# Algebraic convergence tolerance of an implicit stage's Newton solve — not an
# accuracy target, and deliberately not DEFAULT_ATOL (docs/diary.md).
DEFAULT_NEWTON_ATOL = 1e-6

# Ceiling on solver steps per macro step; fires only on runaway dynamics.
DEFAULT_MAX_STEPS = 4_000_000

# Solver-ladder step budgets: past these a group moves to the next rung
# (explicit -> Kvaerno5 -> Kvaerno3). A healthy solve is thousands of steps.
LADDER_STEP_BUDGET = 100_000
EXPLICIT_STEP_BUDGET = 500_000

# First adaptive step; None estimates it from the field (Hairer's rule).
DEFAULT_DT0 = None

# Where XLA caches compiled executables between processes. Set
# HALLSIM_COMPILATION_CACHE_DIR to relocate it, or to "" / "0" / "off" to
# disable. Caches codegen only; tracing and lowering are Python and still run.
DEFAULT_COMPILATION_CACHE_DIR = "~/.cache/hallsim/jax"

# Minimum compile time worth caching: zero, a run's cost is many small ones.
DEFAULT_COMPILATION_CACHE_MIN_SECS = 0.0

# Stiff when spectral_abscissa x dt, the explicit substeps one interval would
# force, exceeds this; canonical cases sit orders of magnitude either side.
DEFAULT_MAX_EXPLICIT_SUBSTEPS = 100.0


def enable_compilation_cache(directory: str | None = None) -> str | None:
    """Point XLA's persistent compilation cache at ``directory``.

    Returns the path in use, or ``None`` when disabled. Called at package
    import; safe to call again with an explicit path.
    """
    import os

    raw = (
        directory
        if directory is not None
        else os.environ.get(
            "HALLSIM_COMPILATION_CACHE_DIR", DEFAULT_COMPILATION_CACHE_DIR
        )
    )
    if raw.strip().lower() in ("", "0", "off", "false", "none"):
        return None

    import jax

    path = os.path.abspath(os.path.expanduser(raw))
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        # A read-only or unwritable HOME is not a reason to fail an import.
        return None
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs",
        DEFAULT_COMPILATION_CACHE_MIN_SECS,
    )
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return path
