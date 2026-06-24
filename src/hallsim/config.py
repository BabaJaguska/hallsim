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

# Relative / absolute tolerance for the adaptive step-size controller.
# Oscillatory biology (p53-Mdm2, NF-kB, cell cycle) is accuracy-limited,
# so the tolerance is tight; loosening it without screening every
# oscillator first risks numerical anti-damping.
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-9

# Per-state vector atol coefficient: atol_i = max(DEFAULT_ATOL,
# DEFAULT_ATOL_SCALE * |y_i|). Loosens the absolute floor on
# large-magnitude states (molecule counts) that would otherwise force
# stability-tiny steps, while keeping a tight floor near zero.
DEFAULT_ATOL_SCALE = 1e-6

# Safety ceiling on solver steps per macro step. Sized for the
# second-scale t_span values some SBML composites run at; far above any
# healthy integration, it only fires on genuinely runaway dynamics.
DEFAULT_MAX_STEPS = 4_000_000

# Initial step size handed to the adaptive controller.
DEFAULT_DT0 = 1e-3

# Stiffness diagnostic threshold. The stiffness index ``spectral_abscissa
# × dt`` is the number of stability-limited substeps an explicit method
# would be forced to take across one solve interval (its step is bounded
# by ``Δt ≲ 2/|λ|``). Below this many, explicit integration is cheap and
# robust; far above it, the explicit step is stability- not
# accuracy-limited. Canonical cases separate by orders of magnitude: a
# mildly multiscale but slow oscillator (ERiQ, index ~3–26) sits well
# below; a fast dissipative subsystem (DallePezze 2014 mitochondria
# λ≈-3e5, Ihekwaba NF-κB λ≈-1.7e4, index ~1e4–1e6) sits far above. 100
# leaves a wide margin on both sides.
DEFAULT_MAX_EXPLICIT_SUBSTEPS = 100.0
