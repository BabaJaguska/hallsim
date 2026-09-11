"""Implicit-stage root finders usable through Scheduler's solver overrides."""

import optimistix as optx


class StepChord(optx.Chord):
    """Reuse Diffrax's per-step Jacobian with Chord's convergence criteria.

    Diffrax ESDIRK solvers supply a precomputed root-finder ``init_state``
    to reuse a Jacobian and its factorization across implicit stages.
    Optimistix Chord normally ignores that hint and initializes each stage
    independently. Honoring it here retains Chord's Cauchy termination rule,
    unlike Diffrax VeryChord's convergence-rate-based termination.

    This is opt-in: reuse can reduce large GPU batch costs but increase
    small-batch runtime or rejected steps on nonlinear stiff problems.
    Accuracy and throughput must be checked for the intended workload.
    There is no reuse across integration steps.

    Example::

        Scheduler(implicit_solver=dfx.Kvaerno5(
            root_finder=StepChord(rtol=1e-8, atol=1e-6)))
    """

    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        if "init_state" in options:
            return options["init_state"]
        return super().init(fn, y, args, options, f_struct, aux_struct, tags)
