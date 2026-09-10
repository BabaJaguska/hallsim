"""Reaction-level stochastic simulation through imported SBML processes.

This module intentionally starts with a single-process direct SSA runner. It
does not reinterpret an ODE derivative as birth and death channels: reaction
stoichiometry and source kinetic laws come from :class:`SBMLProcess`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SSAResult:
    """Event trajectory and regularly sampled state trajectory."""

    times: np.ndarray
    states: np.ndarray
    reaction_indices: np.ndarray


class SSACompatibilityError(ValueError):
    """Raised when an imported reaction network is not SSA-compatible."""


def validate_ssa_process(process) -> None:
    """Check structural and initial-state conditions required by direct SSA.

    Species absent from ``_species_names`` are allowed in a channel: SBML
    boundary species and importer-frozen sink species are held externally and
    therefore contribute to a propensity without being updated by SSA.
    """
    channels = tuple(process.reaction_channels())
    if not channels:
        raise SSACompatibilityError("process has no reaction channels")
    species = tuple(process._species_names)
    for channel in channels:
        for _, coefficient in channel.stoichiometry:
            if not float(coefficient).is_integer():
                raise SSACompatibilityError(
                    f"reaction {channel.reaction_id!r} has non-integer "
                    "stoichiometry {coefficient!r}"
                )
    initial = {
        name: float(value) for name, value in zip(species, process._species_y0)
    }
    if any(not np.isfinite(value) or value < 0 for value in initial.values()):
        raise SSACompatibilityError(
            "SSA requires finite, non-negative initial molecule counts"
        )


def simulate_ssa(
    process,
    *,
    t_span: tuple[float, float],
    y0: dict[str, float] | None = None,
    save_dt: float | None = None,
    seed: int = 0,
    max_events: int = 10_000_000,
    input_provider=None,
) -> SSAResult:
    """Run Gillespie's direct method for one imported reaction network.

    The process must expose ``reaction_channels()``,
    ``reaction_propensities(t, state)``, and ``_species_names`` as supplied by
    :class:`hallsim.sbml_import.SBMLProcess`.  This is deliberately not wired
    into :class:`Scheduler` yet: hybrid coupling needs an explicit contract
    for how deterministic store values drive stochastic propensities.  For a
    one-way hybrid, ``input_provider(t, state)`` may return externally driven
    state values before each propensity evaluation.
    """
    if len(t_span) != 2 or t_span[1] < t_span[0]:
        raise ValueError("t_span must be an increasing (start, end) pair")
    if (
        save_dt is not None
        and not np.isfinite(save_dt)
        or (save_dt is not None and save_dt <= 0)
    ):
        raise ValueError("save_dt must be positive and finite")
    if max_events < 0:
        raise ValueError("max_events must be non-negative")

    validate_ssa_process(process)
    channels = tuple(process.reaction_channels())
    species = tuple(process._species_names)
    if not channels:
        raise ValueError("process has no reaction channels")
    if len(process._reaction_propensity_functions) != len(channels):
        raise ValueError("reaction channels and propensity functions disagree")

    state = {
        name: float(value)
        for name, value in zip(
            species,
            (
                process._species_y0
                if y0 is None
                else (y0[name] for name in species)
            ),
        )
    }
    if any(
        not np.isfinite(value) or value < 0 or not float(value).is_integer()
        for value in state.values()
    ):
        raise SSACompatibilityError(
            "SSA initial state must contain finite, non-negative integer "
            "molecule counts"
        )
    rng = np.random.default_rng(seed)
    start, end = map(float, t_span)
    event_time = start
    event_indices: list[int] = []
    event_times: list[float] = []

    if save_dt is None:
        save_times = np.asarray([start, end], dtype=float)
    else:
        count = int(np.floor((end - start) / save_dt + 1e-12))
        save_times = start + save_dt * np.arange(count + 1, dtype=float)
        if save_times.size == 0 or save_times[-1] < end:
            save_times = np.concatenate((save_times, np.asarray([end])))
        else:
            save_times[-1] = end
    saved = np.empty((len(save_times), len(species)), dtype=float)
    save_index = 0
    saved[save_index] = [state[name] for name in species]

    while event_time < end:
        if input_provider is not None:
            updates = input_provider(event_time, dict(state))
            if updates is None:
                updates = {}
            if not isinstance(updates, dict):
                raise TypeError("input_provider must return a dict or None")
            state.update(
                {name: float(value) for name, value in updates.items()}
            )
        rates = np.asarray(
            process.reaction_propensities(event_time, state), dtype=float
        )
        if rates.shape != (len(channels),):
            raise ValueError(
                "reaction_propensities must return one rate per reaction"
            )
        if not np.all(np.isfinite(rates)):
            raise ValueError(
                "SSA encountered a non-finite reaction propensity"
            )
        if np.any(rates < 0):
            raise ValueError("SSA encountered a negative reaction propensity")
        total = float(rates.sum())
        if total == 0:
            break
        event_time += float(rng.exponential(1.0 / total))
        if event_time >= end:
            break
        reaction_index = int(rng.choice(len(channels), p=rates / total))
        for species_name, coefficient in channels[
            reaction_index
        ].stoichiometry:
            if species_name in state:
                state[species_name] += float(coefficient)
        event_times.append(event_time)
        event_indices.append(reaction_index)
        if len(event_indices) > max_events:
            raise RuntimeError(
                f"SSA exceeded max_events={max_events} before t={event_time:g}"
            )
        while save_index + 1 < len(save_times) and (
            save_times[save_index + 1] <= event_time
        ):
            save_index += 1
            saved[save_index] = [state[name] for name in species]

    while save_index + 1 < len(save_times):
        save_index += 1
        saved[save_index] = [state[name] for name in species]

    return SSAResult(
        times=save_times,
        states=saved,
        reaction_indices=np.asarray(event_indices, dtype=int),
    )
