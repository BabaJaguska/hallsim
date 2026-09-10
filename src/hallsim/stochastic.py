"""Reaction-level stochastic simulation through imported SBML processes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class SSAResult:
    """Event trajectory and regularly sampled state trajectory."""

    times: jnp.ndarray
    states: jnp.ndarray
    reaction_indices: jnp.ndarray


class SSACompatibilityError(ValueError):
    """Raised when an imported reaction network is not SSA-compatible."""


def validate_ssa_process(process) -> None:
    """Check structural and initial-state conditions required by direct SSA."""
    channels = tuple(process.reaction_channels())
    if not channels:
        raise SSACompatibilityError("process has no reaction channels")
    for channel in channels:
        for _, coefficient in channel.stoichiometry:
            if not float(coefficient).is_integer():
                raise SSACompatibilityError(
                    f"reaction {channel.reaction_id!r} has non-integer "
                    f"stoichiometry {coefficient!r}"
                )
    initial = tuple(float(value) for value in process._species_y0)
    if any(not math.isfinite(value) or value < 0 for value in initial):
        raise SSACompatibilityError(
            "SSA requires finite, non-negative initial molecule counts"
        )


def _save_grid(start: float, end: float, save_dt: float | None) -> jnp.ndarray:
    if save_dt is None:
        return jnp.asarray([start, end])
    count = int(math.floor((end - start) / save_dt + 1e-12))
    times = start + save_dt * jnp.arange(count + 1, dtype=float)
    if count == 0 or float(times[-1]) < end:
        times = jnp.concatenate((times, jnp.asarray([end])))
    else:
        times = times.at[-1].set(end)
    return times


def _stoichiometry(process, species):
    species_index = {name: i for i, name in enumerate(species)}
    matrix = []
    for channel in process.reaction_channels():
        row = [0.0] * len(species)
        for name, coefficient in channel.stoichiometry:
            if name in species_index:
                row[species_index[name]] += float(coefficient)
        matrix.append(row)
    return jnp.asarray(matrix).T


def ssa_step_jax(
    process,
    *,
    t_span: tuple[float, float],
    state: jnp.ndarray,
    key,
    max_events: int = 100_000,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Advance one reaction network over a macro window in JAX.

    Returns ``(state, key, num_events)``. The event capacity is static so the
    complete step can be embedded in a compiled scheduler scan.
    """
    species = tuple(process._species_names)
    channels = tuple(process.reaction_channels())
    stoich = _stoichiometry(process, species)
    start, end = t_span

    def condition(carry):
        time, _, _, n_events, active = carry
        return active & (time < end) & (n_events < max_events)

    def body(carry):
        time, state, key, n_events, active = carry
        key, wait_key, reaction_key = jax.random.split(key, 3)
        rates = jnp.asarray(
            process.reaction_propensities(
                time,
                {name: state[i] for i, name in enumerate(species)},
            )
        )
        total = jnp.sum(rates)
        valid = jnp.all(jnp.isfinite(rates)) & jnp.all(rates >= 0)
        wait = jax.random.exponential(wait_key) / jnp.maximum(total, 1e-30)
        next_time = time + wait
        can_fire = valid & (total > 0) & (next_time < end)
        reaction = jax.random.choice(
            reaction_key,
            len(channels),
            p=rates / jnp.maximum(total, 1e-30),
        )
        return (
            jnp.where(can_fire, next_time, time),
            jnp.where(can_fire, state + stoich[:, reaction], state),
            key,
            n_events + can_fire.astype(jnp.int32),
            can_fire,
        )

    return jax.lax.while_loop(
        condition,
        body,
        (jnp.asarray(start), state, key, jnp.asarray(0, jnp.int32), True),
    )


def _simulate_ssa_jax(
    process,
    *,
    start: float,
    end: float,
    state: jnp.ndarray,
    save_times: jnp.ndarray,
    key,
    max_events: int,
) -> SSAResult:
    species = tuple(process._species_names)
    channels = tuple(process.reaction_channels())
    stoich = _stoichiometry(process, species)
    n_reactions = len(channels)
    event_times = jnp.zeros((max_events,), dtype=state.dtype)
    event_indices = jnp.zeros((max_events,), dtype=jnp.int32)
    saved = jnp.zeros((save_times.shape[0], state.shape[0]), dtype=state.dtype)
    saved = saved.at[0].set(state)
    save_index = jnp.asarray(0, dtype=jnp.int32)

    def condition(carry):
        time, _, _, n_events, _, _, _, save_index, active = carry
        return active & (time < end) & (n_events < max_events)

    def body(carry):
        (
            time,
            state,
            key,
            n_events,
            event_times,
            event_indices,
            saved,
            save_index,
            active,
        ) = carry
        key, wait_key, reaction_key = jax.random.split(key, 3)
        rates = jnp.asarray(
            process.reaction_propensities(
                time, {name: state[i] for i, name in enumerate(species)}
            )
        )
        total = jnp.sum(rates)
        valid = jnp.all(jnp.isfinite(rates)) & jnp.all(rates >= 0)
        wait = jax.random.exponential(wait_key) / jnp.maximum(total, 1e-30)
        next_time = time + wait
        can_fire = valid & (total > 0) & (next_time < end)
        probabilities = rates / jnp.maximum(total, 1e-30)
        reaction = jax.random.choice(
            reaction_key, n_reactions, p=probabilities
        )
        next_state = state + stoich[:, reaction]
        next_events = n_events + can_fire.astype(jnp.int32)
        event_times = event_times.at[n_events].set(next_time)
        event_indices = event_indices.at[n_events].set(
            jnp.asarray(reaction, dtype=jnp.int32)
        )
        crossed = jnp.searchsorted(save_times, next_time, side="right")
        crossed = jnp.minimum(crossed, save_times.shape[0] - 1)
        grid = jnp.arange(save_times.shape[0])
        fill = (grid > save_index) & (grid <= crossed)
        saved = jnp.where(fill[:, None], next_state[None, :], saved)
        return (
            jnp.where(can_fire, next_time, time),
            jnp.where(can_fire, next_state, state),
            key,
            next_events,
            event_times,
            event_indices,
            saved,
            jnp.where(can_fire, crossed, save_index),
            can_fire,
        )

    (
        time,
        state,
        _,
        n_events,
        event_times,
        event_indices,
        saved,
        save_index,
        active,
    ) = jax.lax.while_loop(
        condition,
        body,
        (
            start,
            state,
            key,
            0,
            event_times,
            event_indices,
            saved,
            save_index,
            True,
        ),
    )
    grid = jnp.arange(save_times.shape[0])
    saved = jnp.where((grid > save_index)[:, None], state[None, :], saved)
    del active
    return SSAResult(
        times=save_times,
        states=saved,
        reaction_indices=event_indices[:n_events],
    )


def _simulate_ssa_with_provider(
    process,
    *,
    start: float,
    end: float,
    state: dict[str, float],
    save_times: jnp.ndarray,
    key,
    max_events: int,
    input_provider,
) -> SSAResult:
    """Eager bridge for Python callbacks driving one-way hybrid inputs."""
    species = tuple(process._species_names)
    channels = tuple(process.reaction_channels())
    stoich = _stoichiometry(process, species)
    event_times = []
    event_indices = []
    saved = []
    save_index = 0
    time = start
    saved.append([state[name] for name in species])

    while time < end and len(event_indices) < max_events:
        updates = input_provider(time, dict(state))
        if updates is not None:
            if not isinstance(updates, dict):
                raise TypeError("input_provider must return a dict or None")
            state.update(
                {name: float(value) for name, value in updates.items()}
            )
        rates = jnp.asarray(process.reaction_propensities(time, state))
        total = float(jnp.sum(rates))
        if not bool(jnp.all(jnp.isfinite(rates))) or bool(jnp.any(rates < 0)):
            raise ValueError("SSA encountered an invalid reaction propensity")
        if total == 0:
            break
        key, wait_key, reaction_key = jax.random.split(key, 3)
        time += float(jax.random.exponential(wait_key) / total)
        if time >= end:
            break
        reaction = int(
            jax.random.choice(reaction_key, len(channels), p=rates / total)
        )
        for i, name in enumerate(species):
            state[name] += float(stoich[i, reaction])
        event_times.append(time)
        event_indices.append(reaction)
        while (
            save_index + 1 < len(save_times)
            and float(save_times[save_index + 1]) <= time
        ):
            save_index += 1
            saved.append([state[name] for name in species])

    while len(saved) < len(save_times):
        saved.append([state[name] for name in species])
    return SSAResult(
        times=save_times,
        states=jnp.asarray(saved),
        reaction_indices=jnp.asarray(event_indices, dtype=jnp.int32),
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
    """Run Gillespie's direct method with JAX-native event execution."""
    if len(t_span) != 2 or t_span[1] < t_span[0]:
        raise ValueError("t_span must be an increasing (start, end) pair")
    if save_dt is not None and (not math.isfinite(save_dt) or save_dt <= 0):
        raise ValueError("save_dt must be positive and finite")
    if max_events < 0:
        raise ValueError("max_events must be non-negative")

    validate_ssa_process(process)
    species = tuple(process._species_names)
    channels = tuple(process.reaction_channels())
    if len(process._reaction_propensity_functions) != len(channels):
        raise ValueError("reaction channels and propensity functions disagree")
    initial = (
        process._species_y0
        if y0 is None
        else tuple(y0[name] for name in species)
    )
    values = tuple(float(value) for value in initial)
    if any(
        not math.isfinite(value) or value < 0 or not value.is_integer()
        for value in values
    ):
        raise SSACompatibilityError(
            "SSA initial state must contain finite, non-negative integer "
            "molecule counts"
        )
    start, end = map(float, t_span)
    save_times = _save_grid(start, end, save_dt)
    key = jax.random.PRNGKey(seed)
    initial_state = dict(zip(species, values))
    rates = jnp.asarray(process.reaction_propensities(start, initial_state))
    if rates.shape != (len(channels),):
        raise ValueError(
            "reaction_propensities must return one rate per reaction"
        )
    if not bool(jnp.all(jnp.isfinite(rates))):
        raise ValueError("SSA encountered a non-finite reaction propensity")
    if bool(jnp.any(rates < 0)):
        raise ValueError("SSA encountered a negative reaction propensity")
    if input_provider is not None:
        return _simulate_ssa_with_provider(
            process,
            start=start,
            end=end,
            state=initial_state,
            save_times=save_times,
            key=key,
            max_events=max_events,
            input_provider=input_provider,
        )
    return _simulate_ssa_jax(
        process,
        start=start,
        end=end,
        state=jnp.asarray(values),
        save_times=save_times,
        key=key,
        max_events=max_events,
    )
