"""The composite behind a page: compiled once, every setting solved once."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from hallsim.composite import Composite
from hallsim.handles import with_handles
from hallsim.scheduler import Scheduler
from hallsim.view._page import Lever, Page

log = logging.getLogger(__name__)


def _unfreeze_panel_species(composite, panels):
    """The composite with every panel species integrated: an import holds a
    species nothing reads at its initial value, and a panel reads it."""
    procs = dict(composite.processes)
    for name, proc in procs.items():
        frozen = getattr(proc, "_frozen_indices", ())
        if not frozen:
            continue
        held = {proc._species_names[i] for i in frozen}
        wanted = {
            p.split("/", 1)[1]
            for row in panels.values()
            for pn in row
            for p in pn.paths
            if p.startswith(f"{name}/")
        }
        if held & wanted:
            procs[name] = proc.with_unfrozen(*sorted(held & wanted))
    if all(procs[n] is composite.processes[n] for n in procs):
        return composite
    return Composite(
        processes=procs,
        topology=composite.topology,
        validate=False,
        semantic_validation=False,
    )


def series(ys, shown, cols):
    """One panel's series: the summed paths over the shown times, any cell
    axis kept."""
    return ys[shown][..., cols].sum(axis=-1)


class ViewModel:
    """One variant of a page's composite, compiled once.

    ``solve(severities)`` is the deterministic trajectory, memoised on the
    severity vector. ``population(severities)`` runs the page's reaction-level
    processes as ``cells`` members on the same grid under common random
    numbers, sampled once per setting and kept; the presets are sampled at
    construction on a thread.
    """

    def __init__(self, page: Page, variant: str | None = None):
        self.page = page
        self.variant = variant or page.first
        self.base = _unfreeze_panel_species(
            page.variants[self.variant], page.panels
        )
        self.registry = page.registry
        self.keys = list(self.base.store_keys())
        index = {k: i for i, k in enumerate(self.keys)}
        wanted = [
            p for row in page.panels.values() for pn in row for p in pn.paths
        ]
        missing = [p for p in wanted if p not in index]
        if missing:
            raise KeyError(f"panel paths not in the composite: {missing}")
        self.columns = {
            name: [np.array([index[p] for p in pn.paths]) for pn in row]
            for name, row in page.panels.items()
        }
        self.n_cells = int(page.cells) if page.population else 0
        self.seed = int(page.seed)
        self._sampling: set[tuple] = set()
        self._sample_lock = threading.Lock()
        self._scheduler = Scheduler()
        self._solve = jax.jit(self._trajectory)
        t0 = time.perf_counter()
        self.ts, self.reference = self.solve(page.presets[page.reference])
        self.compile_seconds = time.perf_counter() - t0
        self._populations: dict[tuple, tuple] = {}
        self.population_compile_seconds = 0.0
        if self.n_cells:
            procs = dict(self.base.processes)
            for name in page.population:
                procs[name] = procs[name].as_stochastic()
            self.stochastic_base = Composite(
                processes=procs,
                topology=self.base.topology,
                validate=False,
                semantic_validation=False,
            )
            self._presets = threading.Thread(
                target=self._sample_presets, daemon=True
            )
            self._presets.start()
        log.info(
            "%s ready: %d states, compiled in %.1f s; %d-cell presets "
            "sampling in the background",
            self.variant,
            len(self.keys),
            self.compile_seconds,
            self.n_cells,
        )

    def _sample_presets(self) -> None:
        t0 = time.perf_counter()
        for preset in self.page.presets.values():
            self.population(preset)
        self.population_compile_seconds = time.perf_counter() - t0
        log.info(
            "%s: %d-cell presets sampled in %.1f s",
            self.variant,
            self.n_cells,
            self.population_compile_seconds,
        )

    def wait_for_presets(self) -> None:
        thread = getattr(self, "_presets", None)
        if thread is not None:
            thread.join()

    def presets_ready(self) -> bool:
        return self.reference_population is not None

    @property
    def reference_population(self) -> np.ndarray | None:
        cached = self.population_cached(self.page.presets[self.page.reference])
        return None if cached is None else cached[0]

    def _run_kwargs(self) -> dict:
        page = self.page
        kw = dict(t_span=(-page.preroll, page.t_end), macro_dt=page.macro_dt)
        if page.save_dt is not None:
            kw["save_dt"] = page.save_dt
        return kw

    def _with_severities(self, base, severities):
        if not self.page.levers:
            return base
        return with_handles(
            base, self.page.severities(severities), registry=self.registry
        )

    def _trajectory(self, severities):
        comp = self._with_severities(self.base, severities)
        res = self._scheduler.run(
            comp, y0=comp.initial_state_vec(), **self._run_kwargs()
        )
        return res.ts, res.ys

    def _population(self, severities, key):
        comp = self._with_severities(self.stochastic_base, severities)
        y0 = comp.initial_state_vec()
        res = self._scheduler.run(
            comp,
            y0=jnp.tile(y0[None], (self.n_cells, 1)),
            key=key,
            **self._run_kwargs(),
        )
        events = jnp.stack(
            [res.stats[name]["num_events"] for name in self.page.population]
        )
        return res.ys, events

    @lru_cache(maxsize=32)
    def _solve_cached(self, severities: tuple):
        ts, ys = self._solve(jnp.asarray(severities, dtype=jnp.float64))
        return np.asarray(ts), np.asarray(ys)

    def solve(self, severities):
        """``(ts, ys)`` as numpy, ``ys`` being ``(n_time, n_vars)``."""
        return self._solve_cached(self.key(severities))

    @staticmethod
    def key(severities) -> tuple:
        return tuple(round(float(s), 6) for s in severities)

    def population_cached(self, severities):
        return self._populations.get(self.key(severities))

    def population_async(self, severities):
        """The sample at these severities if it is taken, else ``None``,
        having started it on a host thread."""
        key = self.key(severities)
        cached = self._populations.get(key)
        if cached is not None:
            return cached
        with self._sample_lock:
            if key in self._sampling:
                return None
            self._sampling.add(key)

        def take():
            try:
                self.population(key)
            finally:
                with self._sample_lock:
                    self._sampling.discard(key)

        threading.Thread(target=take, daemon=True).start()
        return None

    def population(self, severities):
        """``(ys, events_per_cell)``: ``ys`` is ``(n_time, n_cells,
        n_vars)``. Sampled once per setting and kept."""
        key = self.key(severities)
        if key not in self._populations:
            ys, events = self._population(
                jnp.asarray(key, dtype=jnp.float64),
                jax.random.PRNGKey(self.seed),
            )
            self._populations[key] = (
                np.asarray(ys),
                float(np.asarray(events).mean()),
            )
        return self._populations[key]

    def moved(self, lever: Lever, severity: float) -> list[dict]:
        """What the lever set on this composite: ``{target, published,
        now, description}`` per parameter, published being the value at
        zero severity."""
        handle = self.registry[lever.handle]
        procs = self.base.processes
        present = [m for m in handle.mappings if m.process_name in procs]
        at_zero = handle.summary(0.0, procs)
        at_now = handle.summary(float(severity), procs)
        return [
            {
                "target": f"{m.process_name} · {m.param_name.split('.')[-1]}",
                "published": float(
                    at_zero[f"{m.process_name}.{m.param_name}"]
                ),
                "now": float(at_now[f"{m.process_name}.{m.param_name}"]),
                "description": m.description,
            }
            for m in present
        ]


class ModelBank:
    """One :class:`ViewModel` per variant. The first is built here, the rest
    on a background thread; a variant still compiling reports ``None``."""

    def __init__(self, page: Page):
        self.page = page
        self.variants = list(page.variants)
        self._models: dict[str, ViewModel] = {}
        self.first = page.first
        self._models[self.first] = ViewModel(page, self.first)
        self._thread = threading.Thread(target=self._build_rest, daemon=True)
        self._thread.start()

    def _build_rest(self):
        rest = [v for v in self.variants if v not in self._models]
        if not rest:
            return
        with ThreadPoolExecutor(max_workers=len(rest)) as pool:
            pending = {pool.submit(ViewModel, self.page, v): v for v in rest}
            for future in as_completed(pending):
                self._models[pending[future]] = future.result()

    def get(self, name: str) -> ViewModel | None:
        return self._models.get(name)

    def ready(self, name: str) -> bool:
        return name in self._models

    def wait(self):
        self._thread.join()
        for model in self._models.values():
            model.wait_for_presets()
