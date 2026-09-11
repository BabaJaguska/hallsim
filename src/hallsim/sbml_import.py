"""SBML auto-import — convert BioModels SBML files into Process instances.

The model's math is compiled by :mod:`hallsim.sbml_core` (libsbml → sympy →
JAX); this module wraps the compiled core as a :class:`Process` with
auto-generated ports and metadata.

Example
-------
>>> proc = process_from_sbml(10, name="mapk_cascade")
>>> proc.ports_schema()    # auto-generated from SBML species
>>> proc.metadata()        # SBML annotations
"""

from __future__ import annotations

import logging
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from hallsim.imported import ImportedODEProcess
from hallsim.process import Port, PortRole, ReactionChannel
from hallsim.sbml_core import (  # noqa: F401  (re-exported)
    SBMLCore,
    UnsupportedSBMLFeatureError,
    compile_sbml,
)

log = logging.getLogger(__name__)


SBMLReactionChannel = ReactionChannel


class SBMLProcess(ImportedODEProcess):
    """Process built on a compiled SBML model, exposing
    SBML species as EVOLVED ports. Built by :func:`process_from_sbml`, not
    directly.

    The inherited ``parameters`` field is the substitutable surface for every
    SBML ``<parameter>`` and constant-rate species, auto-populated at import
    with published defaults — so the full mechanism surface is discoverable via
    :meth:`calibratable_params`. Hallmarks and Calibrator substitute into it
    through a dotted ``parameters.<key>`` path.
    """

    _param_label = "SBML constant"

    # Everything below is structure, not fitted values: static, so
    # ports_schema() stays concrete under a trace and these round-trip
    # untouched through eqx.tree_at substitutions on `parameters`.
    _species_names: tuple[str, ...] = eqx.field(static=True, default=())
    _species_y0: tuple[float, ...] = eqx.field(static=True, default=())
    # ``((species_id, display_name), ...)``. A CellDesigner export — a large
    # part of BioModels — gives every species a UUID id and puts the gene
    # symbol in the name, so matching on id alone cannot find IL6.
    _species_labels: tuple = eqx.field(static=True, default=())
    _species_ontology: tuple[dict[str, str], ...] = eqx.field(
        static=True, default=()
    )
    # Param constancy + SBO, dynamic variables, and the assignment-rule graph,
    # so a driver aimed at a rate constant the model modulates via a rule can
    # be flagged (see hallsim.coupling_wiring).
    _coupling_meta: dict = eqx.field(static=True, default=None)
    # Species × reaction stoichiometry from the source SBML — the exact,
    # parameter-independent basis for conserved-moiety analysis.
    _stoichiometry: dict = eqx.field(static=True, default=None)
    _reaction_channels: tuple[SBMLReactionChannel, ...] = eqx.field(
        static=True, default=()
    )
    _stochastic_enabled: bool = eqx.field(static=True, default=False)
    _model: Any = None  # the compiled SBMLCore
    _w0: Any = None
    _c: Any = None
    # Parallel to _param_names, fixed at construction so derivative-time
    # lookup stays JIT-safe even when tree_at reorders the parameters dict.
    _param_indexes: tuple[int, ...] = eqx.field(static=True, default=())
    # Boundary-input species (Irradiation, Insulin) — the model's experimental
    # input ports. They live in the `w` vector but are exposed through the same
    # `parameters` surface; these route their values into `_w0`.
    _w_names: tuple[str, ...] = eqx.field(static=True, default=())
    _w_indexes: tuple[int, ...] = eqx.field(static=True, default=())
    # Quantities the SBML determines by an <assignmentRule> rather than by
    # integration. They live in the generated model's `w` vector, which
    # `derivative` already evaluates every step and then discarded: without a
    # port they are invisible (`geneProduct`), and a species that is also in
    # `y` reads its unchanging `y` slot instead of the rule (`CRP` held its
    # initial value for a whole run while the rule evaluated to 158). Surfaced
    # as ASSIGNED, which is the role for exactly this.
    _assigned_names: tuple[str, ...] = eqx.field(static=True, default=())
    _assigned_indexes: tuple[int, ...] = eqx.field(static=True, default=())
    # Static, like _species_y0: ports_schema() must stay concrete under a
    # trace, and reading them off the traced _w0 breaks that.
    _assigned_y0: tuple[float, ...] = eqx.field(static=True, default=())
    # Inert sinks: written by degradation, read by nothing. Frozen to dy/dt=0
    # so they can't accumulate unboundedly and wreck the state scaling — exact,
    # since no rate law reads them.
    _frozen_indices: tuple[int, ...] = eqx.field(static=True, default=())
    # Translated SBML <event> elements; expand with sbml_events.expand_events.
    # Compartment sizes live in `parameters` but are geometry, not mechanism;
    # excluded from the calibration surface.
    _compartment_names: frozenset = eqx.field(static=True, default=frozenset())
    _events: tuple = eqx.field(static=True, default=())
    # ``((input_name, input_port), ...)`` — boundary inputs driven from an
    # INPUT port, overriding their native SBML rule (:meth:`with_input_driver`).
    # Undriven inputs keep that rule, so a raw import reproduces the source
    # model's own experiment.
    _input_drivers: tuple = eqx.field(static=True, default=())
    # ``((param_name, t_step, value_before), ...)`` — a constant that holds
    # ``value_before`` until ``t_step``, for a timed intervention rather than a
    # severity applied across the whole trajectory.
    _param_steps: tuple = eqx.field(static=True, default=())
    # Species read from the store instead of integrated: the port keeps the
    # species' name and ontology and becomes INPUT, so another model's pool
    # can be wired to it (:meth:`with_species_input`).
    _species_inputs: tuple[str, ...] = eqx.field(static=True, default=())

    def coupling_structure(self) -> dict:
        """SBML equation structure for the coupling-wiring check (extracted at
        import; see :func:`_extract_coupling_metadata`)."""
        return self._coupling_meta

    def stoichiometry(self) -> dict | None:
        """Species × reaction ``N`` from the source SBML (see
        :func:`_extract_stoichiometry`). ``None`` when the model declares no
        reactions or a symbolic stoichiometry, where ``N`` would not be
        parameter-independent."""
        if not self._stoichiometry or not self._stoichiometry["species"]:
            return None
        return self._stoichiometry

    def reaction_channels(self) -> tuple[SBMLReactionChannel, ...]:
        """Return reaction stoichiometry and source rate laws.

        This is deliberately separate from :meth:`derivative`: an SBML
        reaction network can support a stochastic SSA interpretation, while
        the deterministic ODE remains the default execution mode.
        """
        return self._reaction_channels

    def as_stochastic(self) -> "SBMLProcess":
        """Return a copy selecting reaction-level execution in Scheduler."""
        import copy

        new = copy.copy(self)
        object.__setattr__(new, "_stochastic_enabled", True)
        return new

    def reaction_propensities(self, t, state):
        """Evaluate source reaction rates without collapsing them into ``dy``.

        The returned vector follows :meth:`reaction_channels` order.  These
        values are the imported SBML kinetic laws; an SSA caller must still
        validate that the source law has the molecule-count units and
        combinatorial interpretation required for a propensity.
        """
        y = jnp.stack([state[name] for name in self._species_names])
        c = self._constants(t)
        if self._param_drivers:
            driven = self._driven_param_values(state)
            names = list(driven)
            indexes = jnp.asarray(
                [
                    self._param_indexes[self._param_names.index(n)]
                    for n in names
                ]
            )
            c = c.at[indexes].set(jnp.stack([driven[n] for n in names]))
        t_native = t * self.time_scale
        w = self._model.assignmentfunc(y, self._w0, c, t_native)
        if self._input_drivers:
            name_to_widx = dict(zip(self._w_names, self._w_indexes))
            for input_name, port in self._input_drivers:
                w = w.at[name_to_widx[input_name]].set(state[port])
        values = self._model.reaction_velocities(y, w, c, t_native)
        return values * self.time_scale

    def with_param_step(
        self, param_name: str, t_step: float, value_before: float
    ) -> "SBMLProcess":
        """Return a copy whose SBML constant ``param_name`` steps at
        ``t_step``: it holds ``value_before`` while ``t < t_step`` and its
        configured ``parameters[param_name]`` value once ``t >= t_step``.
        ``t_step`` is in composite time. Use for a timed pharmacological
        intervention where the pre-intervention level differs from the
        (severity-set) post-intervention level."""
        if param_name not in self._param_names:
            raise KeyError(
                f"{param_name!r} is not an SBML constant on {self._name!r}; "
                f"available: {sorted(self._param_names)}"
            )
        import copy

        new = copy.copy(self)
        object.__setattr__(
            new,
            "_param_steps",
            self._param_steps
            + ((param_name, float(t_step), float(value_before)),),
        )
        return new

    def with_input_driver(
        self, input_name: str, input_port: str
    ) -> "SBMLProcess":
        """Return a copy that drives boundary input ``input_name`` from an
        INPUT port ``input_port`` each step, overriding its native SBML rule.
        This is the general port-coupling path for boundary inputs — the
        ``w``-vector analogue of :meth:`with_param_input`. Wire ``input_port``
        via topology to a forcing source
        (:class:`hallsim.models.forcing.PulseSource`) or another model's
        state; undriven inputs keep their native SBML drive. A prescribed dose
        (pulse/ramp) is composed, not special-cased — see
        :func:`hallsim.models.forcing.drive_pulse`."""
        if input_name not in self._w_names:
            raise KeyError(
                f"{input_name!r} is not a boundary input on {self._name!r}; "
                f"available: {sorted(self._w_names)}"
            )
        import copy

        new = copy.copy(self)
        object.__setattr__(
            new,
            "_input_drivers",
            self._input_drivers + ((input_name, input_port),),
        )
        return new

    def with_unfrozen(self, *species: str):
        """Copy that integrates ``species`` normally instead of holding them
        at their initial value.

        Import freezes species nothing reads back, which catches unbounded
        degradation counters and a model's terminal products alike::

            k14 = process_from_sbml(524).with_unfrozen("tBid")

        A Composite lifts the freeze on its own for any frozen species another
        process reads; this is for the rest — plotting, reporters, scoring.
        """
        unknown = [s for s in species if s not in self._species_names]
        if unknown:
            raise KeyError(
                f"{unknown} are not species on {self._name!r}; "
                f"available: {sorted(self._species_names)}"
            )
        drop = {self._species_names.index(s) for s in species}
        import copy

        new = copy.copy(self)
        object.__setattr__(
            new,
            "_frozen_indices",
            tuple(i for i in self._frozen_indices if i not in drop),
        )
        return new

    def with_species_input(self, *species: str) -> "SBMLProcess":
        """Copy that reads ``species`` from the store instead of integrating
        them. Each port keeps the species' name and ontology and becomes
        INPUT, so the topology can point it at another model's pool::

            p07 = process_from_sbml(105).with_species_input("ROS")
            topology["ros_link"] = {"source": "dp14/ROS", "signal": "p07/ROS"}

        This is the composition move for one entity that two deposits both
        carry — SBML comp's replaced element: one model owns the pool, this
        one reads it, and a level edge between them holds the conversion
        factor. Every rate law here that reads the species sees the external
        value, and the species' own reactions stop moving anything, which is
        what handing the pool over means. Unwired, the port defaults to the
        species' published initial value, so a solo run of a species the
        deposit held constant reproduces the source model exactly.
        """
        unknown = [s for s in species if s not in self._species_names]
        if unknown:
            raise KeyError(
                f"{unknown} are not species on {self._name!r}; "
                f"available: {sorted(self._species_names)}"
            )
        ruled = [s for s in species if s in self._assigned_names]
        if ruled:
            raise ValueError(
                f"{ruled} are set by an assignment rule on {self._name!r}, "
                "so the model computes them itself and cannot read them "
                "from an external pool."
            )
        set_by_event = sorted(
            {
                s
                for s in species
                for ev in self._events
                for tgt, _ in getattr(ev, "_assignments", ())
                if tgt == s
            }
        )
        if set_by_event:
            raise ValueError(
                f"{set_by_event} are assigned by an SBML event on "
                f"{self._name!r}; an event cannot write a pool this model "
                "no longer owns. Drop the events (without_events) or keep "
                "the species."
            )
        import copy

        new = copy.copy(self)
        object.__setattr__(
            new,
            "_species_inputs",
            tuple(dict.fromkeys(self._species_inputs + tuple(species))),
        )
        return new

    def native_input_exposure(self, input_name, t_start, t_end, *, n=8000):
        """``∫ native-drive dt`` for driveable quantity ``input_name`` over
        composite time ``[t_start, t_end]`` — the exposure the model's driven
        rates were calibrated to. A forcing source delivering a very different
        integrated exposure runs the model off that calibration;
        :func:`hallsim.models.forcing.drive_pulse` compares against this and
        warns. A constant's native drive is its published value held flat;
        a boundary input's is its assignment rule integrated. Returns 0.0 if
        the input has no time-dependent assignment rule."""
        if t_end <= t_start:
            return 0.0
        if input_name not in self._w_names:
            return float(self.parameters[input_name]) * (
                float(t_end) - float(t_start)
            )
        host = getattr(self._model, "modelstepfunc", self._model)
        af = getattr(host, "assignmentfunc", None)
        if af is None:
            return 0.0
        widx = dict(zip(self._w_names, self._w_indexes))[input_name]
        y0 = getattr(self._model, "y0", None)
        y = (
            jnp.asarray(y0)
            if y0 is not None
            else jnp.zeros(len(self._species_names))
        )
        ts = jnp.linspace(float(t_start), float(t_end), n)
        native = jax.vmap(
            lambda tt: af(y, self._w0, self._c, tt * self.time_scale)[widx]
        )(ts)
        return float(jnp.trapezoid(native, ts))

    def ports_schema(self):
        schema = {
            name: Port(
                role=PortRole.EVOLVED,
                default=float(y0),
                units="dimensionless",
                description=f"SBML species: {name}",
                ontology=dict(ont) if ont else {},
            )
            for name, y0, ont in zip(
                self._species_names,
                self._species_y0,
                self._species_ontology or ({},) * len(self._species_names),
            )
        }
        # A species handed over to another model's pool keeps its name and
        # identity and is read, not integrated (with_species_input).
        for name in self._species_inputs:
            owned = schema[name]
            schema[name] = Port(
                role=PortRole.INPUT,
                default=owned.default,
                units=owned.units,
                description=f"SBML species {name}, read from an external pool",
                ontology=owned.ontology,
            )
        # An assignment rule determines its target outright, so ASSIGNED
        # replaces the EVOLVED port when the species is in `y` as well.
        schema.update(
            {
                name: Port(
                    role=PortRole.ASSIGNED,
                    default=y0,
                    units="dimensionless",
                    description=f"SBML assignment rule: {name}",
                )
                for name, y0 in zip(self._assigned_names, self._assigned_y0)
            }
        )
        schema.update(self._driver_input_ports())
        schema.update(
            {
                port: Port(
                    role=PortRole.INPUT,
                    default=0.0,
                    units="dimensionless",
                    description=f"drives boundary input {name!r}",
                )
                for name, port in self._input_drivers
            }
        )
        return schema

    def assign(self, t, state):
        """Values of the ASSIGNED ports — the SBML assignment rules, evaluated
        at the current state on the model's own clock."""
        if not self._assigned_names:
            return {}
        assignmentfunc = self._model.assignmentfunc
        y = jnp.stack([state[name] for name in self._species_names], axis=-1)
        t_native = t * self.time_scale
        c = self._constants(t)
        if y.ndim > 1:
            w = jax.vmap(assignmentfunc, in_axes=(0, None, None, None))(
                y, self._w0, c, t_native
            )
        else:
            w = assignmentfunc(y, self._w0, c, t_native)
        return {
            name: w[..., idx]
            for name, idx in zip(self._assigned_names, self._assigned_indexes)
        }

    def _constants(self, t):
        """The SBML ``c`` vector with ``parameters`` scattered in — one
        vectorised write covering every constant.

        Left inside the RHS deliberately: memoising it on the instance would
        make the pytree structure change after first use and lose the JIT
        cache. It is loop-invariant unless a ``_param_step`` makes it depend
        on ``t``, and XLA hoists it out of the solver loop (measured: no
        runtime difference when hoisted by hand).
        """
        if not self._param_indexes:
            return self._c
        steps = {n: (ts, v0) for n, ts, v0 in self._param_steps}
        values = jnp.stack(
            [
                (
                    jnp.where(
                        t >= steps[n][0], self.parameters[n], steps[n][1]
                    )
                    if n in steps
                    else jnp.asarray(self.parameters[n], dtype=float)
                )
                for n in self._param_names
            ]
        )
        return self._c.at[jnp.asarray(self._param_indexes)].set(values)

    def derivative(self, t, state):
        # Trailing-axis stack, matching Composite.flatten/unflatten, so this
        # Process is shape-polymorphic and batched runs need no extra vmap.
        y = jnp.stack([state[name] for name in self._species_names], axis=-1)
        ratefunc = self._model.ratefunc
        assignmentfunc = self._model.assignmentfunc
        is_batched = y.ndim > 1

        c = self._constants(t)

        # Live drivers override a constant with an INPUT-port value. A batched
        # driving signal makes c per-batch, so the ratefunc vmaps over c too.
        c_batched = False
        if self._param_drivers:
            dv = self._driven_param_values(state)  # {param_name: value}
            names = list(dv)
            driven = jnp.stack([dv[n] for n in names], axis=-1)
            d_idx = jnp.asarray(
                [
                    self._param_indexes[self._param_names.index(n)]
                    for n in names
                ]
            )
            if driven.ndim > 1:  # batched signal → per-batch c
                batch = driven.shape[0]
                c = (
                    jnp.broadcast_to(c, (batch,) + c.shape)
                    .at[:, d_idx]
                    .set(driven)
                )
                c_batched = True
            else:
                c = c.at[d_idx].set(driven)

        # τ = t·time_scale, dy/dt = (dy/dτ)·time_scale — so time-referencing
        # assignment rules stay on the model's own clock.
        t_native = t * self.time_scale

        # Assignment rules evaluated from the *current* state; freezing `w` at
        # its initial value would leave a state-dependent rule stuck at t=0.
        w_batched = False
        if is_batched:
            w = jax.vmap(assignmentfunc, in_axes=(0, None, None, None))(
                y, self._w0, c, t_native
            )
            w_batched = True
        else:
            w = assignmentfunc(y, self._w0, c, t_native)

        # A driven input overrides the native SBML drive already in `w` with
        # its INPUT-port value, so a prescribed dose is a wired forcing source
        # rather than a special case.
        if self._input_drivers:
            name_to_widx = dict(zip(self._w_names, self._w_indexes))
            drv_idx = jnp.asarray(
                [name_to_widx[n] for n, _ in self._input_drivers]
            )
            drv_vals = jnp.stack(
                [
                    jnp.asarray(state[port], dtype=float)
                    for _, port in self._input_drivers
                ],
                axis=-1,
            )
            if drv_vals.ndim > 1 and not w_batched:  # batched drive, scalar w
                w = jnp.broadcast_to(w, drv_vals.shape[:-1] + w.shape)
                w_batched = True
            w = w.at[..., drv_idx].set(drv_vals)

        if is_batched:
            w_in = 0 if w_batched else None
            c_in = 0 if c_batched else None
            dydt = jax.vmap(ratefunc, in_axes=(0, None, w_in, c_in))(
                y, t_native, w, c
            )
        else:
            dydt = ratefunc(y, t_native, w, c)
        dydt = dydt * self.time_scale

        if self._frozen_indices:
            dydt = dydt.at[..., jnp.asarray(self._frozen_indices)].set(0.0)

        # A species read from an external pool is not this model's to move,
        # and one set by an assignment rule is ASSIGNED, not integrated.
        skip = set(self._species_inputs) | set(self._assigned_names)
        return {
            name: dydt[..., i]
            for i, name in enumerate(self._species_names)
            if name not in skip
        }

    def metadata(self):
        base = super().metadata()
        base["sbml_name"] = self._name
        base["n_species"] = len(self._species_names)
        base["species_inputs"] = list(self._species_inputs)
        return base


def _atomic_write(out_path: str, write) -> None:
    """``write(tmp_path)`` into the destination directory, then rename.

    A reader in another process sees either the old file or the new one, never
    a half-written document.
    """
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(out_path), prefix=".tmp-", suffix=".xml"
    )
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _download_biomodel_to_cache(model_id) -> str:
    """Fetch SBML XML for a BioModels ID and cache it under
    ``~/.cache/hallsim/biomodels``. Returns the cached path.

    Subsequent calls with the same ID reuse the cached file (BioModels
    IDs are immutable post-curation), so this is a one-time download per
    model per machine.
    """
    import os
    import urllib.request

    from hallsim.discovery import BIOMODELS_DOWNLOAD, _accession

    cache_dir = os.path.expanduser("~/.cache/hallsim/biomodels")
    os.makedirs(cache_dir, exist_ok=True)
    if isinstance(model_id, int):
        fname = f"BIOMD{model_id:010d}.xml"
    else:
        fname = f"{model_id}.xml"
    cache_path = os.path.join(cache_dir, fname)
    if not os.path.exists(cache_path):
        accession = _accession(model_id)
        url = (
            BIOMODELS_DOWNLOAD.format(model_id=accession)
            + f"?filename={accession}_url.xml"
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            xml = response.read().decode("utf-8")
        # Atomic, so an interrupted or concurrent download cannot leave a
        # truncated file that every later run then trusts.
        _atomic_write(cache_path, lambda p: open(p, "w").write(xml))
    return cache_path


def _extract_compartment_names(xml_path: str) -> frozenset[str]:
    """Compartment ids. They reach ``parameters`` as sizes, but a compartment
    volume is geometry: it scales every rate at once, so it is the most
    sensitive and the most degenerate thing in a fit."""
    import libsbml

    model = libsbml.SBMLReader().readSBMLFromFile(str(xml_path)).getModel()
    if model is None:
        return frozenset()
    return frozenset(
        model.getCompartment(i).getId()
        for i in range(model.getNumCompartments())
    )


def _extract_species_labels(xml_path: str) -> dict[str, str]:
    """``{species_id: display name}`` from the SBML.

    A CellDesigner export gives every species a UUID id and puts the gene
    symbol in the ``name`` attribute, so anything matching on id alone is
    blind to it (Dwivedi 2014 produces IL6 under ``mwf626e95e_543f_...``).
    """
    import libsbml

    model = libsbml.SBMLReader().readSBMLFromFile(str(xml_path)).getModel()
    if model is None:
        return {}
    return {
        model.getSpecies(i).getId(): (model.getSpecies(i).getName() or "")
        for i in range(model.getNumSpecies())
    }


def _extract_species_ontology(xml_path: str) -> dict[str, dict[str, str]]:
    """Pull MIRIAM identifier URIs from each species' annotation block.

    SBML curators annotate species with controlled-vocabulary URIs that
    point to entries in registries like UniProt, ChEBI, GO, SBO, and
    Reactome — the canonical form is
    ``http(s)://identifiers.org/<namespace>/<id>``. This function reads
    every species' CVTerm resources, parses the URIs, and returns a
    ``{species_id: {namespace: id}}`` mapping suitable for populating
    :attr:`hallsim.process.Port.ontology`. The first URI seen per
    namespace wins when a species has multiple resources in the same
    collection.

    Species without parseable annotations get an empty dict. Returns an
    empty mapping if libsbml cannot parse the file.
    """
    import re

    import libsbml

    pattern = re.compile(r"https?://identifiers\.org/([^/]+)/(.+)$")

    reader = libsbml.SBMLReader()
    doc = reader.readSBMLFromFile(str(xml_path))
    model = doc.getModel()
    if model is None:
        return {}

    result: dict[str, dict[str, str]] = {}
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        sp_id = sp.getId()
        ontology: dict[str, str] = {}
        for j in range(sp.getNumCVTerms()):
            cv = sp.getCVTerm(j)
            for k in range(cv.getNumResources()):
                uri = cv.getResourceURI(k)
                match = pattern.match(uri)
                if match:
                    namespace, identifier = match.group(1), match.group(2)
                    ontology.setdefault(namespace, identifier)
        result[sp_id] = ontology
    return result


def _extract_stoichiometry(xml_path: str) -> dict:
    """Species × reaction stoichiometry ``N`` straight from the SBML.

    Returns ``{"species": (id, ...), "reactions": (id, ...), "matrix":
    ((coeff, ...), ...)}`` with one matrix row per species. This is the
    model's wiring, not its kinetics: it fixes the conserved moieties exactly
    and independently of every rate constant, which is what distinguishes a
    moiety from a merely slow direction.

    Species the network cannot change are excluded — ``boundaryCondition``
    (held by the experiment) and ``constant`` — since a reaction touching one
    is not a constraint on the state. A species carrying a non-integer or
    symbolic stoichiometry (``stoichiometryMath``) makes the matrix
    parameter-dependent, so the extraction reports nothing rather than
    something conditionally true. Likewise a ``rateRule`` on one of these
    species — its dynamics are then not ``N·v`` at all, so ``N`` no longer
    settles what is conserved.

    Empty structure if libsbml cannot parse the file or the model has no
    reactions (a rules-only model, where ``N`` says nothing).
    """
    import libsbml

    empty = {"species": (), "reactions": (), "matrix": ()}
    model = libsbml.SBMLReader().readSBMLFromFile(str(xml_path)).getModel()
    if model is None or model.getNumReactions() == 0:
        return empty

    dynamic = [
        s.getId()
        for s in model.getListOfSpecies()
        if not s.getBoundaryCondition() and not s.getConstant()
    ]
    if not dynamic:
        return empty
    rate_ruled = {
        r.getVariable() for r in model.getListOfRules() if r.isRate()
    }
    if rate_ruled & set(dynamic):
        return empty
    row_of = {sid: i for i, sid in enumerate(dynamic)}

    reactions = [r.getId() for r in model.getListOfReactions()]
    matrix = [[0.0] * len(reactions) for _ in dynamic]
    for col, reaction in enumerate(model.getListOfReactions()):
        for refs, sign in (
            (reaction.getListOfReactants(), -1.0),
            (reaction.getListOfProducts(), +1.0),
        ):
            for ref in refs:
                if ref.isSetStoichiometryMath():
                    return empty
                row = row_of.get(ref.getSpecies())
                if row is None:
                    continue
                matrix[row][col] += sign * ref.getStoichiometry()

    return {
        "species": tuple(dynamic),
        "reactions": tuple(reactions),
        "matrix": tuple(tuple(r) for r in matrix),
    }


def _extract_reaction_channels(
    xml_path: str, core: SBMLCore
) -> tuple[SBMLReactionChannel, ...]:
    """Each reaction's id, rate law and signed net stoichiometry."""
    import libsbml

    model = libsbml.SBMLReader().readSBMLFromFile(str(xml_path)).getModel()
    if model is None:
        return ()
    channels = []
    for i, law in enumerate(core.rate_laws):
        rxn = model.getReaction(i)
        net: dict[str, float] = {}
        for refs, sign in (
            (rxn.getListOfReactants(), -1.0),
            (rxn.getListOfProducts(), 1.0),
        ):
            for ref in refs:
                sid = ref.getSpecies()
                net[sid] = net.get(sid, 0.0) + sign * float(
                    ref.getStoichiometry()
                )
        channels.append(
            ReactionChannel(
                reaction_id=rxn.getId(),
                rate_law=law,
                stoichiometry=tuple(net.items()),
            )
        )
    return tuple(channels)


def _extract_coupling_metadata(xml_path: str) -> dict:
    """Structure a coupling-wiring checker needs to judge what may drive what.

    Returns ``{param_constant, param_sbo, variables, rules,
    rate_rule_targets, boundary}``:
    - ``param_constant`` — ``{param_id: bool}`` (SBML ``constant`` flag).
    - ``param_sbo`` — ``{param_id: int}`` SBO term (−1 if unset); lets a driver
      target be classified as a kinetic rate constant.
    - ``variables`` — ids of *dynamic* quantities: species, rule targets, and
      non-constant parameters. These are the model's own state / input
      channels.
    - ``rules`` — ``[(target_id, frozenset(referenced_ids)), …]`` for every
      rule with a set target and math, so the checker can see that e.g.
      ``kd2_0`` is modulated by both the constant ``kd2`` and the variable
      ``DNAdamage`` — i.e. the model routes the influence through
      ``DNAdamage``, not ``kd2``.
    - ``rate_rule_targets`` — the subset of ``rules`` targets set by *rate*
      rules. A rate rule declares ``d(target)/dt``, so its target is an
      integrated state that an added derivative contribution sums into;
      an assignment-rule target is recomputed algebraically and would
      overwrite one. Only the latter is a wiring error.

    Empty structure if libsbml cannot parse the file.
    """
    import libsbml

    reader = libsbml.SBMLReader()
    model = reader.readSBMLFromFile(str(xml_path)).getModel()
    if model is None:
        return {
            "param_constant": {},
            "param_sbo": {},
            "variables": frozenset(),
            "rules": (),
            "rate_rule_targets": frozenset(),
        }

    def ast_names(node) -> frozenset:
        if node is None:
            return frozenset()
        names, stack = set(), [node]
        while stack:
            n = stack.pop()
            if n.getType() == libsbml.AST_NAME:
                names.add(n.getName())
            for i in range(n.getNumChildren()):
                stack.append(n.getChild(i))
        return frozenset(names)

    param_constant = {
        p.getId(): p.getConstant() for p in model.getListOfParameters()
    }
    param_sbo = {
        p.getId(): p.getSBOTerm() for p in model.getListOfParameters()
    }
    species_ids = {s.getId() for s in model.getListOfSpecies()}
    boundary = frozenset(
        s.getId()
        for s in model.getListOfSpecies()
        if s.getBoundaryCondition() or s.getConstant()
    )
    rules, rule_targets, rate_rule_targets = [], set(), set()
    for r in model.getListOfRules():
        if r.isSetVariable() and r.isSetMath():
            rules.append((r.getVariable(), ast_names(r.getMath())))
            rule_targets.add(r.getVariable())
            if r.isRate():
                rate_rule_targets.add(r.getVariable())
    nonconst_params = {k for k, c in param_constant.items() if not c}
    variables = frozenset(species_ids | rule_targets | nonconst_params)
    return {
        "param_constant": param_constant,
        "param_sbo": param_sbo,
        "variables": variables,
        "rules": tuple(rules),
        "rate_rule_targets": frozenset(rate_rule_targets),
        "boundary": boundary,
    }


def _extract_native_time_seconds(xml_path: str) -> tuple[float, bool]:
    """``(seconds_per_time_unit, declared)`` for the model's rate constants.

    SBML rate laws use a model-specific time unit, so composing models that
    disagree (days vs hours vs seconds) silently runs them at different
    real-world speeds on a shared ``t``. This is the conversion
    :attr:`SBMLProcess.time_scale` uses to put them on one clock.

    ``declared`` is False when the seconds value is the SBML fallback rather
    than something the modeller stated — a per-minute model that omits
    ``timeUnits`` is indistinguishable by value from a genuine seconds model,
    so reconciling it is silently 60x wrong. Callers warn on it.

    Resolution order: model-level ``timeUnits`` naming a ``<unitDefinition>``
    (L3); a ``<unitDefinition id="time">`` (the L2 convention); a base-unit
    ``timeUnits``; otherwise ``(1.0, False)``.
    """
    import libsbml

    doc = libsbml.SBMLReader().readSBMLFromFile(str(xml_path))
    model = doc.getModel()
    if model is None:
        return 1.0, False

    tu = model.getTimeUnits()  # "" when unset (L3); empty on L2 models
    unit_def = model.getUnitDefinition(tu or "time")
    if unit_def is None:
        # No <unitDefinition> resolved. A base-unit timeUnits ("second") is a
        # real declaration (SBML's only base time unit); unset or dimensionless
        # is not — treat as a guess so the caller can warn.
        if tu and tu != "dimensionless":
            return 1.0, True
        return 1.0, False

    seconds = 1.0
    for k in range(unit_def.getNumUnits()):
        u = unit_def.getUnit(k)
        seconds *= (
            u.getMultiplier() * 10.0 ** u.getScale()
        ) ** u.getExponent()
    return float(seconds), True


def _load_local_sbml(sbml_path: str):
    """``(core, y0, w0, c)`` for an SBML file, from :func:`compile_sbml`."""
    core = compile_sbml(sbml_path)
    as_vec = lambda values: jnp.asarray(values, dtype=float)  # noqa: E731
    return core, as_vec(core.y0), as_vec(core.w0), as_vec(core.c0)


def _collect_boundary_inputs(xml_path: str) -> set[str]:
    """Boundary species that are exogenous inputs, not observable outputs.

    A boundary species in SBML is imposed on the model rather than computed
    by its reactions — i.e. an input port. Those whose assignment rule
    references only ``time`` and constants (or that have no rule) are
    experimental forcing inputs (DallePezze 2014's ``Irradiation``,
    ``Insulin``, ``Amino_Acids``); HallSim surfaces them as settable
    ``parameters`` so hallmarks / Calibrator can drive them. Boundary
    species whose rule references other species are observable readouts (the
    ``_obs`` outputs) and are left alone.

    Returns the set of input-species ids. Empty if the file cannot be parsed.
    """
    import re

    import libsbml

    doc = libsbml.SBMLReader().readSBMLFromFile(str(xml_path))
    model = doc.getModel()
    if model is None:
        return set()

    species_ids = {
        model.getSpecies(i).getId() for i in range(model.getNumSpecies())
    }
    rule_formula: dict[str, str] = {}
    for i in range(model.getNumRules()):
        r = model.getRule(i)
        if r.isSetVariable() and r.isSetMath():
            rule_formula[r.getVariable()] = libsbml.formulaToString(
                r.getMath()
            )

    inputs: set[str] = set()
    for i in range(model.getNumSpecies()):
        s = model.getSpecies(i)
        if not s.getBoundaryCondition():
            continue
        sid = s.getId()
        formula = rule_formula.get(sid)
        if formula is None:
            inputs.add(sid)  # constant boundary species, no rule
            continue
        references_species = any(
            re.search(r"\b" + re.escape(other) + r"\b", formula)
            for other in species_ids
        )
        if not references_species:
            inputs.add(sid)
    return inputs


def _detect_inert_sinks(xml_path: str) -> set[str]:
    """Species that are written by reactions but read by nothing.

    A degradation "sink" (conventionally named ``Nil``/``Sink``/``∅``):
    reactions dump degraded material into it as a formal product, but no
    rate law or rule ever reads it. Integrating such a species is
    pointless and, because it only accumulates, it grows without bound
    (a "total-degraded" counter) — ruining the state's numerical scaling.
    It should be a boundary species. We detect it (read by no kinetic law
    or rule, yet a product of some reaction) so the caller can freeze it.

    Returns the set of inert-sink species ids. Empty if unparseable.
    """
    import re

    import libsbml

    doc = libsbml.SBMLReader().readSBMLFromFile(str(xml_path))
    model = doc.getModel()
    if model is None:
        return set()

    # Every identifier that appears in a kinetic law or rule expression —
    # i.e. every quantity the dynamics actually read.
    read: set[str] = set()
    for i in range(model.getNumReactions()):
        kl = model.getReaction(i).getKineticLaw()
        if kl is not None and kl.isSetMath():
            read.update(
                re.findall(
                    r"[A-Za-z_]\w*", libsbml.formulaToString(kl.getMath())
                )
            )
    for i in range(model.getNumRules()):
        r = model.getRule(i)
        if r.isSetMath():
            read.update(
                re.findall(
                    r"[A-Za-z_]\w*", libsbml.formulaToString(r.getMath())
                )
            )

    sinks: set[str] = set()
    for i in range(model.getNumSpecies()):
        s = model.getSpecies(i)
        sid = s.getId()
        if s.getBoundaryCondition() or sid in read:
            continue
        is_product = any(
            model.getReaction(j).getProduct(k).getSpecies() == sid
            for j in range(model.getNumReactions())
            for k in range(model.getReaction(j).getNumProducts())
        )
        if is_product:
            sinks.add(sid)
    return sinks


JWS_SBML_URL = "https://jjj.bio.vu.nl/models/{slug}/sbml/"


def _download_jws_to_cache(slug: str) -> str:
    """Fetch a JWS Online model's SBML and cache it under
    ``~/.cache/hallsim/jws``. Returns the cached path."""
    import urllib.request
    from pathlib import Path

    out = Path.home() / ".cache" / "hallsim" / "jws" / f"{slug}.xml"
    if out.exists():
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        JWS_SBML_URL.format(slug=slug), timeout=60
    ) as fh:
        body = fh.read()
    if b"<sbml" not in body[:4000]:
        raise ValueError(
            f"JWS model {slug!r} did not return SBML — check the slug at "
            f"https://jjj.bio.vu.nl/models/{slug}/"
        )
    out.write_bytes(body)
    return str(out)


def _resolve_source(model_id, name):
    """``(xml_path, name)`` for a local path, a BioModels ID, or ``jws:<slug>``.

    A bare integer or ``BIOMD...`` is BioModels; ``jws:glycolysis1`` is JWS
    Online. Both cache to disk, so a repeated import is a local read. A local
    path ending ``.cps`` is a COPASI model and is converted to SBML first.
    """
    import os

    if isinstance(model_id, str) and os.path.isfile(model_id):
        name = name or os.path.splitext(os.path.basename(model_id))[0]
        # A COPASI file is converted to SBML here, at the boundary, so every
        # downstream check sees an ordinary SBML import (hallsim.cps_import).
        if model_id.lower().endswith(".cps"):
            from hallsim.cps_import import cps_to_sbml

            log.info(f"Converting COPASI file '{model_id}' as '{name}'...")
            return cps_to_sbml(model_id), name
        log.info(f"Loading local SBML file '{model_id}' as '{name}'...")
        return model_id, name
    if isinstance(model_id, str) and model_id.lower().startswith("jws:"):
        slug = model_id.split(":", 1)[1]
        name = name or f"jws_{slug}"
        log.info(f"Fetching JWS Online '{slug}' as '{name}'...")
        return _download_jws_to_cache(slug), name
    name = name or f"biomodel_{model_id}"
    log.info(f"Fetching BioModels #{model_id} as '{name}'...")
    return _download_biomodel_to_cache(model_id), name


def _ordered_species(core) -> tuple[str, ...]:
    """Species names in state-vector order."""
    return tuple(core.y_indexes)


def _index_maps(core):
    """``(c_indexes, w_indexes)`` — the core's constant and assigned maps."""
    return core.c_indexes, core.w_indexes


def _settable_surface(xml_path, c, w0, c_indexes, w_indexes):
    """Every SBML constant at its published default, plus boundary-input
    species (Irradiation, Insulin, …) at theirs — the whole surface addressable
    by calibration targets and hallmark substitution, uncurated.

    Boundary inputs live in the ``w`` vector but are surfaced through the same
    dict and routed at derivative time. Returns ``(params_dict, param_names,
    param_indexes, w_names, w_index_tuple, boundary_inputs)``.
    """
    if c_indexes is None:
        params_dict, param_names, param_indexes = {}, (), ()
    else:
        params_dict = {n: float(c[i]) for n, i in c_indexes.items()}
        param_names = tuple(c_indexes.keys())
        param_indexes = tuple(c_indexes[n] for n in param_names)

    boundary_inputs = _collect_boundary_inputs(xml_path) & set(w_indexes)
    params_dict.update({n: float(w0[w_indexes[n]]) for n in boundary_inputs})
    w_names = tuple(sorted(boundary_inputs))
    # Everything else in `w` is determined by an <assignmentRule> each step.
    # Those are outputs, not inputs, and go out as ASSIGNED ports.
    assigned = tuple(sorted(set(w_indexes) - boundary_inputs))
    return (
        params_dict,
        param_names,
        param_indexes,
        w_names,
        tuple(w_indexes[n] for n in w_names),
        boundary_inputs,
        assigned,
        tuple(w_indexes[n] for n in assigned),
    )


def _frozen_sink_indices(xml_path, species_names, name) -> tuple[int, ...]:
    """Indices of inert sinks — written by degradation, read by nothing.
    Frozen so they don't accumulate and ruin the state scaling."""
    inert = _detect_inert_sinks(xml_path)
    frozen = tuple(i for i, n in enumerate(species_names) if n in inert)
    if frozen:
        log.warning(
            "%s: inert sink species %s are written but read by nothing; "
            "freezing them (treated as boundary) so they cannot accumulate "
            "unboundedly. They now hold their initial value and are UNUSABLE "
            "as coupling sources or reporter observables. A terminal product "
            "this model exports is indistinguishable from a degradation "
            "counter by this test — lift the freeze with "
            "proc.with_unfrozen(...), or mark boundaryCondition=true in the "
            "source SBML.",
            name,
            [species_names[i] for i in frozen],
        )
    return frozen


def _apply_parameter_overrides(
    params_dict, parameters, c_indexes, boundary_inputs
):
    """Overwrite defaults with caller-supplied values, validated against the
    combined settable surface (constants + boundary inputs)."""
    if not parameters:
        return
    settable = set(c_indexes or ()) | boundary_inputs
    missing = [p for p in parameters if p not in settable]
    if missing:
        raise KeyError(
            f"parameters {missing} not found in SBML constants or "
            f"boundary inputs. Available constants: "
            f"{sorted(c_indexes or ())}; boundary inputs: "
            f"{sorted(boundary_inputs)}"
        )
    for n, v in parameters.items():
        params_dict[n] = float(v)


def _native_clock(xml_path, name, supplied=None):
    """``(seconds_per_native_unit, source)`` where source is ``"declared"``,
    ``"supplied"`` or ``"assumed"``. Warns loudly when assumed — an assumed
    clock is silently 60x/3600x/86400x wrong once reconciled."""
    seconds, declared = _extract_native_time_seconds(xml_path)
    if supplied is not None:
        supplied = float(supplied)
        if supplied <= 0:
            raise ValueError(
                f"native_time_seconds must be positive, got {supplied}"
            )
        if declared and supplied != seconds:
            log.warning(
                "%s: SBML declares native_time_seconds=%g but the caller "
                "supplied %g; using the supplied value and overriding the "
                "source's own assertion.",
                name,
                seconds,
                supplied,
            )
        return supplied, "supplied"
    if not declared:
        log.warning(
            "%s: SBML declares no time unit; assuming native_time_seconds=1.0 "
            "(seconds). If this model's rate laws are in minutes/hours/days its "
            "clock is now a GUESS — reconciling it onto a shared canonical axis "
            "will be silently 60x/3600x/86400x wrong. Set the source SBML's "
            "timeUnits, or pass process_from_sbml(native_time_seconds=...) if "
            "you know the true value. Check `proc.native_time_source` before "
            "composing.",
            name,
        )
    return seconds, "declared" if declared else "assumed"


def process_from_sbml(
    model_id: int | str,
    name: str | None = None,
    timescale: float | None = None,
    parameters: dict[str, float] | None = None,
    native_time_seconds: float | None = None,
) -> SBMLProcess:
    """Load an SBML model and wrap it as a Process.

    Parameters
    ----------
    model_id:
        A BioModels numeric ID (``10`` = Kholodenko2000 MAPK) or a path to a
        local SBML XML file.
    name:
        Process name; defaults to ``"biomodel_{model_id}"`` or the filename.
    timescale:
        Characteristic timescale for multi-rate scheduling. ``None`` uses the
        model's native time unit.
    parameters:
        ``{c_name: value}`` overriding SBML defaults at construction. Every
        constant is auto-populated at its published default first, so this
        only replaces the listed keys.
    native_time_seconds:
        Real seconds per unit of the model's own time axis, for the common case
        of a file that declares no ``timeUnits`` and whose rate laws are not in
        seconds — Kallenberger 2014 is in minutes, so ``60.0``. Without it the
        importer assumes seconds and ``reconciled_to`` is silently 60×/3600×/
        86400× wrong. Recorded as ``native_time_source == "supplied"``, kept
        distinct from a clock the source actually asserts.

    Returns an :class:`SBMLProcess` with ports auto-generated from the species.
    Raises ``UnsupportedSBMLFeatureError`` for a construct the importer does
    not translate, ``KeyError`` on an unknown parameter name.
    """
    xml_path, name = _resolve_source(model_id, name)

    # Single import path: local files and downloads alike go through
    # _load_local_sbml, which caches one compiled core per file.
    model, y0, w0, c = _load_local_sbml(xml_path)
    species_names = _ordered_species(model)
    log.info(f"Loaded {len(species_names)} species: {species_names}")

    # MIRIAM annotations on each species → Port.ontology, so the
    # composability analyzer can detect shared biology across imported
    # SBML models by their identifiers.org references.
    ontology_map = _extract_species_ontology(xml_path)
    coupling_meta = _extract_coupling_metadata(xml_path)
    compartment_names = _extract_compartment_names(xml_path)
    stoichiometry = _extract_stoichiometry(xml_path)
    reaction_channels = _extract_reaction_channels(xml_path, model)
    species_ontology = tuple(ontology_map.get(s, {}) for s in species_names)
    _species_label_map = _extract_species_labels(xml_path)

    native_time_seconds, native_time_source = _native_clock(
        xml_path, name, native_time_seconds
    )

    c_indexes, w_indexes_map = _index_maps(model)
    (
        params_dict,
        param_names,
        param_indexes,
        w_names,
        w_index_tuple,
        boundary_inputs,
        assigned_names,
        assigned_indexes,
    ) = _settable_surface(xml_path, c, w0, c_indexes, w_indexes_map)
    frozen_indices = _frozen_sink_indices(xml_path, species_names, name)
    _apply_parameter_overrides(
        params_dict, parameters, c_indexes, boundary_inputs
    )

    # Translate SBML <event> elements (the compiled core ignores them)
    # into EVENT processes. Expand into a composite via
    # hallsim.sbml_events.expand_events(proc).
    from hallsim.sbml_events import translate_events

    events = translate_events(xml_path, species_names, params_dict, name)
    # Through __init__, never object.__new__ + setattr: JAX rebuilds this pytree
    # at every jit/partition boundary, and a field-by-field instance does not
    # match what tree_unflatten produces — its structure shifts on the
    # round-trip and eqx.partition rejects it.
    proc = SBMLProcess(
        _species_names=species_names,
        _species_y0=tuple(float(y0[i]) for i in range(len(species_names))),
        _species_labels=tuple(
            (sid, _species_label_map.get(sid, "")) for sid in species_names
        ),
        _species_ontology=species_ontology,
        _coupling_meta=coupling_meta,
        _stoichiometry=stoichiometry,
        _reaction_channels=reaction_channels,
        native_time_seconds=native_time_seconds,
        native_time_source=native_time_source,
        time_scale=1.0,
        _model=model,
        _w0=w0,
        _c=c,
        _name=name,
        parameters=params_dict,
        _param_names=param_names,
        _param_indexes=param_indexes,
        _w_names=w_names,
        _w_indexes=w_index_tuple,
        _assigned_names=assigned_names,
        _assigned_indexes=assigned_indexes,
        _assigned_y0=tuple(float(w0[i]) for i in assigned_indexes),
        _frozen_indices=frozen_indices,
        _compartment_names=compartment_names,
        # Default the scheduler timescale to the model's native time unit (a
        # day-scale model has day-scale dynamics) so auto_groups clusters
        # mixed-rate composites correctly. Never None for SBML processes, so
        # reconciled_to / tree_at can replace it without None-leaf ambiguity.
        timescale=(
            float(timescale) if timescale is not None else native_time_seconds
        ),
        _events=tuple(events),
    )
    if events:
        log.info(
            "%s: imported %d SBML event(s); compose with "
            "sbml_events.expand_events(proc).",
            name,
            len(events),
        )

    return proc
