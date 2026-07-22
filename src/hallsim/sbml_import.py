"""SBML auto-import — convert BioModels SBML files into Process instances.

Uses ``sbmltoodejax`` to convert SBML species and reactions into a
JAX-compatible RHS function, then wraps it as a :class:`Process` with
auto-generated ports and metadata.

Example
-------
>>> proc = process_from_sbml(10, name="mapk_cascade")
>>> proc.ports_schema()    # auto-generated from SBML species
>>> proc.metadata()        # SBML annotations

Requires ``sbmltoodejax`` to be installed::

    pip install sbmltoodejax
"""

from __future__ import annotations

import logging
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from hallsim.imported import ImportedODEProcess
from hallsim.kinetics import hill_gate
from hallsim.process import Port, PortRole

log = logging.getLogger(__name__)


class UnsupportedSBMLFeatureError(Exception):
    """Raised when an SBML file uses features sbmltoodejax cannot translate.

    The pre-flight check in :func:`_precheck_sbml_supported` catches the
    documented limitations (events; named functions outside
    ``sbmltoodejax.modulegeneration.mathFuncs``) before ``GenerateModel``
    runs, so users get one clear message naming the offending feature
    rather than a cryptic traceback from inside the generated module.
    """


def _supported_function_names() -> set[str]:
    """Names sbmltoodejax recognises as named function calls.

    Source of truth is the ``mathFuncs`` dict literal inside
    ``sbmltoodejax.modulegeneration.GenerateModel``. Upstream defines
    it as a local variable, so we extract the keys via ``ast`` rather
    than copying them — the set stays in sync as the table grows. If
    upstream ever promotes ``mathFuncs`` to module scope, the direct
    attribute lookup below picks it up automatically. Arithmetic
    primitives (``+``, ``*``, ``**``, …) are not in this set because
    libsbml's ``formulaToString`` emits them as Python operators that
    never hit the function-name lookup.
    """
    import ast
    import inspect

    import sbmltoodejax.modulegeneration as mg

    if hasattr(mg, "mathFuncs") and isinstance(mg.mathFuncs, dict):
        keys: set[str] = set(mg.mathFuncs.keys())
    else:
        try:
            src = inspect.getsource(mg.GenerateModel)
        except (TypeError, OSError):
            return set()
        keys = set()
        for node in ast.walk(ast.parse(src)):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "mathFuncs"
                and isinstance(node.value, ast.Dict)
            ):
                keys = {
                    k.value
                    for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                }
                break
    # ParseRHS also special-cases bare identifiers 'time' and 'pi'.
    return keys | {"time", "pi"}


def _collect_known_symbols(sbml_model) -> set[str]:
    """Identifiers libsbml's ``formulaToString`` may emit that refer to
    model components (species, parameters, compartments, reactions,
    local kineticLaw parameters) rather than function calls."""
    names: set[str] = set()
    for i in range(sbml_model.getNumSpecies()):
        names.add(sbml_model.getSpecies(i).getId())
    for i in range(sbml_model.getNumParameters()):
        names.add(sbml_model.getParameter(i).getId())
    for i in range(sbml_model.getNumCompartments()):
        names.add(sbml_model.getCompartment(i).getId())
    for i in range(sbml_model.getNumReactions()):
        rxn = sbml_model.getReaction(i)
        names.add(rxn.getId())
        kl = rxn.getKineticLaw()
        if kl is None:
            continue
        for j in range(kl.getNumParameters()):
            names.add(kl.getParameter(j).getId())
        if hasattr(kl, "getNumLocalParameters"):
            for j in range(kl.getNumLocalParameters()):
                names.add(kl.getLocalParameter(j).getId())
    names.discard("")
    return names


def _collect_math_nodes(sbml_model):
    """Yield every libsbml ASTNode root attached to the model.

    Covers kinetic laws, rules, initial assignments, constraints,
    event triggers/delays/assignments, and user function definitions —
    everywhere SBML carries an evaluatable expression.
    """
    for i in range(sbml_model.getNumReactions()):
        kl = sbml_model.getReaction(i).getKineticLaw()
        if kl is not None and kl.isSetMath():
            yield kl.getMath()
    for i in range(sbml_model.getNumRules()):
        r = sbml_model.getRule(i)
        if r.isSetMath():
            yield r.getMath()
    for i in range(sbml_model.getNumInitialAssignments()):
        ia = sbml_model.getInitialAssignment(i)
        if ia.isSetMath():
            yield ia.getMath()
    for i in range(sbml_model.getNumConstraints()):
        c = sbml_model.getConstraint(i)
        if c.isSetMath():
            yield c.getMath()
    for i in range(sbml_model.getNumEvents()):
        e = sbml_model.getEvent(i)
        if e.isSetTrigger() and e.getTrigger().isSetMath():
            yield e.getTrigger().getMath()
        if e.isSetDelay() and e.getDelay().isSetMath():
            yield e.getDelay().getMath()
        for j in range(e.getNumEventAssignments()):
            ea = e.getEventAssignment(j)
            if ea.isSetMath():
                yield ea.getMath()
    for i in range(sbml_model.getNumFunctionDefinitions()):
        fd = sbml_model.getFunctionDefinition(i)
        if fd.isSetMath():
            yield fd.getMath()


def _precheck_sbml_supported(xml_path: str) -> list[str]:
    """Scan an SBML file for features sbmltoodejax cannot translate.

    Covers the two limitations documented at
    https://developmentalsystems.org/sbmltoodejax/why_use.html#limitations:

    1. ``<event>`` elements (discrete state changes).
    2. Named function calls in any math expression whose name is not in
       ``sbmltoodejax.modulegeneration.mathFuncs`` (and not the built-in
       ``time`` or ``pi`` identifiers that ``ParseRHS`` special-cases).
       Distinguishing operators from named function calls is delegated
       to libsbml's ``ASTNode.isFunction`` so arithmetic primitives are
       not flagged.

    Returns
    -------
    list of human-readable issue strings. Empty list means OK.
    """
    import libsbml

    reader = libsbml.SBMLReader()
    doc = reader.readSBMLFromFile(str(xml_path))
    model = doc.getModel()
    if model is None:
        return [f"libsbml could not parse {xml_path!r} as SBML"]

    issues: list[str] = []

    # <event> elements are translated separately (hallsim.sbml_events) and
    # stripped from the copy sbmltoodejax generates from, so they are not a
    # blocker here.

    # Mirror sbmltoodejax's identifier-resolution path: serialize each
    # math AST to infix via libsbml (the same conversion sbmltoodejax
    # itself feeds into ParseRHS), find every function-call identifier
    # (name immediately followed by ``(``), and flag any that isn't a
    # model symbol and isn't in the supported function set. Doing the
    # check post-formulaToString avoids false positives like ``power``
    # and ``root`` that libsbml rewrites to ``pow`` and ``sqrt``.
    import re

    supported = _supported_function_names()
    known_symbols = _collect_known_symbols(model)
    unsupported: dict[str, int] = {}
    for math_root in _collect_math_nodes(model):
        infix = libsbml.formulaToString(math_root)
        for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", infix):
            name = match.group(1)
            if name in supported or name in known_symbols:
                continue
            # libsbml renders <lambda> inside functionDefinitions as
            # "lambda(...)" in the infix string. That's an SBML construct,
            # not a call site; the actual unsupported event is the
            # call to the user-defined function elsewhere, which the
            # scan catches on its own.
            if name == "lambda":
                continue
            unsupported[name] = unsupported.get(name, 0) + 1

    if unsupported:
        listing = ", ".join(
            f"{name}() (x{n})" for name, n in sorted(unsupported.items())
        )
        issues.append(
            f"calls function(s) outside sbmltoodejax's mathFuncs table: "
            f"{listing}"
        )

    return issues


class HillParamDriver(eqx.Module):
    """Drives one SBML constant from another model's state, live.

    An imported model's parameter is normally a fixed constant. This turns
    a named parameter into a coupling target: at every derivative
    evaluation the parameter is Hill-interpolated between its own basal
    value (the ``parameters[param_name]`` entry — still a fittable point on
    the parameter surface) and ``hi``, gated on a signal read from an INPUT
    port::

        c[param_name] = basal + (hi - basal) · hill_gate(signal; K, n)

    so a slow upstream state (e.g. DP14 ``DNA_damage``) can drive an
    imported oscillator's damage input (e.g. GZ06 ``psi``) mechanistically,
    rather than both being set independently by an external knob. The
    basal stays on ``parameters`` so it calibrates through the ordinary
    ``parameters.<name>`` path; ``hi``/``K``/``n`` are structural.

    Parameter-value analogue of
    :class:`hallsim.models.hill_edge.HillActivationEdge`, which instead
    adds a Hill-gated *flux* to a state derivative.
    """

    param_name: str = eqx.field(static=True)
    input_port: str = eqx.field(static=True)
    c_index: int = eqx.field(static=True)
    hi: float = eqx.field(static=True)
    K: float = eqx.field(static=True)
    n: float = eqx.field(static=True)

    def value(self, basal, signal):
        return basal + (self.hi - basal) * hill_gate(
            jnp.asarray(signal), jnp.asarray(self.K), jnp.asarray(self.n)
        )


class SBMLProcess(ImportedODEProcess):
    """Process auto-generated from an SBML model via sbmltoodejax.

    This wraps the JAX-compiled model step function from sbmltoodejax,
    exposing SBML species as EVOLVED ports.

    Not constructed directly — use :func:`process_from_sbml`.

    The ``parameters`` field (inherited) is the named, substitutable
    surface for every SBML ``<parameter>`` and constant-rate species in the
    model. At construction time it is auto-populated with every SBML
    constant at its published default value, so the full mechanism surface
    is immediately discoverable via :meth:`calibratable_params` and
    :meth:`hallsim.composite.Composite.calibration_targets`. User-supplied
    ``parameters`` at construction override specific defaults; hallmarks
    substitute values via ``eqx.tree_at`` on a dotted ``parameters.<key>``
    field path; Calibrator does the same.
    """

    _param_label = "SBML constant"

    _species_names: tuple[str, ...] = ()
    _species_y0: tuple[float, ...] = ()
    _species_ontology: tuple[dict[str, str], ...] = ()
    # Structure for the coupling-wiring checker: param constancy + SBO, the
    # set of dynamic variables, and the assignment-rule dependency graph — so a
    # driver targeting a rate constant that the model modulates via a rule can
    # be flagged (see hallsim.coupling_wiring, _extract_coupling_metadata).
    # Static metadata; round-trips untouched through eqx.tree_at substitutions.
    _coupling_meta: dict = eqx.field(static=True, default=None)
    # native_time_seconds / time_scale / parameters / _param_names / _name
    # are inherited from ImportedODEProcess. Extraction fills them at import
    # (native_time_seconds = day 86400 / hour 3600 / second 1). Parallel
    # tuple below is fixed at construction so derivative-time lookup by name
    # stays JIT-safe even when eqx.tree_at reorders the parameters dict.
    _model: Any = None  # sbmltoodejax model object
    _w0: Any = None
    _c: Any = None
    _param_indexes: tuple[int, ...] = ()
    # Boundary-input species (e.g. Irradiation, Insulin) are the model's
    # experimental input ports. They live in the SBML `w` vector, not `c`,
    # but are exposed through the same `parameters` surface; these parallel
    # tuples route their current values into `_w0` at derivative time.
    _w_names: tuple[str, ...] = ()
    _w_indexes: tuple[int, ...] = ()
    # Inert "sink" species (written by degradation reactions, read by
    # nothing — e.g. a `Nil` degradation collector). Integrating them is
    # pointless and lets them accumulate unboundedly, wrecking the state's
    # numerical scaling. Their derivative is frozen to 0 (treated as a
    # boundary species), which is exact since no rate law reads them.
    _frozen_indices: tuple[int, ...] = ()
    # Live parameter couplings: each drives one SBML constant from an INPUT
    # port every derivative step (see :class:`HillParamDriver`). Empty for
    # a plain imported model; populated via :meth:`with_param_driver`. Static
    # (pure metadata) so it round-trips untouched through the ``eqx.tree_at``
    # substitutions the hallmark / Calibrator paths apply to `parameters`.
    _param_drivers: tuple = eqx.field(static=True, default=())
    # Translated SBML <event> elements as EVENT processes (see
    # hallsim.sbml_events). Static metadata; expand into a composite with
    # ``hallsim.sbml_events.expand_events``. Empty for event-free models.
    _events: tuple = eqx.field(static=True, default=())
    # Boundary inputs delivered as a bounded dose pulse:
    # ``((input_name, t_start, t_end), ...)``. The input holds its
    # ``parameters`` value while composite time ``t_start <= t < t_end``,
    # else 0 — a dose-then-washout protocol (e.g. etoposide at [-2, 0],
    # washout at day 0) instead of a sustained exposure.
    _pulse_windows: tuple = eqx.field(static=True, default=())
    # SBML constants delivered as a one-time step at a threshold time:
    # ``((param_name, t_step, value_before), ...)``. The constant holds
    # ``value_before`` while composite time ``t < t_step`` and its configured
    # ``parameters`` value once ``t >= t_step`` — a timed intervention (e.g.
    # rapamycin added at washout day 2 drops the mTOR rate) instead of a
    # severity applied for the whole trajectory.
    _param_steps: tuple = eqx.field(static=True, default=())

    def coupling_structure(self) -> dict:
        """SBML equation structure for the coupling-wiring check (extracted at
        import; see :func:`_extract_coupling_metadata`)."""
        return self._coupling_meta

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

    def with_pulse_window(
        self, input_name: str, t_start: float, t_end: float
    ) -> "SBMLProcess":
        """Return a copy that delivers boundary input ``input_name`` as a
        pulse: its ``parameters`` value while ``t_start <= t < t_end``, else
        0. ``input_name`` must be a boundary input (in ``_w_names``). The
        window is in composite time, so a dose at ``[-2, 0]`` needs the run
        to start at (or before) ``-2``."""
        if input_name not in self._w_names:
            raise KeyError(
                f"{input_name!r} is not a boundary input on {self._name!r}; "
                f"available: {sorted(self._w_names)}"
            )
        import copy

        new = copy.copy(self)
        object.__setattr__(
            new,
            "_pulse_windows",
            self._pulse_windows
            + ((input_name, float(t_start), float(t_end)),),
        )
        return new

    def with_param_driver(
        self,
        param_name: str,
        input_port: str,
        *,
        hi: float,
        K: float,
        n: float = 2.0,
    ) -> "SBMLProcess":
        """Return a copy whose ``param_name`` is driven by ``input_port``.

        Adds an INPUT port ``input_port`` (wire it to the driving store
        path via topology) and Hill-interpolates ``param_name`` between its
        basal ``parameters[param_name]`` value and ``hi`` on the port's
        signal. See :class:`HillParamDriver`.
        """
        if param_name not in self._param_names:
            raise KeyError(
                f"{param_name!r} is not an SBML constant on {self._name!r}; "
                f"available: {sorted(self._param_names)}"
            )
        c_index = self._param_indexes[self._param_names.index(param_name)]
        driver = HillParamDriver(
            param_name=param_name,
            input_port=input_port,
            c_index=int(c_index),
            hi=float(hi),
            K=float(K),
            n=float(n),
        )
        # HillParamDriver is pure static metadata (no array leaves), so
        # tree_at can't grow the tuple; copy + set the field directly,
        # mirroring the object.__setattr__ construction in process_from_sbml.
        import copy

        new = copy.copy(self)
        object.__setattr__(
            new, "_param_drivers", self._param_drivers + (driver,)
        )
        return new

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
        # INPUT ports feeding live parameter drivers, wired to the driving
        # store path via topology.
        for d in self._param_drivers:
            schema[d.input_port] = Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description=f"drives SBML constant {d.param_name!r}",
            )
        return schema

    def derivative(self, t, state):
        # Stack species along the trailing axis. For scalar state values
        # this yields shape (n_species,); for batched values (batch,) it
        # yields (batch, n_species). The trailing-axis convention matches
        # Composite.flatten/unflatten, so this Process is shape-polymorphic
        # and works under jax.vmap'd / batched Scheduler runs without
        # extra vmap'ing at the composite level.
        y = jnp.stack([state[name] for name in self._species_names], axis=-1)
        host = getattr(self._model, "modelstepfunc", self._model)
        ratefunc = host.ratefunc
        # AssignmentRule recomputes the `w` vector (observables, computed
        # totals, time-dependent inputs) from the current state each step.
        assignmentfunc = getattr(host, "assignmentfunc", None)
        is_batched = y.ndim > 1

        # Build the constants vector. parameters.<key> entries
        # substitute into c via a single vectorised scatter — one
        # XLA-fused write covering every constant, whether or not it
        # was modified from its SBML default.
        if self._param_indexes:
            indexes = jnp.asarray(self._param_indexes)
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
            c = self._c.at[indexes].set(values)
        else:
            c = self._c

        # Live parameter drivers: override each driven constant with a value
        # read from an INPUT port this step (see HillParamDriver). A batched
        # driving signal makes c per-batch, so the ratefunc vmaps over c too.
        c_batched = False
        if self._param_drivers:
            driven = jnp.stack(
                [
                    d.value(self.parameters[d.param_name], state[d.input_port])
                    for d in self._param_drivers
                ],
                axis=-1,
            )
            d_idx = jnp.asarray([d.c_index for d in self._param_drivers])
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

        # Map canonical time to this model's native time, evaluate the
        # native rate law there, and scale dy back to canonical time:
        #   τ = t · time_scale,  dy/dt = (dy/dτ)·time_scale.
        # Feeding t_native (not t) keeps time-referencing assignment rules
        # on the model's own clock. time_scale=1.0 leaves both untouched.
        t_native = t * self.time_scale

        # Evaluate the SBML assignment rules from the *current* state rather
        # than freezing `w` at its initial value. A state-dependent rule
        # that feeds a rate law (a conserved moiety, a computed total) is
        # then correct instead of silently stuck at t=0; SBML observables
        # (`*_obs`) are likewise computed live.
        w_batched = False
        if assignmentfunc is not None:
            if is_batched:
                w = jax.vmap(assignmentfunc, in_axes=(0, None, None, None))(
                    y, self._w0, c, t_native
                )
                w_batched = True
            else:
                w = assignmentfunc(y, self._w0, c, t_native)
        else:
            w = self._w0

        # Boundary inputs (Irradiation, Insulin, …) are experimenter knobs:
        # the current `parameters` value overrides whatever the model's own
        # (often time-pulse) assignment rule computed, so a sustained
        # severity-driven exposure wins over the source model's transient.
        if self._w_indexes:
            w_indexes = jnp.asarray(self._w_indexes)
            windows = {n: (ts, te) for n, ts, te in self._pulse_windows}
            w_values = jnp.stack(
                [
                    (
                        self.parameters[n]
                        * jnp.where(
                            (t >= windows[n][0]) & (t < windows[n][1]),
                            1.0,
                            0.0,
                        )
                        if n in windows
                        else jnp.asarray(self.parameters[n], dtype=float)
                    )
                    for n in self._w_names
                ]
            )
            w = w.at[..., w_indexes].set(w_values)

        if is_batched:
            w_in = 0 if w_batched else None
            c_in = 0 if c_batched else None
            dydt = jax.vmap(ratefunc, in_axes=(0, None, w_in, c_in))(
                y, t_native, w, c
            )
        else:
            dydt = ratefunc(y, t_native, w, c)
        dydt = dydt * self.time_scale

        # Freeze inert sink species (written by degradation, read by
        # nothing) so they don't accumulate and wreck the state scaling.
        # Exact: no rate law reads them, so zeroing their rate changes no
        # other trajectory.
        if self._frozen_indices:
            dydt = dydt.at[..., jnp.asarray(self._frozen_indices)].set(0.0)

        return {
            name: dydt[..., i] for i, name in enumerate(self._species_names)
        }

    def metadata(self):
        base = super().metadata()
        base["sbml_name"] = self._name
        base["n_species"] = len(self._species_names)
        return base


def _preprocess_sbml(sbml_path: str) -> str:
    """Apply libsbml converters that flatten SBML features sbmltoodejax
    cannot translate but that have well-defined equivalent forms.

    Currently runs:

    * **expandFunctionDefinitions** — inlines every ``<functionDefinition>``
      body at every call site. After this pass, the model has zero
      user-defined functions and no ``function_X(...)`` references, so
      sbmltoodejax (which rejects custom functions in ``ParseRHS``) can
      translate the model directly. Idempotent on models that have no
      function definitions to begin with.

    Returns a path to the converted SBML file under
    ``~/.cache/hallsim/converted``. The cache key is the basename of the
    input, so a converted local file lives alongside any converted
    BioModels download.
    """
    import os

    import libsbml

    reader = libsbml.SBMLReader()
    doc = reader.readSBMLFromFile(str(sbml_path))
    if doc.getModel() is None:
        # libsbml couldn't parse it; let the downstream pre-check produce
        # the actual diagnostic — we just hand back the original path.
        return sbml_path

    props = libsbml.ConversionProperties()
    props.addOption("expandFunctionDefinitions", True)
    doc.convert(props)

    cache_dir = os.path.expanduser("~/.cache/hallsim/converted")
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, os.path.basename(sbml_path))
    libsbml.writeSBMLToFile(doc, out_path)
    return out_path


def _download_biomodel_to_cache(model_id) -> str:
    """Fetch SBML XML for a BioModels ID and cache it under
    ``~/.cache/hallsim/biomodels``. Returns the cached path.

    Subsequent calls with the same ID reuse the cached file (BioModels
    IDs are immutable post-curation), so this is a one-time download per
    model per machine.
    """
    import os

    from sbmltoodejax.biomodels_api import get_content_for_model

    cache_dir = os.path.expanduser("~/.cache/hallsim/biomodels")
    os.makedirs(cache_dir, exist_ok=True)
    if isinstance(model_id, int):
        fname = f"BIOMD{model_id:010d}.xml"
    else:
        fname = f"{model_id}.xml"
    cache_path = os.path.join(cache_dir, fname)
    if not os.path.exists(cache_path):
        xml = get_content_for_model(model_id)
        with open(cache_path, "w") as f:
            f.write(xml)
    return cache_path


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


def _extract_coupling_metadata(xml_path: str) -> dict:
    """Structure a coupling-wiring checker needs to judge what may drive what.

    Returns ``{param_constant, param_sbo, variables, rules}``:
    - ``param_constant`` — ``{param_id: bool}`` (SBML ``constant`` flag).
    - ``param_sbo`` — ``{param_id: int}`` SBO term (−1 if unset); lets a driver
      target be classified as a kinetic rate constant.
    - ``variables`` — ids of *dynamic* quantities: species, assignment-rule
      targets, and non-constant parameters. These are the model's own
      state / input channels.
    - ``rules`` — ``[(target_id, frozenset(referenced_ids)), …]`` for every
      assignment rule, so the checker can see that e.g. ``kd2_0`` is modulated
      by both the constant ``kd2`` and the variable ``DNAdamage`` — i.e. the
      model routes the influence through ``DNAdamage``, not ``kd2``.

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
    rules, rule_targets = [], set()
    for r in model.getListOfRules():
        if r.isSetVariable() and r.isSetMath():
            rules.append((r.getVariable(), ast_names(r.getMath())))
            rule_targets.add(r.getVariable())
    nonconst_params = {k for k, c in param_constant.items() if not c}
    variables = frozenset(species_ids | rule_targets | nonconst_params)
    return {
        "param_constant": param_constant,
        "param_sbo": param_sbo,
        "variables": variables,
        "rules": tuple(rules),
        "boundary": boundary,
    }


def _extract_native_time_seconds(xml_path: str) -> float:
    """Seconds-per-time-unit the model's rate constants are expressed in.

    SBML rate laws are written in a model-specific time unit. Composing
    models that disagree (DallePezze 2014 in days, Geva-Zatorsky 2006 in
    hours, an unannotated model in seconds) silently runs them at
    different real-world speeds on a shared ``t`` axis. This value is the
    conversion the composite uses to put every sub-model on one canonical
    clock (see :attr:`SBMLProcess.time_scale`).

    Resolution order:

    1. The model-level ``timeUnits`` attribute (SBML L3).
    2. A ``<unitDefinition id="time">`` (the SBML L2 convention; this is
       where DallePezze 2014's ``second × 86400`` lives).
    3. The SBML default time unit, ``second`` → ``1.0``.

    A time unit is a product of ``<unit>`` terms; for a well-formed time
    definition that is a single ``second`` term with a multiplier (and
    optional power-of-ten scale). Returns ``1.0`` if the file cannot be
    parsed.
    """
    import libsbml

    doc = libsbml.SBMLReader().readSBMLFromFile(str(xml_path))
    model = doc.getModel()
    if model is None:
        return 1.0

    unit_def = model.getUnitDefinition(model.getTimeUnits() or "time")
    if unit_def is None:
        return 1.0

    seconds = 1.0
    for k in range(unit_def.getNumUnits()):
        u = unit_def.getUnit(k)
        seconds *= (
            u.getMultiplier() * 10.0 ** u.getScale()
        ) ** u.getExponent()
    return float(seconds)


def _strip_events(sbml_path: str) -> str:
    """Write an event-free copy of the SBML for sbmltoodejax.

    Events carry no ODE-core information (they only impose discrete state
    changes, imported separately by :mod:`hallsim.sbml_events`), so
    removing them lets the continuous model generate. Returns the original
    path unchanged when there are no events.
    """
    import os

    import libsbml

    doc = libsbml.SBMLReader().readSBMLFromFile(str(sbml_path))
    model = doc.getModel()
    if model is None or model.getNumEvents() == 0:
        return sbml_path
    while model.getNumEvents() > 0:
        model.removeEvent(0)
    cache_dir = os.path.expanduser("~/.cache/hallsim/converted")
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.basename(sbml_path)
    out_path = os.path.join(cache_dir, f"noevents_{base}")
    libsbml.writeSBMLToFile(doc, out_path)
    return out_path


def _load_local_sbml(sbml_path: str):
    """Load a local SBML XML file via sbmltoodejax.

    sbmltoodejax works by generating a Python module from SBML; this
    is the same approach load_biomodel uses internally, just with a
    local file instead of a BioModels API download.

    Returns (model, y0, w0, c) matching load_biomodel's signature.
    """
    import importlib.util
    import os
    import tempfile

    from sbmltoodejax.utils import ParseSBMLFile, GenerateModel

    # Flatten features that sbmltoodejax can't translate but libsbml
    # knows how to expand (currently: user-defined function definitions).
    sbml_path = _preprocess_sbml(sbml_path)
    # Events are imported separately (hallsim.sbml_events); strip them so
    # the ODE core generates cleanly.
    sbml_path = _strip_events(sbml_path)

    issues = _precheck_sbml_supported(sbml_path)
    if issues:
        bullets = "\n  - ".join(issues)
        raise UnsupportedSBMLFeatureError(
            f"Cannot import {sbml_path!r} via sbmltoodejax:\n  - {bullets}\n"
            f"See https://developmentalsystems.org/sbmltoodejax/why_use.html"
            f"#limitations"
        )

    model_data = ParseSBMLFile(sbml_path)

    # sbmltoodejax generates a .py file — use a temp location
    # Use home directory for temp files (avoid /tmp space issues)
    tmp_dir = os.path.expanduser("~/.cache/hallsim")
    os.makedirs(tmp_dir, exist_ok=True)
    fd, tmp_py = tempfile.mkstemp(
        suffix=".py", prefix="sbml_jax_", dir=tmp_dir
    )
    os.close(fd)
    try:
        GenerateModel(model_data, tmp_py)
        # Patch sbmltoodejax bug: some SBML models (e.g. Sivakumar2011
        # crosstalk BIOMD0000000398) contain MathML that sbmltoodejax
        # translates into bare `no.sqrt(...)` calls — an undefined name
        # `no` that is never imported.  This appears to come from the
        # SBML MathML namespace prefix "no" being emitted verbatim
        # instead of mapping to jax.numpy.  We patch the generated
        # source to alias `no` to `jax.numpy` before loading.
        with open(tmp_py, "r") as f:
            code = f.read()
        patched = False
        # Patch MathML namespace prefix "no" → jax.numpy
        if "\tno " in code or " no." in code or "\tno." in code:
            code = code.replace("import no\n", "import jax.numpy as no\n")
            if "import no" not in code:
                code = "import jax.numpy as no\n" + code
            patched = True
        # Patch deprecated eqx.static_field() → eqx.field(static=True)
        if "eqx.static_field()" in code:
            code = code.replace("eqx.static_field()", "eqx.field(static=True)")
            patched = True
        # Patch hardcoded float32 → float64. sbmltoodejax emits the
        # stoichiometric matrix, constants, and rate vectors as
        # ``dtype=jnp.float32``; an explicit dtype overrides the global
        # ``jax_enable_x64`` setting, so the RHS is computed at float32
        # precision (~1e-7) even when HallSim runs in float64. With an
        # adaptive controller at ``rtol=1e-6`` (below the float32 floor)
        # the embedded error estimate is dominated by float32 roundoff
        # and the step controller thrashes — ~57% step rejection that
        # masquerades as stiffness, defeating implicit solvers. float64
        # constants let the controller actually meet tolerance. (With
        # ``jax_enable_x64`` off, JAX downcasts float64 → float32, so this
        # is a no-op rather than a precision promotion.)
        if "float32" in code:
            code = code.replace("float32", "float64")
            patched = True
        if patched:
            with open(tmp_py, "w") as f:
                f.write(code)
        spec = importlib.util.spec_from_file_location(
            "_sbml_generated", tmp_py
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # sbmltoodejax versions use different class names
        model_cls = getattr(mod, "ModelSpec", None) or getattr(
            mod, "ModelStep", None
        )
        if model_cls is None:
            raise AttributeError(
                f"Generated SBML module has neither ModelSpec nor ModelStep. "
                f"Available: {[a for a in dir(mod) if not a.startswith('_')]}"
            )
        return model_cls(), mod.y0, mod.w0, mod.c
    finally:
        os.unlink(tmp_py)


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


def process_from_sbml(
    model_id: int | str,
    name: str | None = None,
    timescale: float | None = None,
    parameters: dict[str, float] | None = None,
) -> SBMLProcess:
    """Load an SBML model and wrap it as a Process.

    Parameters
    ----------
    model_id:
        Either a BioModels numeric ID (e.g. ``10`` for Kholodenko2000
        MAPK) or a path to a local SBML XML file.
    name:
        Human-readable name for the process. Defaults to
        ``"biomodel_{model_id}"`` or the filename.
    timescale:
        Characteristic timescale in hours for multi-rate scheduling.
        If ``None``, the process lands in the default group.
    parameters:
        Optional ``{c_name: initial_value}`` dict that overrides the
        SBML default for specific constants at construction. Each name
        must exist in ``c_indexes``. The returned Process's
        :attr:`SBMLProcess.parameters` field is auto-populated with
        the published default for **every** SBML constant, then
        ``parameters`` entries replace those defaults for the
        listed keys. After construction the dict supports the standard
        ``eqx.tree_at`` substitution path used by hallmarks
        (``param_name="parameters.<c_name>"``) and Calibrator.

    Returns
    -------
    SBMLProcess instance with auto-generated ports from SBML species.

    Raises
    ------
    ImportError
        If ``sbmltoodejax`` is not installed.
    KeyError
        If any name in ``parameters`` is not a constant in
        the SBML model.
    """
    try:
        import sbmltoodejax  # noqa: F401  (only checking availability)
    except ImportError:
        raise ImportError(
            "sbmltoodejax is required for SBML import. "
            "Install it with: pip install sbmltoodejax"
        )

    import os

    is_local_file = isinstance(model_id, str) and os.path.isfile(model_id)

    if is_local_file:
        name = name or os.path.splitext(os.path.basename(model_id))[0]
        log.info(f"Loading local SBML file '{model_id}' as '{name}'...")
        xml_path = model_id
    else:
        name = name or f"biomodel_{model_id}"
        log.info(f"Fetching BioModels #{model_id} as '{name}'...")
        xml_path = _download_biomodel_to_cache(model_id)

    # Single import path: both local files and downloaded BioModels go
    # through _load_local_sbml so the pre-check and the generated-module
    # patches (namespace alias, eqx.static_field) apply uniformly.
    model, y0, w0, c = _load_local_sbml(xml_path)

    # Extract species names from the model's index mapping
    # sbmltoodejax versions differ: ModelSpec uses model.modelstepfunc.y_indexes,
    # ModelStep puts y_indexes directly on the model
    if hasattr(model, "modelstepfunc") and hasattr(
        model.modelstepfunc, "y_indexes"
    ):
        y_indexes = model.modelstepfunc.y_indexes
    elif hasattr(model, "y_indexes"):
        y_indexes = model.y_indexes
    else:
        raise AttributeError(
            f"Cannot find y_indexes on model ({type(model).__name__}). "
            f"Available attrs: {[a for a in dir(model) if not a.startswith('_')]}"
        )
    # y_indexes maps species_name -> index; sort by index to get ordered names
    species_ordered = sorted(y_indexes.items(), key=lambda x: x[1])
    species_names = tuple(name for name, _ in species_ordered)

    # MIRIAM annotations on each species → Port.ontology, so the
    # composability analyzer can detect shared biology across imported
    # SBML models by their identifiers.org references.
    ontology_map = _extract_species_ontology(xml_path)
    coupling_meta = _extract_coupling_metadata(xml_path)
    species_ontology = tuple(ontology_map.get(s, {}) for s in species_names)

    native_time_seconds = _extract_native_time_seconds(xml_path)

    log.info(f"Loaded {len(species_names)} species: {species_names}")

    # Resolve any name-indexed accesses against the model's c_indexes.
    c_indexes = getattr(
        getattr(model, "modelstepfunc", model), "c_indexes", None
    ) or getattr(model, "c_indexes", None)
    w_indexes_map = (
        getattr(getattr(model, "modelstepfunc", model), "w_indexes", None)
        or getattr(model, "w_indexes", None)
        or {}
    )

    # Auto-populate `parameters` with every SBML constant at its published
    # default. This exposes the full mechanism surface for
    # Composite.calibration_targets / hallmark substitution / Calibrator
    # fitting without per-composite hand-curation.
    if c_indexes is None:
        params_dict = {}
        param_names = ()
        param_indexes = ()
    else:
        params_dict = {n: float(c[i]) for n, i in c_indexes.items()}
        param_names = tuple(c_indexes.keys())
        param_indexes = tuple(c_indexes[n] for n in param_names)

    # Boundary-input species (Irradiation, Insulin, …) are the model's
    # experimental input ports. Surface them through the same `parameters`
    # dict as constants, routed into the `w` vector at derivative time.
    boundary_inputs = _collect_boundary_inputs(xml_path) & set(w_indexes_map)
    params_dict.update(
        {n: float(w0[w_indexes_map[n]]) for n in boundary_inputs}
    )
    w_names = tuple(sorted(boundary_inputs))
    w_index_tuple = tuple(w_indexes_map[n] for n in w_names)

    # Inert sink species (written by degradation, read by nothing) are
    # frozen so they don't accumulate and ruin the state scaling.
    inert_sinks = _detect_inert_sinks(xml_path)
    frozen_indices = tuple(
        i for i, n in enumerate(species_names) if n in inert_sinks
    )
    if frozen_indices:
        log.warning(
            "%s: inert sink species %s are written but read by nothing; "
            "freezing them (treated as boundary). Consider marking "
            "boundaryCondition=true in the source SBML.",
            name,
            [species_names[i] for i in frozen_indices],
        )

    # Apply user overrides against the combined settable surface
    # (constants + boundary inputs).
    if parameters:
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

    proc = object.__new__(SBMLProcess)
    # Set fields directly (bypassing __init__ since this is a dynamic construction)
    object.__setattr__(proc, "_species_names", species_names)
    object.__setattr__(
        proc,
        "_species_y0",
        tuple(float(y0[i]) for i in range(len(species_names))),
    )
    object.__setattr__(proc, "_species_ontology", species_ontology)
    object.__setattr__(proc, "_coupling_meta", coupling_meta)
    object.__setattr__(proc, "native_time_seconds", native_time_seconds)
    object.__setattr__(proc, "time_scale", 1.0)
    object.__setattr__(proc, "_model", model)
    object.__setattr__(proc, "_w0", w0)
    object.__setattr__(proc, "_c", c)
    object.__setattr__(proc, "_name", name)
    object.__setattr__(proc, "parameters", params_dict)
    object.__setattr__(proc, "_param_names", param_names)
    object.__setattr__(proc, "_param_indexes", param_indexes)
    object.__setattr__(proc, "_w_names", w_names)
    object.__setattr__(proc, "_w_indexes", w_index_tuple)
    object.__setattr__(proc, "_frozen_indices", frozen_indices)
    # Default the scheduler timescale to the model's native time unit (a
    # day-scale model has day-scale dynamics) so auto_groups clusters
    # mixed-rate composites correctly. Never None for SBML processes, so
    # reconciled_to / tree_at can replace it without None-leaf ambiguity.
    object.__setattr__(
        proc,
        "timescale",
        float(timescale) if timescale is not None else native_time_seconds,
    )

    # Translate SBML <event> elements (stripped from the ODE core above)
    # into EVENT processes. Expand into a composite via
    # hallsim.sbml_events.expand_events(proc).
    from hallsim.sbml_events import translate_events

    events = translate_events(
        _preprocess_sbml(xml_path), species_names, params_dict, name
    )
    object.__setattr__(proc, "_events", tuple(events))
    if events:
        log.info(
            "%s: imported %d SBML event(s); compose with "
            "sbml_events.expand_events(proc).",
            name,
            len(events),
        )

    return proc
