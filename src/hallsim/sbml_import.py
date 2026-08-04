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


class SBMLProcess(ImportedODEProcess):
    """Process auto-generated from an SBML model via sbmltoodejax, exposing
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
    _species_ontology: tuple[dict[str, str], ...] = eqx.field(
        static=True, default=()
    )
    # Param constancy + SBO, dynamic variables, and the assignment-rule graph,
    # so a driver aimed at a rate constant the model modulates via a rule can
    # be flagged (see hallsim.coupling_wiring).
    _coupling_meta: dict = eqx.field(static=True, default=None)
    _model: Any = None  # sbmltoodejax model object
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
    # Inert sinks: written by degradation, read by nothing. Frozen to dy/dt=0
    # so they can't accumulate unboundedly and wreck the state scaling — exact,
    # since no rate law reads them.
    _frozen_indices: tuple[int, ...] = eqx.field(static=True, default=())
    # Translated SBML <event> elements; expand with sbml_events.expand_events.
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

    def native_input_exposure(self, input_name, t_start, t_end, *, n=8000):
        """``∫ native-drive dt`` for boundary input ``input_name`` over
        composite time ``[t_start, t_end]`` — the exposure the model's driven
        rates were calibrated to. A forcing source delivering a very different
        integrated exposure runs the model off that calibration;
        :func:`hallsim.models.forcing.drive_pulse` compares against this and
        warns. Returns 0.0 if the input has no time-dependent assignment
        rule."""
        host = getattr(self._model, "modelstepfunc", self._model)
        af = getattr(host, "assignmentfunc", None)
        if af is None or t_end <= t_start:
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
        host = getattr(self._model, "modelstepfunc", self._model)
        ratefunc = host.ratefunc
        assignmentfunc = getattr(host, "assignmentfunc", None)
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


_GENERATED_MODELS: dict = {}


def _load_local_sbml(sbml_path: str):
    """Load a local SBML file via sbmltoodejax's codegen, returning
    ``(model, y0, w0, c)`` to match ``load_biomodel``.

    Cached per source file (path + mtime + size): each generated module defines
    a *new* model class, so importing the same SBML twice would otherwise give
    one model two pytree node types and share no compiled solve.
    """
    import os

    try:
        st = os.stat(sbml_path)
        key = (os.path.abspath(sbml_path), st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is not None and key in _GENERATED_MODELS:
        model_cls, y0, w0, c = _GENERATED_MODELS[key]
        return model_cls(), y0, w0, c

    model_cls, y0, w0, c = _generate_model_module(sbml_path)
    if key is not None:
        _GENERATED_MODELS[key] = (model_cls, y0, w0, c)
    return model_cls(), y0, w0, c


def _generate_model_module(sbml_path: str):
    """Run sbmltoodejax codegen and import the result, returning
    ``(model_cls, y0, w0, c)``. Patches the generated source on the way in —
    see the inline notes for each."""
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

    tmp_dir = os.path.expanduser("~/.cache/hallsim")
    os.makedirs(tmp_dir, exist_ok=True)
    fd, tmp_py = tempfile.mkstemp(
        suffix=".py", prefix="sbml_jax_", dir=tmp_dir
    )
    os.close(fd)
    try:
        GenerateModel(model_data, tmp_py)
        with open(tmp_py, "r") as f:
            code = f.read()
        patched = False
        # Some models (Sivakumar2011) emit bare `no.sqrt(...)`: the MathML
        # namespace prefix passes through instead of mapping to jax.numpy.
        if "\tno " in code or " no." in code or "\tno." in code:
            code = code.replace("import no\n", "import jax.numpy as no\n")
            if "import no" not in code:
                code = "import jax.numpy as no\n" + code
            patched = True
        if "eqx.static_field()" in code:
            code = code.replace("eqx.static_field()", "eqx.field(static=True)")
            patched = True
        # sbmltoodejax hardcodes dtype=jnp.float32, which overrides
        # jax_enable_x64. At rtol=1e-6 — below the float32 floor — the error
        # estimate is then dominated by roundoff and the controller thrashes
        # (~57% rejection masquerading as stiffness). No-op with x64 off.
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
        return model_cls, mod.y0, mod.w0, mod.c
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


def _resolve_source(model_id, name):
    """``(xml_path, name)`` for a local file path or a BioModels ID."""
    import os

    if isinstance(model_id, str) and os.path.isfile(model_id):
        name = name or os.path.splitext(os.path.basename(model_id))[0]
        log.info(f"Loading local SBML file '{model_id}' as '{name}'...")
        return model_id, name
    name = name or f"biomodel_{model_id}"
    log.info(f"Fetching BioModels #{model_id} as '{name}'...")
    return _download_biomodel_to_cache(model_id), name


def _ordered_species(model) -> tuple[str, ...]:
    """Species names in state-vector order, from the model's ``y_indexes``.

    sbmltoodejax versions differ: ModelSpec exposes it on ``modelstepfunc``,
    ModelStep directly on the model.
    """
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
    return tuple(n for n, _ in sorted(y_indexes.items(), key=lambda x: x[1]))


def _index_maps(model):
    """``(c_indexes, w_indexes)`` — the model's constant and boundary maps."""
    host = getattr(model, "modelstepfunc", model)
    c_indexes = getattr(host, "c_indexes", None) or getattr(
        model, "c_indexes", None
    )
    w_indexes = (
        getattr(host, "w_indexes", None)
        or getattr(model, "w_indexes", None)
        or {}
    )
    return c_indexes, w_indexes


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
    return (
        params_dict,
        param_names,
        param_indexes,
        w_names,
        tuple(w_indexes[n] for n in w_names),
        boundary_inputs,
    )


def _frozen_sink_indices(xml_path, species_names, name) -> tuple[int, ...]:
    """Indices of inert sinks — written by degradation, read by nothing.
    Frozen so they don't accumulate and ruin the state scaling."""
    inert = _detect_inert_sinks(xml_path)
    frozen = tuple(i for i, n in enumerate(species_names) if n in inert)
    if frozen:
        log.warning(
            "%s: inert sink species %s are written but read by nothing; "
            "freezing them (treated as boundary). Consider marking "
            "boundaryCondition=true in the source SBML.",
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


def _native_clock(xml_path, name):
    """``(seconds_per_native_unit, declared)``, warning loudly when undeclared —
    an assumed clock is silently 60x/3600x/86400x wrong once reconciled."""
    seconds, declared = _extract_native_time_seconds(xml_path)
    if not declared:
        log.warning(
            "%s: SBML declares no time unit; assuming native_time_seconds=1.0 "
            "(seconds). If this model's rate laws are in minutes/hours/days its "
            "clock is now a GUESS — reconciling it onto a shared canonical axis "
            "will be silently 60x/3600x/86400x wrong. Set the source SBML's "
            "timeUnits, or pass the true seconds-per-unit if you know it. "
            "Check `proc.native_time_declared` before composing.",
            name,
        )
    return seconds, declared


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

    Returns an :class:`SBMLProcess` with ports auto-generated from the species.
    Raises ``ImportError`` without sbmltoodejax, ``KeyError`` on an unknown
    parameter name.
    """
    try:
        import sbmltoodejax  # noqa: F401  (only checking availability)
    except ImportError:
        raise ImportError(
            "sbmltoodejax is required for SBML import. "
            "Install it with: pip install sbmltoodejax"
        )

    xml_path, name = _resolve_source(model_id, name)

    # Single import path: both local files and downloaded BioModels go
    # through _load_local_sbml so the pre-check and the generated-module
    # patches (namespace alias, eqx.static_field) apply uniformly.
    model, y0, w0, c = _load_local_sbml(xml_path)
    species_names = _ordered_species(model)
    log.info(f"Loaded {len(species_names)} species: {species_names}")

    # MIRIAM annotations on each species → Port.ontology, so the
    # composability analyzer can detect shared biology across imported
    # SBML models by their identifiers.org references.
    ontology_map = _extract_species_ontology(xml_path)
    coupling_meta = _extract_coupling_metadata(xml_path)
    species_ontology = tuple(ontology_map.get(s, {}) for s in species_names)

    native_time_seconds, native_time_declared = _native_clock(xml_path, name)

    c_indexes, w_indexes_map = _index_maps(model)
    (
        params_dict,
        param_names,
        param_indexes,
        w_names,
        w_index_tuple,
        boundary_inputs,
    ) = _settable_surface(xml_path, c, w0, c_indexes, w_indexes_map)
    frozen_indices = _frozen_sink_indices(xml_path, species_names, name)
    _apply_parameter_overrides(
        params_dict, parameters, c_indexes, boundary_inputs
    )

    # Translate SBML <event> elements (stripped from the ODE core above)
    # into EVENT processes. Expand into a composite via
    # hallsim.sbml_events.expand_events(proc).
    from hallsim.sbml_events import translate_events

    events = translate_events(
        _preprocess_sbml(xml_path), species_names, params_dict, name
    )
    # Through __init__, never object.__new__ + setattr: JAX rebuilds this pytree
    # at every jit/partition boundary, and a field-by-field instance does not
    # match what tree_unflatten produces — its structure shifts on the
    # round-trip and eqx.partition rejects it.
    proc = SBMLProcess(
        _species_names=species_names,
        _species_y0=tuple(float(y0[i]) for i in range(len(species_names))),
        _species_ontology=species_ontology,
        _coupling_meta=coupling_meta,
        native_time_seconds=native_time_seconds,
        native_time_declared=native_time_declared,
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
        _frozen_indices=frozen_indices,
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
