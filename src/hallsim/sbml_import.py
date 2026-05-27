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

import jax
import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process

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

    n_events = model.getNumEvents()
    if n_events > 0:
        ids = [
            model.getEvent(i).getId() or "<unnamed>" for i in range(n_events)
        ]
        issues.append(
            f"contains {n_events} <event> element(s) "
            f"(ids: {', '.join(ids)}); discrete state changes are not "
            f"translated by sbmltoodejax"
        )

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


class SBMLProcess(Process):
    """Process auto-generated from an SBML model via sbmltoodejax.

    This wraps the JAX-compiled model step function from sbmltoodejax,
    exposing SBML species as EVOLVED ports.

    Not constructed directly — use :func:`process_from_sbml`.

    Parameters
    ----------
    tunable_params:
        Names of SBML constants (entries in the model's ``c`` array) that
        should be exposed as INPUT ports rather than read from the SBML
        defaults. When this is non-empty, the derivative reads each named
        parameter from the state dict and substitutes it into ``c`` before
        calling the SBML rate function. This is what lets an upstream
        Process (e.g. a damage-accumulation module) drive an SBML model's
        constants — the connection that powers HallSim's cross-publication
        composability story.

    tunable_param_indexes:
        Indices in ``c`` for each name in ``tunable_params``, in the same
        order. Set by :func:`process_from_sbml`; do not pass directly.
    """

    _species_names: tuple[str, ...] = ()
    _species_y0: tuple[float, ...] = ()
    _species_ontology: tuple[dict[str, str], ...] = ()
    _model: Any = None  # sbmltoodejax model object
    _w0: Any = None
    _c: Any = None
    _name: str = ""
    _tunable_params: tuple[str, ...] = ()
    _tunable_param_indexes: tuple[int, ...] = ()
    _tunable_param_defaults: tuple[float, ...] = ()
    # parameter_overrides is a dict-of-floats field consumed via
    # eqx.tree_at by the hallmark machinery. Each key is an SBML
    # constant name (must exist in the model's c_indexes); values are
    # substituted into the `c` array at derivative time. This is how
    # a HallmarkHandle parameterises a specific SBML rate constant.
    parameter_overrides: dict[str, float] = None  # type: ignore[assignment]
    _override_indexes: tuple[int, ...] = ()

    def ports_schema(self):
        ports = {
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
        for name, default in zip(
            self._tunable_params, self._tunable_param_defaults
        ):
            ports[name] = Port(
                role=PortRole.INPUT,
                default=float(default),
                units="dimensionless",
                description=f"SBML tunable constant: {name}",
            )
        return ports

    def derivative(self, t, state):
        # Stack species along the trailing axis. For scalar state values
        # this yields shape (n_species,); for batched values (batch,) it
        # yields (batch, n_species). The trailing-axis convention matches
        # Composite.flatten/unflatten, so this Process is shape-polymorphic
        # and works under jax.vmap'd / batched Scheduler runs without
        # extra vmap'ing at the composite level.
        y = jnp.stack([state[name] for name in self._species_names], axis=-1)
        ratefunc = getattr(
            getattr(self._model, "modelstepfunc", self._model),
            "ratefunc",
        )

        # Batched state is signalled by y having a leading batch axis.
        # ratefunc was generated by sbmltoodejax for unbatched (n_species,)
        # input, so we vmap it for batched calls. The condition is on
        # y.ndim (a compile-time static property under JIT), not on a
        # traced value, so this is JIT-safe.
        is_batched = y.ndim > 1

        # Build the constants vector. Two override sources stack here:
        # parameter_overrides (static, hallmark-driven) is applied first,
        # then tunable_params (dynamic, state-driven). State-driven
        # values win when a name appears in both — most recent
        # substitution dominates.
        needs_batched_c = is_batched and self._tunable_params
        if needs_batched_c:
            c = jnp.broadcast_to(self._c, y.shape[:-1] + self._c.shape)
        else:
            c = self._c

        if self.parameter_overrides and self._override_indexes:
            override_names = tuple(self.parameter_overrides.keys())
            for name, idx in zip(override_names, self._override_indexes):
                value = self.parameter_overrides[name]
                if needs_batched_c:
                    c = c.at[..., idx].set(value)
                else:
                    c = c.at[idx].set(value)

        if self._tunable_params:
            for name, idx in zip(
                self._tunable_params, self._tunable_param_indexes
            ):
                if needs_batched_c:
                    c = c.at[..., idx].set(state[name])
                else:
                    c = c.at[idx].set(state[name])

        if is_batched:
            # vmap ratefunc and (if present) the batched c along axis 0.
            in_axes = (0, None, None, 0 if self._tunable_params else None)
            dydt = jax.vmap(ratefunc, in_axes=in_axes)(y, t, self._w0, c)
        else:
            dydt = ratefunc(y, t, self._w0, c)

        return {
            name: dydt[..., i] for i, name in enumerate(self._species_names)
        }

    def metadata(self):
        base = super().metadata()
        base["sbml_name"] = self._name
        base["n_species"] = len(self._species_names)
        if self._tunable_params:
            base["tunable_params"] = list(self._tunable_params)
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


def process_from_sbml(
    model_id: int | str,
    name: str | None = None,
    timescale: float | None = None,
    tunable_params: tuple[str, ...] = (),
    parameter_overrides: dict[str, float] | None = None,
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
    tunable_params:
        Names of SBML constants to expose as INPUT ports. Each named
        parameter must exist in the model's ``c_indexes`` mapping. When
        non-empty, the returned process will read these values from the
        state dict on every derivative call and substitute them into the
        ``c`` array before invoking the SBML rate function. This is the
        mechanism for letting an upstream HallSim Process drive a
        constant of an imported SBML model (e.g. damage signal driving a
        DDR model's input).
    parameter_overrides:
        Optional ``{c_name: initial_value}`` dict exposing specific
        SBML constants as a single :attr:`SBMLProcess.parameter_overrides`
        eqx field. Each name must exist in ``c_indexes``. Substitution
        is static (not state-driven): the value comes from the field at
        derivative time, so :class:`hallsim.hallmarks.HallmarkHandle`
        can target it via ``param_name="parameter_overrides.<c_name>"``
        and update it through ``eqx.tree_at``. If both ``tunable_params``
        and ``parameter_overrides`` reference the same constant, the
        state-driven (tunable) value wins.

    Returns
    -------
    SBMLProcess instance with auto-generated ports from SBML species.

    Raises
    ------
    ImportError
        If ``sbmltoodejax`` is not installed.
    KeyError
        If any name in ``tunable_params`` or ``parameter_overrides`` is
        not a constant in the SBML model.
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
    species_ontology = tuple(ontology_map.get(s, {}) for s in species_names)

    log.info(f"Loaded {len(species_names)} species: {species_names}")

    # Resolve any name-indexed accesses against the model's c_indexes.
    c_indexes = getattr(
        getattr(model, "modelstepfunc", model), "c_indexes", None
    ) or getattr(model, "c_indexes", None)

    if tunable_params:
        if c_indexes is None:
            raise AttributeError(
                f"Cannot expose tunable params: model has no c_indexes "
                f"({type(model).__name__})"
            )
        missing = [p for p in tunable_params if p not in c_indexes]
        if missing:
            raise KeyError(
                f"Tunable param(s) {missing} not found in SBML constants. "
                f"Available: {sorted(c_indexes.keys())}"
            )
        tunable_indexes = tuple(c_indexes[p] for p in tunable_params)
        tunable_defaults = tuple(float(c[i]) for i in tunable_indexes)
    else:
        tunable_indexes = ()
        tunable_defaults = ()

    if parameter_overrides:
        if c_indexes is None:
            raise AttributeError(
                f"Cannot expose parameter overrides: model has no "
                f"c_indexes ({type(model).__name__})"
            )
        missing = [p for p in parameter_overrides if p not in c_indexes]
        if missing:
            raise KeyError(
                f"parameter_overrides {missing} not found in SBML "
                f"constants. Available: {sorted(c_indexes.keys())}"
            )
        override_dict = {
            n: float(parameter_overrides[n]) for n in parameter_overrides
        }
        override_indexes = tuple(c_indexes[n] for n in parameter_overrides)
    else:
        override_dict = {}
        override_indexes = ()

    proc = object.__new__(SBMLProcess)
    # Set fields directly (bypassing __init__ since this is a dynamic construction)
    object.__setattr__(proc, "_species_names", species_names)
    object.__setattr__(
        proc,
        "_species_y0",
        tuple(float(y0[i]) for i in range(len(species_names))),
    )
    object.__setattr__(proc, "_species_ontology", species_ontology)
    object.__setattr__(proc, "_model", model)
    object.__setattr__(proc, "_w0", w0)
    object.__setattr__(proc, "_c", c)
    object.__setattr__(proc, "_name", name)
    object.__setattr__(proc, "_tunable_params", tuple(tunable_params))
    object.__setattr__(proc, "_tunable_param_indexes", tunable_indexes)
    object.__setattr__(proc, "_tunable_param_defaults", tunable_defaults)
    object.__setattr__(proc, "parameter_overrides", override_dict)
    object.__setattr__(proc, "_override_indexes", override_indexes)
    if timescale is not None:
        object.__setattr__(proc, "timescale", timescale)

    return proc
