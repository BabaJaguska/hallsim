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

import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process

log = logging.getLogger(__name__)


class SBMLProcess(Process):
    """Process auto-generated from an SBML model via sbmltoodejax.

    This wraps the JAX-compiled model step function from sbmltoodejax,
    exposing SBML species as EVOLVED ports.

    Not constructed directly — use :func:`process_from_sbml`.
    """

    _species_names: tuple[str, ...] = ()
    _species_y0: tuple[float, ...] = ()
    _model: Any = None  # sbmltoodejax model object
    _w0: Any = None
    _c: Any = None
    _name: str = ""

    def ports_schema(self):
        return {
            name: Port(
                role=PortRole.EVOLVED,
                default=float(y0),
                units="dimensionless",
                description=f"SBML species: {name}",
            )
            for name, y0 in zip(self._species_names, self._species_y0)
        }

    def derivative(self, t, state):
        # Build y vector in the order sbmltoodejax expects
        y = jnp.array([state[name] for name in self._species_names])
        # sbmltoodejax versions: ModelSpec uses model.modelstepfunc.ratefunc,
        # ModelStep has ratefunc directly
        ratefunc = getattr(
            getattr(self._model, "modelstepfunc", self._model),
            "ratefunc",
        )
        dydt = ratefunc(y, t, self._w0, self._c)
        return {
            name: dydt[i]
            for i, name in enumerate(self._species_names)
        }

    def metadata(self):
        base = super().metadata()
        base["sbml_name"] = self._name
        base["n_species"] = len(self._species_names)
        return base


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

    model_data = ParseSBMLFile(sbml_path)

    # sbmltoodejax generates a .py file — use a temp location
    # Use home directory for temp files (avoid /tmp space issues)
    tmp_dir = os.path.expanduser("~/.cache/hallsim")
    os.makedirs(tmp_dir, exist_ok=True)
    fd, tmp_py = tempfile.mkstemp(suffix=".py", prefix="sbml_jax_", dir=tmp_dir)
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
        spec = importlib.util.spec_from_file_location("_sbml_generated", tmp_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # sbmltoodejax versions use different class names
        model_cls = getattr(mod, "ModelSpec", None) or getattr(mod, "ModelStep", None)
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

    Returns
    -------
    SBMLProcess instance with auto-generated ports from SBML species.

    Raises
    ------
    ImportError
        If ``sbmltoodejax`` is not installed.
    """
    try:
        from sbmltoodejax.utils import load_biomodel
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
        model, y0, w0, c = _load_local_sbml(model_id)
    else:
        name = name or f"biomodel_{model_id}"
        log.info(f"Loading BioModels #{model_id} as '{name}'...")
        model, y0, w0, c = load_biomodel(model_id)

    # Extract species names from the model's index mapping
    # sbmltoodejax versions differ: ModelSpec uses model.modelstepfunc.y_indexes,
    # ModelStep puts y_indexes directly on the model
    if hasattr(model, "modelstepfunc") and hasattr(model.modelstepfunc, "y_indexes"):
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

    log.info(f"Loaded {len(species_names)} species: {species_names}")

    proc = object.__new__(SBMLProcess)
    # Set fields directly (bypassing __init__ since this is a dynamic construction)
    object.__setattr__(proc, '_species_names', species_names)
    object.__setattr__(proc, '_species_y0', tuple(float(y0[i]) for i in range(len(species_names))))
    object.__setattr__(proc, '_model', model)
    object.__setattr__(proc, '_w0', w0)
    object.__setattr__(proc, '_c', c)
    object.__setattr__(proc, '_name', name)
    if timescale is not None:
        object.__setattr__(proc, 'timescale', timescale)

    return proc
