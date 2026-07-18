"""Translate SBML ``<event>`` elements into HallSim EVENT processes.

``sbmltoodejax`` cannot import SBML events, so :func:`process_from_sbml`
strips them for the ODE core and this module re-expresses each event as an
:class:`SBMLEvent` (``ProcessKind.EVENT``): the trigger becomes
``condition``, the event assignments become a ``handler`` that mutates the
target species' store paths. Event math (trigger + assignment RHS) is
compiled from the libsbml AST into a small pure-Python IR — no libsbml
objects are retained — and evaluated with ``jax.numpy`` so it stays
shape-polymorphic under batched runs.

Supported: triggers over time and species; assignments to species.
Assignments to plain parameters are reported and skipped (they need
LATCHED parameter promotion — a follow-up). Delays and priorities raise.
"""

from __future__ import annotations

import functools
import logging
import math

import equinox as eqx
import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process, ProcessKind

log = logging.getLogger(__name__)


class UnsupportedEventFeatureError(Exception):
    """An SBML event uses a construct the translator does not handle."""


# ── libsbml AST → pure-Python IR ───────────────────────────────────
#
# IR nodes are plain tuples so the compiled form retains no libsbml
# objects (whose lifetime is tied to the SBMLDocument) and is safe as an
# Equinox static field. Numeric ops use jax.numpy in the evaluator.


def _compile_ast(node, species: set) -> tuple:
    """Compile a libsbml ASTNode to the tuple IR (see :func:`_eval_ir`)."""
    import libsbml

    ty = node.getType()
    kids = [node.getChild(i) for i in range(node.getNumChildren())]

    def C(k):
        return _compile_ast(k, species)

    if ty == libsbml.AST_NAME:
        nm = node.getName()
        if nm in species:
            return ("var", nm)
        if nm.lower() in ("time", "t"):
            return ("time",)
        return ("name", nm)  # a parameter — resolved from consts at build
    if ty == libsbml.AST_NAME_TIME:
        return ("time",)
    if ty == libsbml.AST_INTEGER:
        return ("const", float(node.getInteger()))
    if ty in (libsbml.AST_REAL, libsbml.AST_REAL_E, libsbml.AST_RATIONAL):
        return ("const", float(node.getValue()))
    if ty == libsbml.AST_CONSTANT_TRUE:
        return ("const", 1.0)
    if ty == libsbml.AST_CONSTANT_FALSE:
        return ("const", 0.0)
    if ty == libsbml.AST_CONSTANT_PI:
        return ("const", math.pi)
    if ty == libsbml.AST_CONSTANT_E:
        return ("const", math.e)

    nary = {
        libsbml.AST_PLUS: "add",
        libsbml.AST_TIMES: "mul",
        libsbml.AST_LOGICAL_AND: "and",
        libsbml.AST_LOGICAL_OR: "or",
    }
    binary = {
        libsbml.AST_DIVIDE: "div",
        libsbml.AST_POWER: "pow",
        libsbml.AST_FUNCTION_POWER: "pow",
        libsbml.AST_RELATIONAL_GEQ: "geq",
        libsbml.AST_RELATIONAL_LEQ: "leq",
        libsbml.AST_RELATIONAL_GT: "gt",
        libsbml.AST_RELATIONAL_LT: "lt",
        libsbml.AST_RELATIONAL_EQ: "eq",
        libsbml.AST_RELATIONAL_NEQ: "neq",
    }
    funcs = {
        libsbml.AST_FUNCTION_EXP: "exp",
        libsbml.AST_FUNCTION_LN: "log",
        libsbml.AST_FUNCTION_LOG: "log10",
        libsbml.AST_FUNCTION_ABS: "abs",
        libsbml.AST_FUNCTION_ROOT: "sqrt",
        libsbml.AST_FUNCTION_SIN: "sin",
        libsbml.AST_FUNCTION_COS: "cos",
        libsbml.AST_FUNCTION_TAN: "tan",
    }

    if ty == libsbml.AST_MINUS:
        return (
            ("neg", C(kids[0]))
            if len(kids) == 1
            else ("sub", C(kids[0]), C(kids[1]))
        )
    if ty in nary:
        return (nary[ty], [C(k) for k in kids])
    if ty in binary:
        return (binary[ty], C(kids[0]), C(kids[1]))
    if ty == libsbml.AST_LOGICAL_NOT:
        return ("not", C(kids[0]))
    if ty in funcs:
        return ("func", funcs[ty], [C(k) for k in kids])
    raise UnsupportedEventFeatureError(
        f"unhandled MathML node type {ty} in event expression"
    )


def _bake_consts(ir: tuple, consts: dict) -> tuple:
    """Replace ``('name', p)`` leaves with the parameter's constant value."""
    tag = ir[0]
    if tag == "name":
        if ir[1] not in consts:
            raise UnsupportedEventFeatureError(
                f"event math references unknown symbol {ir[1]!r}"
            )
        return ("const", float(consts[ir[1]]))
    if tag in ("const", "var", "time"):
        return ir
    if tag in ("add", "mul", "and", "or", "func"):
        head = ir[:-1]
        return (*head, [_bake_consts(k, consts) for k in ir[-1]])
    return (tag, *[_bake_consts(k, consts) for k in ir[1:]])


def _collect_species(ir: tuple, out: set) -> None:
    tag = ir[0]
    if tag == "var":
        out.add(ir[1])
    elif tag in ("add", "mul", "and", "or", "func"):
        for k in ir[-1]:
            _collect_species(k, out)
    elif tag in ("const", "time", "name"):
        return
    else:
        for k in ir[1:]:
            _collect_species(k, out)


_FUNCS = {
    "exp": jnp.exp,
    "log": jnp.log,
    "log10": jnp.log10,
    "abs": jnp.abs,
    "sqrt": jnp.sqrt,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "tan": jnp.tan,
}


def _eval_ir(ir: tuple, t, view: dict):
    """Evaluate the IR at time ``t`` against a ``{species: value}`` view."""
    tag = ir[0]
    if tag == "const":
        return ir[1]
    if tag == "var":
        return view[ir[1]]
    if tag == "time":
        return t
    if tag == "add":
        return functools.reduce(
            lambda a, b: a + b, (_eval_ir(k, t, view) for k in ir[1])
        )
    if tag == "mul":
        return functools.reduce(
            lambda a, b: a * b, (_eval_ir(k, t, view) for k in ir[1])
        )
    if tag == "sub":
        return _eval_ir(ir[1], t, view) - _eval_ir(ir[2], t, view)
    if tag == "div":
        return _eval_ir(ir[1], t, view) / _eval_ir(ir[2], t, view)
    if tag == "neg":
        return -_eval_ir(ir[1], t, view)
    if tag == "pow":
        return _eval_ir(ir[1], t, view) ** _eval_ir(ir[2], t, view)
    if tag == "geq":
        return _eval_ir(ir[1], t, view) >= _eval_ir(ir[2], t, view)
    if tag == "leq":
        return _eval_ir(ir[1], t, view) <= _eval_ir(ir[2], t, view)
    if tag == "gt":
        return _eval_ir(ir[1], t, view) > _eval_ir(ir[2], t, view)
    if tag == "lt":
        return _eval_ir(ir[1], t, view) < _eval_ir(ir[2], t, view)
    if tag == "eq":
        return _eval_ir(ir[1], t, view) == _eval_ir(ir[2], t, view)
    if tag == "neq":
        return _eval_ir(ir[1], t, view) != _eval_ir(ir[2], t, view)
    if tag == "and":
        return functools.reduce(
            jnp.logical_and, (_eval_ir(k, t, view) for k in ir[1])
        )
    if tag == "or":
        return functools.reduce(
            jnp.logical_or, (_eval_ir(k, t, view) for k in ir[1])
        )
    if tag == "not":
        return jnp.logical_not(_eval_ir(ir[1], t, view))
    if tag == "func":
        return _FUNCS[ir[1]](*[_eval_ir(k, t, view) for k in ir[2]])
    raise UnsupportedEventFeatureError(f"cannot evaluate IR node {tag!r}")


# ── the EVENT process ──────────────────────────────────────────────


class SBMLEvent(Process):
    """One SBML ``<event>`` as a HallSim EVENT process.

    Reads referenced species through INPUT ports (named by species id) and
    writes each assignment target through a LATCHED ``__set_<species>``
    port. The handler applies ``target := rhs`` as an additive delta
    ``rhs − current`` (the scheduler scatter-adds), i.e. a true assignment.
    """

    kind: ProcessKind = ProcessKind.EVENT
    _name: str = ""
    _trigger_ir: tuple = eqx.field(static=True, default=())
    _assign_ir: tuple = eqx.field(static=True, default=())  # ((tgt, rhs),..)
    _read_species: tuple = eqx.field(static=True, default=())

    def ports_schema(self):
        ports = {
            s: Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description=f"reads species {s}",
            )
            for s in self._read_species
        }
        for tgt, _ in self._assign_ir:
            ports[f"__set_{tgt}"] = Port(
                role=PortRole.LATCHED,
                default=0.0,
                units="dimensionless",
                description=f"event assignment target {tgt}",
            )
        return ports

    def condition(self, t, state):
        return _eval_ir(self._trigger_ir, t, state)

    def handler(self, t, state):
        return {
            f"__set_{tgt}": _eval_ir(rhs, t, state) - state[tgt]
            for tgt, rhs in self._assign_ir
        }

    def metadata(self):
        base = super().metadata()
        base["event_targets"] = [t for t, _ in self._assign_ir]
        return base


def translate_events(
    xml_path: str, species_names, consts: dict, model_name: str
) -> list[SBMLEvent]:
    """Read the SBML at ``xml_path`` and return one SBMLEvent per event.

    ``species_names`` is the model's ordered species ids; ``consts`` maps
    parameter names to their (constant) values, baked into event math.
    Parameter-target assignments are logged and skipped.
    """
    import libsbml

    doc = libsbml.SBMLReader().readSBMLFromFile(str(xml_path))
    model = doc.getModel()
    if model is None:
        return []
    species = set(species_names)
    out: list[SBMLEvent] = []
    for i in range(model.getNumEvents()):
        ev = model.getEvent(i)
        eid = ev.getId() or f"event{i}"
        if ev.getDelay() is not None:
            raise UnsupportedEventFeatureError(
                f"event {eid!r} on {model_name} has a delay — not supported"
            )
        if ev.isSetPriority():
            raise UnsupportedEventFeatureError(
                f"event {eid!r} on {model_name} has a priority — "
                "not supported"
            )
        trigger = _bake_consts(
            _compile_ast(ev.getTrigger().getMath(), species), consts
        )
        read: set = set()
        _collect_species(trigger, read)
        assigns = []
        for j in range(ev.getNumEventAssignments()):
            ea = ev.getEventAssignment(j)
            var = ea.getVariable()
            if var not in species:
                log.warning(
                    "SBML event %r on %s assigns to non-species %r "
                    "(parameter target) — skipped; needs LATCHED param "
                    "promotion.",
                    eid,
                    model_name,
                    var,
                )
                continue
            rhs = _bake_consts(_compile_ast(ea.getMath(), species), consts)
            _collect_species(rhs, read)
            read.add(var)  # current value needed for the assignment delta
            assigns.append((var, rhs))
        if not assigns:
            log.warning(
                "SBML event %r on %s has no species assignments — skipped.",
                eid,
                model_name,
            )
            continue
        out.append(
            SBMLEvent(
                _name=f"{model_name}__{eid}",
                _trigger_ir=trigger,
                _assign_ir=tuple(assigns),
                _read_species=tuple(sorted(read & species)),
            )
        )
        log.info(
            "Translated SBML event %r on %s (targets: %s).",
            eid,
            model_name,
            [t for t, _ in assigns],
        )
    return out


def expand_events(proc) -> tuple[dict, dict]:
    """``(extra_processes, topology)`` composing an SBMLProcess's events.

    Each event's INPUT/LATCHED ports are wired to the owning model's
    ``<name>/<species>`` store paths. Empty when the model has no events.
    """
    procs: dict = {}
    topo: dict = {}
    for ev in getattr(proc, "_events", ()):
        procs[ev._name] = ev
        wiring = {s: f"{proc._name}/{s}" for s in ev._read_species}
        for tgt, _ in ev._assign_ir:
            wiring[f"__set_{tgt}"] = f"{proc._name}/{tgt}"
        topo[ev._name] = wiring
    return procs, topo
