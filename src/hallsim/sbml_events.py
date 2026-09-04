"""Translate SBML ``<event>`` elements into HallSim EVENT processes.

``sbmltoodejax`` cannot import SBML events, so :func:`process_from_sbml`
strips them for the ODE core and this module re-expresses each event as an
:class:`SBMLEvent` (``ProcessKind.EVENT``): the trigger becomes
``condition``, the event assignments become a ``handler`` that mutates the
target species' store paths. Event math (trigger + assignment RHS) is
compiled from the libsbml AST into a small pure-Python IR — no libsbml
objects are retained — and evaluated with ``jax.numpy`` so it stays
shape-polymorphic under batched runs.

Supported: triggers over time and species; assignments to species and to
parameters. A parameter target is promoted on the owning process via
:meth:`~hallsim.imported.ImportedODEProcess.with_param_input` by
:func:`expand_events`, so the assignment reaches the rate laws through a
store path rather than being dropped — SBML models routinely deliver a dose
that way. Symbols a parameter's own assignment rule defines are folded to
constants first (:func:`fold_constant_rules`), which is how COPASI writes a
ModelValue. A nonzero delay and any priority raise; ``<delay>0</delay>``,
which COPASI emits on every event, is not a delay.
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


def _delay_seconds(event) -> float:
    """The event's delay as a number, or NaN when it is not a constant.

    Returns 0.0 for no ``<delay>`` element and for one whose math evaluates to
    a literal zero — the form COPASI emits unconditionally. A state- or
    time-dependent delay is not constant and comes back NaN, which compares
    unequal to zero and so is rejected by the caller.
    """
    import libsbml

    delay = event.getDelay()
    if delay is None or not delay.isSetMath():
        return 0.0
    math = delay.getMath()
    if math.isNumber():
        return float(libsbml.formulaToL3String(math))
    return float("nan")


def _has_time(ir: tuple) -> bool:
    """True if the IR reads the time symbol anywhere."""
    tag = ir[0]
    if tag == "time":
        return True
    if tag in ("const", "var", "name"):
        return False
    kids = ir[-1] if tag in ("add", "mul", "and", "or", "func") else ir[1:]
    return any(_has_time(k) for k in kids)


def fold_constant_rules(model, species, consts: dict) -> dict:
    """``consts`` extended with parameters an assignment rule defines.

    COPASI exports a "ModelValue" as a non-constant parameter plus an
    assignment rule (``DNAdamagefoci_0 = Gy * FociPerGy``), so a symbol that
    is constant in every meaningful sense is absent from the constant table
    and event math referencing it fails to resolve. Rules whose right-hand
    side reduces to numbers are folded here, to a fixpoint so a rule may
    depend on another rule. A rule that reads a species or the time symbol is
    genuinely dynamic and is left out.
    """
    pending = {}
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        if not (rule.isAssignment() and rule.isSetVariable()):
            continue
        if not rule.isSetMath() or rule.getVariable() in species:
            continue
        pending[rule.getVariable()] = rule.getMath()

    resolved = dict(consts)
    while pending:
        progressed = False
        for name, rule_math in list(pending.items()):
            try:
                ir = _bake_consts(_compile_ast(rule_math, species), resolved)
            except UnsupportedEventFeatureError:
                continue  # depends on something not resolved yet, or dynamic
            seen: set = set()
            _collect_species(ir, seen)
            if seen or _has_time(ir):
                del pending[name]  # dynamic: never a constant
                progressed = True
                continue
            resolved[name] = float(_eval_ir(ir, 0.0, {}))
            del pending[name]
            progressed = True
        if not progressed:
            break
    return resolved


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
    # Targets that are SBML parameters rather than species. They reach the
    # rate laws through SBMLProcess.with_param_input rather than through the
    # species vector, so expand_events wires them differently.
    _param_targets: tuple = eqx.field(static=True, default=())
    # (target, value) — the value the target's store path starts at. A
    # parameter target must start at its published value, not at zero, or the
    # model runs off a different constant until the event first fires.
    _target_defaults: tuple = eqx.field(static=True, default=())

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
        # A parameter target needs a read port too: the handler forms the
        # assignment as a delta (rhs - current), so it must see the current
        # value, and a parameter is not in _read_species.
        ports.update(
            {
                tgt: Port(
                    role=PortRole.INPUT,
                    default=0.0,
                    units="dimensionless",
                    description=f"reads parameter {tgt}",
                )
                for tgt in self._param_targets
            }
        )
        defaults = dict(self._target_defaults)
        for tgt, _ in self._assign_ir:
            ports[f"__set_{tgt}"] = Port(
                role=PortRole.LATCHED,
                default=float(defaults.get(tgt, 0.0)),
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
    parameter names to their (constant) values, baked into event math and
    extended by :func:`fold_constant_rules` so a rule-defined ModelValue
    resolves. An assignment whose target is a parameter rather than a species
    is kept and recorded in ``_param_targets``; :func:`expand_events` promotes
    it on the owning process.
    """
    import libsbml

    doc = libsbml.SBMLReader().readSBMLFromFile(str(xml_path))
    model = doc.getModel()
    if model is None:
        return []
    species = set(species_names)
    consts = fold_constant_rules(model, species, consts)
    out: list[SBMLEvent] = []
    for i in range(model.getNumEvents()):
        ev = model.getEvent(i)
        eid = ev.getId() or f"event{i}"
        # COPASI writes <delay>0</delay> on every event it exports, so a
        # delay element is not itself a delay. Only a nonzero one is.
        if _delay_seconds(ev) != 0.0:
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
        assigns, param_targets, defaults = [], [], []
        for j in range(ev.getNumEventAssignments()):
            ea = ev.getEventAssignment(j)
            var = ea.getVariable()
            if var not in species and var not in consts:
                log.warning(
                    "SBML event %r on %s assigns to %r, which is neither a "
                    "species nor a resolvable parameter — skipped.",
                    eid,
                    model_name,
                    var,
                )
                continue
            rhs = _bake_consts(_compile_ast(ea.getMath(), species), consts)
            _collect_species(rhs, read)
            if var in species:
                read.add(var)  # current value, for the assignment delta
            else:
                # A parameter target reaches the rate laws through a promoted
                # INPUT port; its store path starts at the published value.
                param_targets.append(var)
                defaults.append((var, float(consts[var])))
            assigns.append((var, rhs))
        if not assigns:
            log.warning(
                "SBML event %r on %s has no usable assignments — skipped.",
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
                _param_targets=tuple(param_targets),
                _target_defaults=tuple(defaults),
            )
        )
        log.info(
            "Translated SBML event %r on %s (targets: %s).",
            eid,
            model_name,
            [t for t, _ in assigns],
        )
    return out


PARAM_PORT_PREFIX = "__par_"


def expand_events(proc) -> tuple[dict, dict]:
    """``(processes, topology)`` composing an SBMLProcess with its events.

    ``processes`` holds the owning process under its own name — promoted, if
    any event assigns to a parameter, via
    :meth:`~hallsim.sbml_import.SBMLProcess.with_param_input` — plus one
    EVENT process per event. ``topology`` holds each event's full wiring and,
    for the owner, *only* the promoted-parameter rows, which the caller merges
    into its own row for that process::

        procs, topo = expand_events(proc)
        topology = {**mine, "dp14": {**mine["dp14"], **topo.get("dp14", {})}}

    Both empty when the model has no events.
    """
    events = getattr(proc, "_events", ())
    if not events:
        return {}, {}

    procs: dict = {}
    topo: dict = {}
    owner_wiring: dict = {}
    for ev in events:
        procs[ev._name] = ev
        wiring = {s: f"{proc._name}/{s}" for s in ev._read_species}
        wiring.update({t: f"{proc._name}/{t}" for t in ev._param_targets})
        for tgt, _ in ev._assign_ir:
            wiring[f"__set_{tgt}"] = f"{proc._name}/{tgt}"
        topo[ev._name] = wiring
        for tgt in ev._param_targets:
            port = f"{PARAM_PORT_PREFIX}{tgt}"
            proc = proc.with_param_input(tgt, port)
            owner_wiring[port] = f"{proc._name}/{tgt}"

    procs[proc._name] = proc
    if owner_wiring:
        topo[proc._name] = owner_wiring
    return procs, topo
