"""Translate SBML ``<event>`` elements into HallSim EVENT processes.

The compiled ODE core (:mod:`hallsim.sbml_core`) ignores SBML events;
:func:`process_from_sbml` has this module re-express each event as an
:class:`SBMLEvent` (``ProcessKind.EVENT``): the trigger becomes
``condition``, the event assignments become a ``handler`` that mutates the
target species' store paths. Event math goes through
:mod:`hallsim.sbml_math` into sympy with the model's constants folded in,
and is printed for JAX once per expression, so it stays shape-polymorphic
under batched runs and an export writes it back out unchanged.

Supported: triggers over time and species; assignments to species and to
parameters. A parameter target is promoted on the owning process via
:meth:`~hallsim.imported.ImportedODEProcess.with_param_input` by
:func:`expand_events`, so the assignment reaches the rate laws through a
store path rather than being dropped — SBML models routinely deliver a dose
that way. Symbols a parameter's own assignment rule defines are folded to
constants first (:func:`fold_constant_rules`), which is how COPASI writes a
ModelValue. A nonzero delay and any priority raise; ``<delay>0</delay>``,
which COPASI emits on every event, is not a delay.

Event math is written in the model's native time. An event carries its
owner's ``time_scale``, so a reconciled model's timed events fire on the
composite clock where the model expects them.
"""

from __future__ import annotations

import functools
import logging

import equinox as eqx
import jax.numpy as jnp
import libsbml
import sympy
from sympy.core.relational import Relational

from hallsim.process import Port, PortRole, Process, ProcessKind
from hallsim.sbml_math import (
    TIME,
    UnsupportedMathError,
    function_definitions,
    inline_functions,
    to_jax,
    to_sympy,
)

log = logging.getLogger(__name__)


class UnsupportedEventFeatureError(Exception):
    """An SBML event uses a construct the translator does not handle."""


def _delay_seconds(event) -> float:
    """The event's delay as a number, or NaN when it is not a constant.

    Returns 0.0 for no ``<delay>`` element and for one whose math evaluates to
    a literal zero — the form COPASI emits unconditionally. A state- or
    time-dependent delay is not constant and comes back NaN, which compares
    unequal to zero and so is rejected by the caller.
    """
    delay = event.getDelay()
    if delay is None or not delay.isSetMath():
        return 0.0
    math = delay.getMath()
    if math.isNumber():
        return float(libsbml.formulaToL3String(math))
    return float("nan")


def _bake(expr, consts: dict, species: set, where: str):
    """``expr`` with every constant folded in. What remains must be a
    species or time, or the event refers to something the model does not
    carry."""
    subs = {}
    for sym in expr.atoms(sympy.Symbol):
        if sym is TIME or sym.name in species:
            continue
        if sym.name not in consts:
            raise UnsupportedEventFeatureError(
                f"{where} references unknown symbol {sym.name!r}"
            )
        subs[sym] = sympy.Float(float(consts[sym.name]))
    return expr.xreplace(subs)


def _symbol_names(expr) -> set[str]:
    return {s.name for s in expr.atoms(sympy.Symbol) if s is not TIME}


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
    defs = function_definitions(model)
    pending: dict[str, sympy.Basic] = {}
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        if not (rule.isAssignment() and rule.isSetVariable()):
            continue
        if not rule.isSetMath() or rule.getVariable() in species:
            continue
        try:
            pending[rule.getVariable()] = inline_functions(
                to_sympy(rule.getMath()), defs
            )
        except UnsupportedMathError:
            continue
    resolved = dict(consts)
    while pending:
        progressed = False
        for name, expr in list(pending.items()):
            names = _symbol_names(expr)
            if TIME in expr.free_symbols or names & set(species):
                del pending[name]  # dynamic: never a constant
                progressed = True
                continue
            if names - set(resolved):
                continue  # waits on another rule
            value = expr.xreplace(
                {
                    sympy.Symbol(n): sympy.Float(float(resolved[n]))
                    for n in names
                }
            )
            resolved[name] = float(value.evalf())
            del pending[name]
            progressed = True
        if not progressed:
            break
    return resolved


# ── trigger pathologies detectable without running the model ────────

_OP = {
    sympy.StrictLessThan: "lt",
    sympy.LessThan: "leq",
    sympy.StrictGreaterThan: "gt",
    sympy.GreaterThan: "geq",
    sympy.Equality: "eq",
    sympy.Unequality: "neq",
}
_FLIP = {
    "lt": "gt",
    "gt": "lt",
    "leq": "geq",
    "geq": "leq",
    "eq": "eq",
    "neq": "neq",
}
#: Relational operators that partition a value at a boundary. Each maps to the
#: operator whose region is its exact complement *including* the boundary
#: point, which is the pair that overlaps at one value.
_OVERLAPPING = {("leq", "gt"), ("gt", "leq"), ("geq", "lt"), ("lt", "geq")}


def _relations(expr) -> list[tuple]:
    """Every comparison in a trigger as ``(op, lhs, rhs)``, operands in a
    canonical order so two spellings of one comparison compare equal."""
    out = []
    for rel in sympy.sympify(expr).atoms(Relational):
        op = _OP.get(type(rel))
        if op is None:
            continue
        lhs, rhs = rel.lhs, rel.rhs
        if sympy.default_sort_key(lhs) > sympy.default_sort_key(rhs):
            lhs, rhs, op = rhs, lhs, _FLIP[op]
        out.append((op, lhs, rhs))
    return out


def _references_time(expr) -> bool:
    return TIME in sympy.sympify(expr).free_symbols


def trigger_pathologies(events) -> list[str]:
    """Trigger defects that make a model's output round-off dependent.

    Both are properties of the trigger expressions alone, so they are decided
    without integrating anything, and both were found the expensive way — by a
    referee running tolerance sweeps — on Stucki 2005 (BIOMD0000001059).

    **Chattering pairs.** Two triggers testing the same quantity against the
    same constant with complementary operators that *share* the boundary
    (``x <= c`` and ``x > c``) are both satisfiable at ``x == c`` — which is
    exactly where an event root-finder lands. Which one fires is then decided
    by round-off in the last Newton step. On Stucki that returned
    ``cascade(7000)`` as 2.0e+01 on one invocation and 7.28e+20 on another,
    same solver and same tolerances. A latch needs a hysteresis band (arm at
    20, disarm at 18) so no value satisfies both.

    **Equality against time.** ``time == c`` is true on a set of measure zero,
    so whether it ever fires depends on whether a sync point lands exactly on
    ``c``. That makes the scheduler's ``macro_dt`` decide whether the model
    receives its input at all, not merely when. A threshold crossing
    (``time >= c``) is the form that survives discretisation.
    """
    found: list[str] = []
    triggers = [
        (getattr(e, "_name", f"event{i}"), _relations(e._trigger))
        for i, e in enumerate(events)
    ]

    for name, relations in triggers:
        for op, lhs, rhs in relations:
            if op == "eq" and (_references_time(lhs) or _references_time(rhs)):
                found.append(
                    f"event {name!r} triggers on an equality against time; "
                    f"whether it fires depends on the integrator landing "
                    f"exactly on that instant — use a threshold crossing"
                )

    for i, (name_a, rel_a) in enumerate(triggers):
        for name_b, rel_b in triggers[i + 1 :]:
            for op_a, lhs_a, rhs_a in rel_a:
                for op_b, lhs_b, rhs_b in rel_b:
                    if (op_a, op_b) not in _OVERLAPPING:
                        continue
                    if lhs_a != lhs_b or rhs_a != rhs_b:
                        continue
                    found.append(
                        f"events {name_a!r} and {name_b!r} have complementary "
                        f"triggers sharing their boundary ({op_a}/{op_b} on the "
                        f"same expression), so both are satisfiable at the "
                        f"crossing and which fires is decided by round-off — "
                        f"a latch needs a hysteresis band"
                    )
    return found


# ── the EVENT process ──────────────────────────────────────────────


@functools.lru_cache(maxsize=None)
def _compiled(expr, names: tuple):
    """The JAX callable for ``expr`` over ``names`` then time; one per
    distinct expression, since a sympy expression hashes by value."""
    return to_jax(expr, [sympy.Symbol(n) for n in names] + [TIME])


class SBMLEvent(Process):
    """One SBML ``<event>`` as a HallSim EVENT process.

    Reads referenced species through INPUT ports (named by species id) and
    writes each assignment target through a LATCHED ``__set_<species>``
    port. The handler applies ``target := rhs`` as an additive delta
    ``rhs − current`` (the scheduler scatter-adds), i.e. a true assignment.
    ``_trigger`` and ``_assignments`` are sympy over species symbols and
    :data:`hallsim.sbml_math.TIME`, in the owner's native time.
    """

    kind: ProcessKind = ProcessKind.EVENT
    _name: str = ""
    _trigger: object = eqx.field(static=True, default=sympy.false)
    _assignments: tuple = eqx.field(static=True, default=())  # ((tgt, expr),)
    _read_species: tuple = eqx.field(static=True, default=())
    # Targets that are SBML parameters rather than species. They reach the
    # rate laws through SBMLProcess.with_param_input rather than through the
    # species vector, so expand_events wires them differently.
    _param_targets: tuple = eqx.field(static=True, default=())
    # (target, value) — the value the target's store path starts at. A
    # parameter target must start at its published value, not at zero, or the
    # model runs off a different constant until the event first fires.
    _target_defaults: tuple = eqx.field(static=True, default=())
    #: The owner's native time per composite time unit.
    time_scale: float = 1.0

    @property
    def _reads(self) -> tuple:
        return tuple(self._read_species) + tuple(
            p for p in self._param_targets if p not in self._read_species
        )

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
        for tgt, _ in self._assignments:
            # Only a parameter target carries a default here, and only a
            # parameter target needs one: a species target shares its store
            # path with the ODE process's own EVOLVED port, which declares
            # the published initial condition. Naming a value too makes this
            # a second writer-tier claim on that path, and the two disagree
            # whenever the species does not start at zero.
            seed = defaults.get(tgt)
            ports[f"__set_{tgt}"] = Port(
                role=PortRole.LATCHED,
                default=None if seed is None else float(seed),
                units="dimensionless",
                description=f"event assignment target {tgt}",
            )
        return ports

    def _args(self, t, state):
        return [state[n] for n in self._reads] + [t * self.time_scale]

    def condition(self, t, state):
        value = _compiled(self._trigger, self._reads)(*self._args(t, state))
        return jnp.asarray(value, dtype=bool)

    def handler(self, t, state):
        args = self._args(t, state)
        return {
            f"__set_{tgt}": _compiled(expr, self._reads)(*args) - state[tgt]
            for tgt, expr in self._assignments
        }

    def metadata(self):
        base = super().metadata()
        base["event_targets"] = [t for t, _ in self._assignments]
        return base


def translate_events(
    xml_path: str, species_names, consts: dict, model_name: str
) -> list[SBMLEvent]:
    """Read the SBML at ``xml_path`` and return one SBMLEvent per event.

    ``species_names`` is the model's ordered species ids; ``consts`` maps
    parameter names to their (constant) values, folded into event math and
    extended by :func:`fold_constant_rules` so a rule-defined ModelValue
    resolves. An assignment whose target is a parameter rather than a species
    is kept and recorded in ``_param_targets``; :func:`expand_events` promotes
    it on the owning process.
    """
    doc = libsbml.SBMLReader().readSBMLFromFile(str(xml_path))
    model = doc.getModel()
    if model is None:
        return []
    species = set(species_names)
    consts = fold_constant_rules(model, species, consts)
    defs = function_definitions(model)

    def expr_of(node, where):
        try:
            return _bake(
                inline_functions(to_sympy(node), defs), consts, species, where
            )
        except UnsupportedMathError as exc:
            raise UnsupportedEventFeatureError(f"{where}: {exc}") from exc

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
        where = f"event {eid!r} on {model_name}"
        trigger = expr_of(ev.getTrigger().getMath(), f"{where} trigger")
        read = _symbol_names(trigger) & species
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
            rhs = expr_of(ea.getMath(), f"{where} assignment to {var!r}")
            read |= _symbol_names(rhs) & species
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
                _trigger=trigger,
                _assignments=tuple(assigns),
                _read_species=tuple(sorted(read)),
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


def expand_events(proc, name: str | None = None) -> tuple[dict, dict]:
    """``(processes, topology)`` composing an SBMLProcess with its events.

    ``name`` is the namespace the store paths are built under; it defaults to
    the process's own ``_name``. A Composite keys a process by whatever the
    caller chose, which need not be that, so the caller passes its key.

    ``processes`` holds the owning process under its own name — promoted, if
    any event assigns to a parameter, via
    :meth:`~hallsim.sbml_import.SBMLProcess.with_param_input` — plus one
    EVENT process per event, each carrying the owner's ``time_scale``.
    ``topology`` holds each event's full wiring and, for the owner, *only*
    the promoted-parameter rows, which the caller merges into its own row for
    that process::

        procs, topo = expand_events(proc)
        topology = {**mine, "dp14": {**mine["dp14"], **topo.get("dp14", {})}}

    Both empty when the model has no events.
    """
    events = getattr(proc, "_events", ())
    if not events:
        return {}, {}
    owner = name or proc._name

    procs: dict = {}
    topo: dict = {}
    owner_wiring: dict = {}
    for ev in events:
        ev = eqx.tree_at(
            lambda e: e.time_scale, ev, jnp.asarray(proc.time_scale)
        )
        procs[ev._name] = ev
        wiring = {s: f"{owner}/{s}" for s in ev._read_species}
        wiring.update({t: f"{owner}/{t}" for t in ev._param_targets})
        for tgt, _ in ev._assignments:
            wiring[f"__set_{tgt}"] = f"{owner}/{tgt}"
        topo[ev._name] = wiring
        for tgt in ev._param_targets:
            port = f"{PARAM_PORT_PREFIX}{tgt}"
            proc = proc.with_param_input(tgt, port)
            owner_wiring[port] = f"{owner}/{tgt}"

    # The events now live as processes; the owner must not carry them into
    # a second expansion when this composite is nested in another.
    procs[owner] = proc.without_events()
    if owner_wiring:
        topo[owner] = owner_wiring
    return procs, topo
