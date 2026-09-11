"""SBML semantics as JAX functions: one compiled core per model file.

:func:`compile_sbml` reads a file with libsbml, translates every expression
through :mod:`hallsim.sbml_math`, and returns an :class:`SBMLCore` holding
the model's vectors and the functions over them that
:class:`hallsim.sbml_import.SBMLProcess` integrates.

Three vectors carry the model. ``y`` is integrated: every species no
assignment rule sets and that is not constant (a boundary species with a
rate rule included), then every non-species rate-rule target. ``w`` holds
the assignment-rule targets in dependency order. ``c`` holds the constants:
global parameters no rule sets, boundary species no rule sets, compartment
sizes, and each reaction's local parameters as ``<reaction>_<parameter>``.

Values are **amounts** throughout. An expression reads a species as its
concentration (``amount / compartment``) unless the species has
``hasOnlySubstanceUnits``; an assignment or rate rule on such a species is
multiplied back by its compartment on the way into the store. Kinetic laws
are amount rates, as SBML specifies. Every construct is either translated
or refused by name.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator

import equinox as eqx
import jax.numpy as jnp
import libsbml
import sympy

from hallsim.sbml_math import (
    TIME,
    UnsupportedMathError,
    function_definitions,
    inline_functions,
    to_jax,
    to_sympy,
)

log = logging.getLogger(__name__)


class UnsupportedSBMLFeatureError(Exception):
    """The file uses an SBML construct the importer does not translate.

    The message names the construct, so a deposit is rejected for a
    reason rather than for a traceback."""


_Y, _W, _C = (
    sympy.IndexedBase("y"),
    sympy.IndexedBase("w"),
    sympy.IndexedBase("c"),
)
_T = sympy.Symbol("t")
_ARGS = (_Y, _W, _C, _T)
_MAX_INITIAL_PASSES = 32


def collect_math_nodes(model) -> Iterator:
    """Every libsbml ``ASTNode`` root attached to ``model``: kinetic laws,
    rules, initial assignments, constraints, event math and function
    definitions — everywhere SBML carries an evaluatable expression."""
    for i in range(model.getNumReactions()):
        kl = model.getReaction(i).getKineticLaw()
        if kl is not None and kl.isSetMath():
            yield kl.getMath()
    for i in range(model.getNumRules()):
        r = model.getRule(i)
        if r.isSetMath():
            yield r.getMath()
    for i in range(model.getNumInitialAssignments()):
        ia = model.getInitialAssignment(i)
        if ia.isSetMath():
            yield ia.getMath()
    for i in range(model.getNumConstraints()):
        c = model.getConstraint(i)
        if c.isSetMath():
            yield c.getMath()
    for i in range(model.getNumEvents()):
        e = model.getEvent(i)
        if e.isSetTrigger() and e.getTrigger().isSetMath():
            yield e.getTrigger().getMath()
        if e.isSetDelay() and e.getDelay().isSetMath():
            yield e.getDelay().getMath()
        for j in range(e.getNumEventAssignments()):
            ea = e.getEventAssignment(j)
            if ea.isSetMath():
                yield ea.getMath()
    for i in range(model.getNumFunctionDefinitions()):
        fd = model.getFunctionDefinition(i)
        if fd.isSetMath():
            yield fd.getMath()


def _read_model(path: str):
    doc = libsbml.readSBMLFromFile(str(path))
    model = doc.getModel()
    if model is None:
        raise UnsupportedSBMLFeatureError(
            f"libsbml could not parse {path!r} as SBML"
        )
    return doc, model


def unsupported_features(path: str) -> list[str]:
    """Why :func:`compile_sbml` would refuse ``path``; empty when it would
    not. Each entry names one construct."""
    try:
        doc, model = _read_model(path)
    except UnsupportedSBMLFeatureError as exc:
        return [str(exc)]
    issues = _structural_issues(doc, model)
    seen: set[str] = set()
    for node in collect_math_nodes(model):
        try:
            to_sympy(node)
        except UnsupportedMathError as exc:
            if str(exc) not in seen:
                seen.add(str(exc))
                issues.append(str(exc))
    return issues


def _structural_issues(doc, model) -> list[str]:
    issues = []
    if (
        doc.getPlugin("qual") is not None
        or model.getPlugin("qual") is not None
    ):
        issues.append(
            "this is an SBML qual (logical/Boolean) model, not a kinetic "
            "one: it declares update rules over discrete levels rather than "
            "rate laws, so there is no ODE to integrate"
        )
    for i in range(model.getNumRules()):
        r = model.getRule(i)
        if r.isAlgebraic():
            issues.append("algebraic rules are not supported")
            break
    for i in range(model.getNumCompartments()):
        comp = model.getCompartment(i)
        if comp.isSetConstant() and not comp.getConstant():
            issues.append(
                f"compartment {comp.getId()!r} is not constant; a varying "
                "volume is not supported"
            )
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        for refs in (rxn.getListOfReactants(), rxn.getListOfProducts()):
            for ref in refs:
                if ref.isSetStoichiometryMath():
                    issues.append(
                        f"reaction {rxn.getId()!r} has a symbolic "
                        "stoichiometry (stoichiometryMath)"
                    )
    return issues


class SBMLCore(eqx.Module):
    """The compiled model: vector layouts, initial values and the functions
    :class:`~hallsim.sbml_import.SBMLProcess` integrates. Every field is
    static; fitted values enter through the ``c`` argument."""

    # Index maps are stored as pairs: every field must hash, and a module
    # of static fields is hashed whenever one of its methods is jitted.
    _y_names: tuple = eqx.field(static=True)
    _w_names: tuple = eqx.field(static=True)
    _c_names: tuple = eqx.field(static=True)
    y0: tuple = eqx.field(static=True)
    w0: tuple = eqx.field(static=True)
    c0: tuple = eqx.field(static=True)
    reaction_ids: tuple = eqx.field(static=True)
    #: Net stoichiometry over ``y``: one row per ``y`` entry, one column per
    #: reaction.
    stoichiometry: tuple = eqx.field(static=True)
    #: Each kinetic law as a sympy expression in the model's own symbols.
    rate_laws: tuple = eqx.field(static=True)
    #: Per reaction, the ``y`` names its law reads, through assignments —
    #: the sparsity pattern of ``∂v/∂y``.
    reads: tuple = eqx.field(static=True)
    #: ``(target, expr)`` per assignment rule, in dependency order, and per
    #: rate rule, in the model's own symbols — what an export re-emits.
    assignment_rules: tuple = eqx.field(static=True)
    rate_rules: tuple = eqx.field(static=True)
    #: ``(id, compartment, has_only_substance_units, boundary, constant)``
    #: per species, ``(id, size)`` per compartment, and per reaction the
    #: ``(local id, c name)`` pairs of its local parameters.
    species_info: tuple = eqx.field(static=True)
    compartment_sizes: tuple = eqx.field(static=True)
    local_parameters: tuple = eqx.field(static=True)
    _species_compartment: tuple = eqx.field(static=True)
    _velocities: object = eqx.field(static=True)
    _rate_rules: object = eqx.field(static=True)
    _assignments: tuple = eqx.field(static=True)

    @property
    def y_indexes(self) -> dict:
        return {n: i for i, n in enumerate(self._y_names)}

    @property
    def w_indexes(self) -> dict:
        return {n: i for i, n in enumerate(self._w_names)}

    @property
    def c_indexes(self) -> dict:
        return {n: i for i, n in enumerate(self._c_names)}

    @property
    def species_compartment(self) -> dict:
        return dict(self._species_compartment)

    def reaction_velocities(self, y, w, c, t):
        """``v(y, w, c, t)``, one amount rate per reaction."""
        if not self.reaction_ids:
            return jnp.zeros((0,), dtype=jnp.asarray(y).dtype)
        return jnp.stack(self._velocities(y, w, c, t))

    def ratefunc(self, y, t, w, c):
        """``dy/dt = N·v + rate rules``."""
        if self.reaction_ids:
            dy = jnp.asarray(self.stoichiometry) @ self.reaction_velocities(
                y, w, c, t
            )
        else:
            dy = jnp.zeros((len(self._y_names),), dtype=jnp.asarray(y).dtype)
        if self._rate_rules is not None:
            dy = dy + jnp.stack(self._rate_rules(y, w, c, t))
        return dy

    def assignmentfunc(self, y, w, c, t):
        """``w`` with every assignment rule recomputed, in dependency order."""
        for j, f in self._assignments:
            w = w.at[j].set(f(y, w, c, t))
        return w


_CORES: dict = {}


def compile_sbml(path: str) -> SBMLCore:
    """The :class:`SBMLCore` for the SBML file at ``path``.

    Cached per file (path, size, mtime): a core holds compiled functions,
    and two cores for one file would be two pytree types sharing no
    compiled solve.
    """
    try:
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is not None and key in _CORES:
        return _CORES[key]
    core = _compile(path)
    if key is not None:
        _CORES[key] = core
    return core


def _compile(path: str) -> SBMLCore:
    doc, model = _read_model(path)
    issues = _structural_issues(doc, model)
    if issues:
        raise UnsupportedSBMLFeatureError("; ".join(issues))
    defs = function_definitions(model)

    def expr_of(node):
        return inline_functions(to_sympy(node), defs)

    assigned, rate_ruled = {}, {}
    for i in range(model.getNumRules()):
        r = model.getRule(i)
        if r.isAssignment():
            assigned[r.getVariable()] = expr_of(r.getMath())
        elif r.isRate():
            rate_ruled[r.getVariable()] = expr_of(r.getMath())

    compartments = {
        model.getCompartment(i).getId(): model.getCompartment(i)
        for i in range(model.getNumCompartments())
    }
    species = [model.getSpecies(i) for i in range(model.getNumSpecies())]
    params = [model.getParameter(i) for i in range(model.getNumParameters())]
    reactions = [model.getReaction(i) for i in range(model.getNumReactions())]
    for cid in compartments:
        if cid in assigned or cid in rate_ruled:
            raise UnsupportedSBMLFeatureError(
                f"compartment {cid!r} is set by a rule; a varying volume "
                "is not supported"
            )

    # ── layout ────────────────────────────────────────────────────
    y_names = [
        s.getId()
        for s in species
        if not s.getConstant()
        and s.getId() not in assigned
        and (not s.getBoundaryCondition() or s.getId() in rate_ruled)
    ]
    y_names += [p.getId() for p in params if p.getId() in rate_ruled]
    w_names = _dependency_order(assigned)
    c_names = [
        p.getId()
        for p in params
        if p.getId() not in assigned and p.getId() not in rate_ruled
    ]
    c_names += [
        s.getId()
        for s in species
        if s.getId() not in y_names and s.getId() not in assigned
    ]
    c_names += list(compartments)
    local_names: dict[str, dict[str, str]] = {}
    for rxn in reactions:
        kl = rxn.getKineticLaw()
        if kl is None:
            continue
        locals_ = {}
        for j in range(kl.getNumParameters()):
            lp = kl.getParameter(j)
            locals_[lp.getId()] = f"{rxn.getId()}_{lp.getId()}"
        local_names[rxn.getId()] = locals_
        c_names += list(locals_.values())
    y_indexes = {n: i for i, n in enumerate(y_names)}
    w_indexes = {n: i for i, n in enumerate(w_names)}
    c_indexes = {n: i for i, n in enumerate(c_names)}

    species_compartment = {s.getId(): s.getCompartment() for s in species}
    substance_only = {
        s.getId() for s in species if s.getHasOnlySubstanceUnits()
    }
    is_species = set(species_compartment)

    def volume(sid):
        """Symbolic compartment size a species is divided by; 1 when the
        compartment has no volume to speak of."""
        comp = compartments.get(species_compartment.get(sid))
        if comp is None or sid in substance_only:
            return None
        if comp.isSetSpatialDimensions() and comp.getSpatialDimensions() == 0:
            return None
        return _C[c_indexes[comp.getId()]]

    def location(name):
        if name in y_indexes:
            return _Y[y_indexes[name]]
        if name in w_indexes:
            return _W[w_indexes[name]]
        if name in c_indexes:
            return _C[c_indexes[name]]
        return None

    def as_read(name):
        """What an expression sees when it names ``name``."""
        loc = location(name)
        if loc is None:
            return None
        vol = volume(name) if name in is_species else None
        return loc / vol if vol is not None else loc

    def bind(expr, reaction_id=None):
        """Map the model's symbols onto ``y``, ``w``, ``c`` and ``t``."""
        subs = {TIME: _T}
        for sym in expr.atoms(sympy.Symbol):
            if sym is TIME:
                continue
            name = sym.name
            if reaction_id is not None and name in local_names.get(
                reaction_id, {}
            ):
                subs[sym] = _C[c_indexes[local_names[reaction_id][name]]]
                continue
            target = as_read(name)
            if target is None:
                where = f" in reaction {reaction_id!r}" if reaction_id else ""
                raise UnsupportedSBMLFeatureError(
                    f"expression{where} refers to {name!r}, which is not a "
                    "species, parameter or compartment of the model"
                )
            subs[sym] = target
        return expr.xreplace(subs)

    def stored(name, expr):
        """A rule's value as it is stored: a concentration species goes in
        as an amount."""
        vol = volume(name) if name in is_species else None
        return expr * vol if vol is not None else expr

    # ── initial values ────────────────────────────────────────────
    values: dict[str, float] = {}
    for cid, comp in compartments.items():
        values[cid] = float(comp.getSize()) if comp.isSetSize() else 1.0
    for p in params:
        if p.isSetValue():
            values[p.getId()] = float(p.getValue())
    for s in species:
        sid = s.getId()
        if s.isSetInitialAmount():
            values[sid] = float(s.getInitialAmount())
        elif s.isSetInitialConcentration():
            size = values.get(species_compartment[sid], 1.0)
            values[sid] = float(s.getInitialConcentration()) * (
                1.0 if sid in substance_only else size
            )
    initial_assignments = {
        model.getInitialAssignment(i).getSymbol(): expr_of(
            model.getInitialAssignment(i).getMath()
        )
        for i in range(model.getNumInitialAssignments())
    }
    _evaluate_initials(
        values,
        initial_assignments,
        assigned,
        w_names,
        is_species,
        substance_only,
        species_compartment,
        compartments,
    )

    def initial(name):
        if name not in values:
            raise UnsupportedSBMLFeatureError(
                f"{name!r} has no initial value: no attribute, no "
                "initial assignment and no assignment rule sets it"
            )
        return values[name]

    y0 = tuple(initial(n) for n in y_names)
    w0 = tuple(initial(n) for n in w_names)
    c0 = tuple(
        initial(n)
        for n in c_names[
            : len(c_names) - sum(len(v) for v in local_names.values())
        ]
    )
    for rxn in reactions:
        kl = rxn.getKineticLaw()
        if kl is None:
            continue
        c0 += tuple(
            float(kl.getParameter(j).getValue())
            for j in range(kl.getNumParameters())
        )

    # ── functions ─────────────────────────────────────────────────
    rate_laws, bound_laws, matrix = [], [], []
    for rxn in reactions:
        kl = rxn.getKineticLaw()
        if kl is None or not kl.isSetMath():
            raise UnsupportedSBMLFeatureError(
                f"reaction {rxn.getId()!r} has no kinetic law"
            )
        law = expr_of(kl.getMath())
        rate_laws.append(law)
        bound_laws.append(bind(law, rxn.getId()))
        column = [0.0] * len(y_names)
        for refs, sign in (
            (rxn.getListOfReactants(), -1.0),
            (rxn.getListOfProducts(), 1.0),
        ):
            for ref in refs:
                row = y_indexes.get(ref.getSpecies())
                if row is not None:
                    column[row] += sign * float(ref.getStoichiometry())
        matrix.append(column)
    stoichiometry = tuple(
        tuple(matrix[j][i] for j in range(len(reactions)))
        for i in range(len(y_names))
    )
    velocities = to_jax(sympy.Tuple(*bound_laws), _ARGS) if reactions else None

    rate_rule_exprs = [sympy.Integer(0)] * len(y_names)
    for name, expr in rate_ruled.items():
        if name not in y_indexes:
            raise UnsupportedSBMLFeatureError(
                f"rate rule on {name!r}, which is not an integrated quantity"
            )
        rate_rule_exprs[y_indexes[name]] = stored(name, bind(expr))
    rate_rules = (
        to_jax(sympy.Tuple(*rate_rule_exprs), _ARGS) if rate_ruled else None
    )
    assignments = tuple(
        (w_indexes[name], to_jax(stored(name, bind(assigned[name])), _ARGS))
        for name in w_names
    )

    reads = tuple(
        frozenset(_reads(law, assigned, y_indexes)) for law in rate_laws
    )
    return SBMLCore(
        _y_names=tuple(y_names),
        _w_names=tuple(w_names),
        _c_names=tuple(c_names),
        y0=y0,
        w0=w0,
        c0=c0,
        reaction_ids=tuple(r.getId() for r in reactions),
        stoichiometry=stoichiometry,
        rate_laws=tuple(rate_laws),
        reads=reads,
        assignment_rules=tuple((n, assigned[n]) for n in w_names),
        rate_rules=tuple(rate_ruled.items()),
        species_info=tuple(
            (
                s.getId(),
                s.getCompartment(),
                bool(s.getHasOnlySubstanceUnits()),
                bool(s.getBoundaryCondition()),
                bool(s.getConstant()),
            )
            for s in species
        ),
        compartment_sizes=tuple((cid, values[cid]) for cid in compartments),
        local_parameters=tuple(
            (rid, tuple(names.items())) for rid, names in local_names.items()
        ),
        _species_compartment=tuple(species_compartment.items()),
        _velocities=velocities,
        _rate_rules=rate_rules,
        _assignments=assignments,
    )


def _dependency_order(assigned: dict) -> list[str]:
    """Assignment targets ordered so each rule reads only rules before it."""
    order, placed = [], set()
    pending = dict(assigned)
    while pending:
        ready = [
            name
            for name, expr in pending.items()
            if not (
                {s.name for s in expr.free_symbols} & set(pending) - {name}
            )
        ]
        if not ready:
            raise UnsupportedSBMLFeatureError(
                f"assignment rules form a cycle: {sorted(pending)}"
            )
        for name in ready:
            order.append(name)
            placed.add(name)
            del pending[name]
    return order


def _reads(expr, assigned, y_indexes) -> set[str]:
    """``y`` names ``expr`` depends on, following assignment rules."""
    out, stack, seen = set(), [expr], set()
    while stack:
        e = stack.pop()
        for sym in e.free_symbols:
            if sym.name in y_indexes:
                out.add(sym.name)
            elif sym.name in assigned and sym.name not in seen:
                seen.add(sym.name)
                stack.append(assigned[sym.name])
    return out


def _evaluate_initials(
    values,
    initial_assignments,
    assigned,
    w_names,
    is_species,
    substance_only,
    species_compartment,
    compartments,
):
    """Fill ``values`` with initial assignments and assignment rules at
    ``t = 0``, iterating until every value is settled."""

    def environment():
        env = {TIME: sympy.Float(0.0)}
        for name, v in values.items():
            conc = v
            if name in is_species and name not in substance_only:
                size = values.get(species_compartment[name], 1.0)
                conc = v / size
            env[sympy.Symbol(name)] = sympy.Float(conc)
        return env

    def store(name, value):
        if name in is_species and name not in substance_only:
            value *= values.get(species_compartment[name], 1.0)
        values[name] = float(value)

    targets = list(initial_assignments) + [
        n for n in w_names if n not in initial_assignments
    ]
    for _ in range(_MAX_INITIAL_PASSES):
        before = dict(values)
        env = environment()
        for name in targets:
            expr = initial_assignments.get(name, assigned.get(name))
            value = expr.xreplace(env)
            if value.free_symbols:
                continue
            store(name, float(value.evalf()))
        if values == before and all(n in values for n in targets):
            return
    missing = [n for n in targets if n not in values]
    if missing:
        raise UnsupportedSBMLFeatureError(
            f"initial values of {missing} cannot be resolved"
        )
