"""Write a composite out as one SBML document.

The wiring a composite holds as data — which processes share a species,
how each model's clock was reconciled, what every rate constant is now —
becomes an SBML model that libRoadRunner or COPASI can run, so a cross-engine
check is a test rather than a hand-merge, and a sub-model can go to an
engine with a method HallSim lacks.

The document transcribes HallSim's semantics literally. Every species is an
**amount** in one unit compartment with ``hasOnlySubstanceUnits`` set, and
where an imported model read a concentration the law divides by that model's
compartment size explicitly, which is now a parameter. A process's
constants, compartments and local parameters are prefixed with the process
name; a store path becomes the species id it maps to, so two processes
sharing a path share the species. Time reconciliation is compiled in: a
model on a rescaled clock has ``time`` substituted and its rates multiplied
by the scale.

An SBML-imported process exports from its compiled core. A hand-written
process exports through :meth:`hallsim.process.Process.reaction_channels`
and :meth:`~hallsim.process.Process.assignment_rules`, its symbolic forms;
one that declares neither is refused by name.
"""

from __future__ import annotations

import re

import libsbml
import sympy

from hallsim.process import ProcessKind
from hallsim.sbml_events import SBMLEvent
from hallsim.sbml_math import TIME, to_ast
from hallsim.store import as_paths

COMPARTMENT = "store"


class UnsupportedExportError(ValueError):
    """The composite holds something the exporter cannot write yet."""


def _sid(text: str) -> str:
    sid = re.sub(r"\W", "_", text)
    return sid if re.match(r"[A-Za-z_]", sid) else "_" + sid


class _Ids:
    """Unique SBML ids for store paths and prefixed model-local names."""

    def __init__(self):
        self.taken: set[str] = set()
        self.by_path: dict[str, str] = {}

    def path(self, store_path: str) -> str:
        if store_path not in self.by_path:
            self.by_path[store_path] = self.fresh(_sid(store_path))
        return self.by_path[store_path]

    def fresh(self, want: str) -> str:
        sid, n = want, 1
        while sid in self.taken:
            n += 1
            sid = f"{want}_{n}"
        self.taken.add(sid)
        return sid


def _set_math(element, expr) -> None:
    element.setMath(to_ast(sympy.sympify(expr)))


def _add_parameter(model, sid, value, constant=True):
    p = model.createParameter()
    p.setId(sid)
    p.setValue(float(value))
    p.setConstant(constant)
    return p


def _add_species(model, sid, amount, *, boundary=False):
    s = model.createSpecies()
    s.setId(sid)
    s.setCompartment(COMPARTMENT)
    s.setInitialAmount(float(amount))
    s.setHasOnlySubstanceUnits(True)
    s.setBoundaryCondition(boundary)
    s.setConstant(False)
    return s


def _add_reaction(model, sid, stoichiometry: dict, law):
    r = model.createReaction()
    r.setId(sid)
    r.setReversible(True)
    r.setFast(False) if hasattr(r, "setFast") else None
    for species, coeff in stoichiometry.items():
        if coeff == 0:
            continue
        ref = r.createReactant() if coeff < 0 else r.createProduct()
        ref.setSpecies(species)
        ref.setStoichiometry(abs(float(coeff)))
        ref.setConstant(True)
    kl = r.createKineticLaw()
    _set_math(kl, law)
    return r


def _add_rule(model, sid, expr, *, rate=False):
    rule = model.createRateRule() if rate else model.createAssignmentRule()
    rule.setVariable(sid)
    _set_math(rule, expr)
    return rule


def composite_to_sbml(composite, *, model_id: str = "composite") -> str:
    """The composite as an SBML Level 3 Version 2 document string."""
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.setId(_sid(model_id))
    comp = model.createCompartment()
    comp.setId(COMPARTMENT)
    comp.setSize(1.0)
    comp.setSpatialDimensions(3)
    comp.setConstant(True)
    ids = _Ids()
    ids.taken.add(COMPARTMENT)

    initial = {k: float(v) for k, v in composite.initial_state().items()}
    written: set[str] = set()
    boundary: set[str] = set()
    rules: list = []
    reactions: list = []

    for name, proc in composite.processes.items():
        topo = composite.topology.get(name, {})
        if isinstance(proc, SBMLEvent):
            exporter = _EventExporter
        elif proc.kind is not ProcessKind.CONTINUOUS:
            raise UnsupportedExportError(
                f"{name!r} is a {proc.kind.name} process; only CONTINUOUS "
                "processes and SBML events export yet"
            )
        elif hasattr(proc, "_model"):
            exporter = _ImportedExporter
        else:
            exporter = _DeclaredExporter
        exporter(name, proc, topo, ids, model).run(
            written, boundary, rules, reactions
        )

    for path, sid in ids.by_path.items():
        _add_species(
            model, sid, initial.get(path, 0.0), boundary=sid in boundary
        )
    for sid, expr, rate in rules:
        _add_rule(model, sid, expr, rate=rate)
    for sid, stoich, law in reactions:
        _add_reaction(model, sid, stoich, law)
    return libsbml.writeSBMLToString(doc)


class _DeclaredExporter:
    """A hand-written process, through its declared symbolic forms."""

    def __init__(self, name, proc, topo, ids, model):
        self.name, self.proc, self.topo, self.ids, self.model = (
            name,
            proc,
            topo,
            ids,
            model,
        )
        self.prefix = _sid(name)

    def _paths(self, port):
        entry = self.topo.get(port, port)
        return as_paths(entry)

    def _bind(self, expr):
        subs = {}
        for sym in expr.atoms(sympy.Symbol):
            if sym is TIME:
                continue
            paths = self._paths(sym.name)
            if len(paths) != 1:
                raise UnsupportedExportError(
                    f"{self.name!r} reads port {sym.name!r}, which binds "
                    f"{len(paths)} store paths; a read must bind one"
                )
            subs[sym] = sympy.Symbol(self.ids.path(paths[0]))
        return expr.xreplace(subs)

    def run(self, written, boundary, rules, reactions):
        channels = self.proc.reaction_channels()
        assignments = self.proc.assignment_rules()
        if channels is None and not assignments:
            raise UnsupportedExportError(
                f"{self.name!r} ({type(self.proc).__name__}) declares no "
                "reaction_channels() or assignment_rules(), so it has no "
                "symbolic form to export"
            )
        for channel in channels or ():
            net: dict[str, float] = {}
            for port, coeff in channel.stoichiometry:
                for path in self._paths(port):
                    sid = self.ids.path(path)
                    net[sid] = net.get(sid, 0.0) + float(coeff)
            reactions.append(
                (
                    self.ids.fresh(
                        f"{self.prefix}__{_sid(channel.reaction_id)}"
                    ),
                    net,
                    self._bind(sympy.sympify(channel.rate_law)),
                )
            )
        for port, expr in assignments:
            for path in self._paths(port):
                rules.append(
                    (
                        self.ids.path(path),
                        self._bind(sympy.sympify(expr)),
                        False,
                    )
                )


class _ImportedExporter:
    """An SBML-imported process, from its compiled core."""

    def __init__(self, name, proc, topo, ids, model):
        self.name, self.proc, self.topo, self.ids, self.model = (
            name,
            proc,
            topo,
            ids,
            model,
        )
        self.core = proc._model
        self.prefix = _sid(name)
        self.scale = float(proc.time_scale)
        for d in proc._param_drivers:
            if type(d).__name__ != "ParamInput":
                raise UnsupportedExportError(
                    f"{name!r} drives {d.param_name!r} through a "
                    f"{type(d).__name__}; only a plain ParamInput exports"
                )
        # a driven constant reads its store path; a stepped one becomes an
        # event; a driven boundary input reads its store path in place of
        # its own rule
        self.driven = {d.param_name: d.input_port for d in proc._param_drivers}
        self.steps = tuple(proc._param_steps)
        self.input_driven = dict(proc._input_drivers)

    def _species_path(self, species):
        entry = self.topo.get(species)
        return as_paths(entry)[0] if entry is not None else None

    def _local(self, name):
        return f"{self.prefix}__{_sid(name)}"

    def _port_path(self, port):
        entry = self.topo.get(port)
        paths = as_paths(entry) if entry is not None else ()
        if len(paths) != 1:
            raise UnsupportedExportError(
                f"{self.name!r}: port {port!r} binds {len(paths)} store "
                "paths; a driving port must bind one"
            )
        return paths[0]

    def run(self, written, boundary, rules, reactions):
        core, proc, model, ids = self.core, self.proc, self.model, self.ids
        params = {k: float(v) for k, v in proc.parameters.items()}
        info = {s[0]: s for s in core.species_info}
        sizes = dict(core.compartment_sizes)
        # constants: every c entry, and the compartment sizes among them
        symbol_of: dict[str, sympy.Expr] = {}  # what a law reads
        sid_of: dict[str, str] = {}  # exported id of a stored quantity
        stepped = {
            name: (t_step, before) for name, t_step, before in self.steps
        }
        step_sid: dict[str, str] = {}
        for cname in core.c_indexes:
            if cname in info:
                continue  # a boundary species: exported as a species below
            if cname in self.driven:
                path = self._port_path(self.driven[cname])
                symbol_of[cname] = sympy.Symbol(ids.path(path))
                continue
            value = params.get(
                cname, dict(zip(core.c_indexes, core.c0))[cname]
            )
            sid = ids.fresh(self._local(cname))
            if cname in stepped:
                _add_parameter(model, sid, stepped[cname][1], constant=False)
                step_sid[cname] = sid
            else:
                _add_parameter(model, sid, value)
            symbol_of[cname] = sympy.Symbol(sid)
        for cname, (t_step, _) in stepped.items():
            ev = model.createEvent()
            ev.setId(ids.fresh(self._local(f"{cname}_step")))
            ev.setUseValuesFromTriggerTime(True)
            trigger = ev.createTrigger()
            trigger.setInitialValue(False)
            trigger.setPersistent(True)
            _set_math(trigger, sympy.Ge(TIME, float(t_step)))
            ea = ev.createEventAssignment()
            ea.setVariable(step_sid[cname])
            _set_math(ea, sympy.Float(params[cname]))
        local_of: dict[str, dict[str, str]] = {}
        for rid, pairs in core.local_parameters:
            local_of[rid] = {lp: cname for lp, cname in pairs}

        def volume(sid_model):
            s = info.get(sid_model)
            if s is None or s[2]:
                return None
            size_sym = symbol_of.get(s[1])
            return size_sym if size_sym is not None else sizes.get(s[1])

        # species: y and w entries are store paths; boundary ones are local
        for (
            sid_model,
            comp_id,
            subst,
            is_boundary,
            is_const,
        ) in core.species_info:
            path = self._species_path(sid_model)
            if sid_model in self.input_driven:
                driving = self._port_path(self.input_driven[sid_model])
                symbol_of[sid_model] = sympy.Symbol(ids.path(driving))
                sid_of[sid_model] = ids.path(driving)
                continue
            if path is not None:
                sid = ids.path(path)
                if sid_model in proc._species_names and any(
                    proc._species_names[i] == sid_model
                    for i in proc._frozen_indices
                ):
                    boundary.add(sid)
            else:
                sid = ids.fresh(self._local(sid_model))
                amount = params.get(sid_model)
                if amount is None:
                    amount = dict(zip(core.c_indexes, core.c0)).get(
                        sid_model, 0.0
                    )
                _add_species(model, sid, amount, boundary=True)
            read = sympy.Symbol(sid)
            vol = volume(sid_model)
            symbol_of[sid_model] = read / vol if vol is not None else read
            sid_of[sid_model] = sid
        # non-species y entries (rate-ruled parameters) and w parameters
        for name in list(core.y_indexes) + list(core.w_indexes):
            if name in symbol_of:
                continue
            path = self._species_path(name)
            if path is None:
                raise UnsupportedExportError(
                    f"{self.name!r}: {name!r} is integrated or assigned but "
                    "bound to no store path"
                )
            symbol_of[name] = sympy.Symbol(ids.path(path))
            sid_of[name] = ids.path(path)

        def bind(expr, reaction_id=None):
            subs = {}
            if self.scale != 1.0:
                subs[TIME] = TIME * self.scale
            for sym in expr.atoms(sympy.Symbol):
                if sym is TIME:
                    continue
                if reaction_id and sym.name in local_of.get(reaction_id, {}):
                    subs[sym] = symbol_of[local_of[reaction_id][sym.name]]
                    continue
                if sym.name not in symbol_of:
                    raise UnsupportedExportError(
                        f"{self.name!r}: no exported symbol for {sym.name!r}"
                    )
                subs[sym] = symbol_of[sym.name]
            return expr.xreplace(subs)

        def stored(name, expr):
            vol = volume(name)
            return expr * vol if vol is not None else expr

        for target, expr in core.assignment_rules:
            if target in self.input_driven:
                continue  # the store path drives it now
            rules.append((sid_of[target], stored(target, bind(expr)), False))
        for target, expr in core.rate_rules:
            rules.append(
                (sid_of[target], self.scale * stored(target, bind(expr)), True)
            )
        for channel in proc.reaction_channels():
            net: dict[str, float] = {}
            for species, coeff in channel.stoichiometry:
                sid = sid_of.get(species)
                if sid is None or sid in boundary:
                    continue
                net[sid] = net.get(sid, 0.0) + float(coeff)
            reactions.append(
                (
                    ids.fresh(self._local(channel.reaction_id)),
                    net,
                    self.scale * bind(channel.rate_law, channel.reaction_id),
                )
            )


class _EventExporter:
    """A translated SBML event, back to ``<event>``."""

    def __init__(self, name, proc, topo, ids, model):
        self.name, self.proc, self.topo, self.ids, self.model = (
            name,
            proc,
            topo,
            ids,
            model,
        )
        self.scale = float(proc.time_scale)

    def _paths(self, port):
        entry = self.topo.get(port)
        return as_paths(entry) if entry is not None else ()

    def _bind(self, expr):
        subs = {TIME: TIME * self.scale} if self.scale != 1.0 else {}
        for sym in expr.atoms(sympy.Symbol):
            if sym is TIME:
                continue
            paths = self._paths(sym.name)
            if len(paths) != 1:
                raise UnsupportedExportError(
                    f"{self.name!r} reads {sym.name!r}, which binds "
                    f"{len(paths)} store paths; a read must bind one"
                )
            subs[sym] = sympy.Symbol(self.ids.path(paths[0]))
        return expr.xreplace(subs)

    def run(self, written, boundary, rules, reactions):
        ev = self.model.createEvent()
        ev.setId(self.ids.fresh(_sid(self.name)))
        ev.setUseValuesFromTriggerTime(True)
        trigger = ev.createTrigger()
        trigger.setInitialValue(False)
        trigger.setPersistent(True)
        _set_math(trigger, self._bind(sympy.sympify(self.proc._trigger)))
        for target, expr in self.proc._assignments:
            for path in self._paths(f"__set_{target}"):
                ea = ev.createEventAssignment()
                ea.setVariable(self.ids.path(path))
                _set_math(ea, self._bind(sympy.sympify(expr)))
