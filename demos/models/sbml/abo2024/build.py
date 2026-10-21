"""Write the Layton-lab sex-specific exercise model as annotated SBML.

Usage::

    python -m demos.models.sbml.abo2024.build      # writes abo2024_{male,female}.xml

The model is Kim, Saidel & Cabrera 2007 as re-implemented by Abo, Casella &
Layton 2024 (parameter tables and provenance in :mod:`.parameters`): seven
pools (arterial blood and six organs) each carrying 22 substrates, 25
enzyme-level fluxes per organ, flow-limited blood-organ exchange, and a
glucagon-insulin controller. Exercise is a work rate ``WR`` held between
``tstart`` and ``tend``, with every drive decaying on a 5-minute exponential
afterwards, exactly as the source does it.

Each organ flux is an SBML reaction whose rate is in mmol/min and whose
species are concentrations in mM, so a species' rate of change is the
reaction rate over its organ's volume — the source's ``(P - U + UR) / V``.
Arterial O2 and CO2 are held constant, as in the source.
"""

from __future__ import annotations

from pathlib import Path

import libsbml

from .parameters import (
    C0,
    EXCHANGED,
    KM,
    PARTITION,
    SUBSTRATES,
    TISSUES,
    UR_OTHER,
    VMAX,
    WEIGHTS,
    _round_sig,
    initial_state,
)

HERE = Path(__file__).resolve().parent

CHEBI = {
    "GLC": "CHEBI:17634",
    "PYR": "CHEBI:15361",
    "LAC": "CHEBI:16651",
    "ALA": "CHEBI:16977",
    "GLR": "CHEBI:17754",
    "FFA": "CHEBI:35366",
    "TG": "CHEBI:17855",
    "O2": "CHEBI:15379",
    "CO2": "CHEBI:16526",
    "G6P": "CHEBI:14314",
    "GLY": "CHEBI:28087",
    "GAP": "CHEBI:29052",
    "GRP": "CHEBI:15978",
    "ACoA": "CHEBI:15351",
    "CoA": "CHEBI:15346",
    "NAD": "CHEBI:15846",
    "NADH": "CHEBI:16908",
    "ATP": "CHEBI:15422",
    "ADP": "CHEBI:16761",
    "Pi": "CHEBI:26020",
    "PCR": "CHEBI:17287",
    "CR": "CHEBI:16919",
}
#: A class rather than one molecule: the model's FFA and TG pools.
CLASSES = {"FFA", "TG"}
UBERON = {
    "blood": "UBERON:0000178",
    "brain": "UBERON:0000955",
    "heart": "UBERON:0000948",
    "muscle": "UBERON:0001134",
    "gi": "UBERON:0001555",
    "liver": "UBERON:0002107",
    "adipose": "UBERON:0001013",
}

#: Net stoichiometry of each flux, read off callflux.m's production (P) and
#: utilisation (U) lists: product positive, reactant negative.
STOICH = {
    1: {"GLC": -1, "ATP": -1, "G6P": 1, "ADP": 1},
    2: {"G6P": -1, "ATP": -1, "GAP": 2, "ADP": 1},
    3: {
        "GAP": -1,
        "NAD": -1,
        "ADP": -2,
        "Pi": -1,
        "PYR": 1,
        "NADH": 1,
        "ATP": 2,
    },
    4: {
        "PYR": -1,
        "NADH": -1,
        "ATP": -3,
        "GAP": 1,
        "NAD": 1,
        "ADP": 3,
        "Pi": 2,
    },
    5: {"GAP": -1, "G6P": 0.5, "Pi": 0.5},
    6: {"G6P": -1, "GLC": 1, "Pi": 1},
    7: {"G6P": -1, "ATP": -1, "GLY": 1, "ADP": 1, "Pi": 2},
    8: {"GLY": -1, "Pi": -1, "G6P": 1},
    9: {"PYR": -1, "NADH": -1, "LAC": 1, "NAD": 1},
    10: {"LAC": -1, "NAD": -1, "PYR": 1, "NADH": 1},
    11: {"GLR": -1, "ATP": -1, "GRP": 1, "ADP": 1},
    12: {"GAP": -1, "NADH": -1, "GRP": 1, "NAD": 1},
    13: {"GRP": -1, "NAD": -1, "GAP": 1, "NADH": 1},
    14: {"PYR": -1, "ALA": 1},
    15: {"ALA": -1, "PYR": 1},
    16: {"PYR": -1, "CoA": -1, "NAD": -1, "ACoA": 1, "CO2": 1, "NADH": 1},
    17: {
        "FFA": -1,
        "CoA": -8,
        "NAD": -14,
        "ATP": -2,
        "ACoA": 8,
        "NADH": 14,
        "ADP": 2,
        "Pi": 2,
    },
    18: {
        "ACoA": -1,
        "NADH": -14 / 8,
        "ATP": -7 / 8,
        "FFA": 1 / 8,
        "CoA": 1,
        "NAD": 14 / 8,
        "ADP": 7 / 8,
        "Pi": 7 / 8,
    },
    19: {"TG": -1, "GLR": 1, "FFA": 3},
    20: {
        "FFA": -1,
        "GRP": -1 / 3,
        "ATP": -2,
        "TG": 1 / 3,
        "ADP": 2,
        "Pi": 7 / 3,
    },
    21: {
        "ACoA": -1,
        "NAD": -4,
        "ADP": -1,
        "Pi": -1,
        "CO2": 2,
        "CoA": 1,
        "NADH": 4,
        "ATP": 1,
    },
    22: {"O2": -1, "NADH": -2, "ADP": -6, "Pi": -6, "NAD": 2, "ATP": 6},
    23: {"PCR": -1, "ADP": -1, "CR": 1, "ATP": 1},
    24: {"CR": -1, "ATP": -1, "PCR": 1, "ADP": 1},
    25: {"ATP": -1, "ADP": 1, "Pi": 1},
}
NAMES = {
    1: "hexokinase",
    2: "glycolysis to GAP",
    3: "GAP to pyruvate",
    4: "pyruvate to GAP (gluconeogenesis)",
    5: "GAP to G6P",
    6: "glucose-6-phosphatase",
    7: "glycogen synthesis",
    8: "glycogenolysis",
    9: "lactate dehydrogenase, forward",
    10: "lactate dehydrogenase, reverse",
    11: "glycerol kinase",
    12: "GAP to glycerol 3-phosphate",
    13: "glycerol 3-phosphate to GAP",
    14: "alanine aminotransferase, forward",
    15: "alanine aminotransferase, reverse",
    16: "pyruvate dehydrogenase",
    17: "fatty acid beta-oxidation",
    18: "lipogenesis",
    19: "lipolysis",
    20: "triglyceride synthesis",
    21: "TCA cycle",
    22: "oxidative phosphorylation",
    23: "creatine kinase, PCr to Cr",
    24: "creatine kinase, Cr to PCr",
    25: "ATP hydrolysis",
}

#: Exercise control on each organ's fluxes: (flux, rate multiplier).
#: ``epi(l, a)`` is ``1 + l x^2/(a + x^2)`` with x = epi - epi0, ``gir(l, a)``
#: the same in the glucagon/insulin ratio; c-indices are 1-based into c0.
CONTROL = {
    "heart": {
        1: "E13",
        2: "R",
        3: "R",
        8: "R",
        16: "R",
        17: "E15",
        19: "E17",
        21: "R",
        22: "R",
    },
    "muscle": {
        1: "E19",
        2: "R",
        3: "R",
        8: "R*E21",
        14: "E23",
        16: "R",
        17: "E25",
        19: "E27",
        21: "R",
        22: "R",
    },
    "gi": {19: "G29+E31"},
    "liver": {4: "G33", 5: "G35", 6: "G37", 8: "G39", 15: "G41"},
    "adipose": {19: "G43+E45"},
}


def _sp(tissue: str, s: int) -> str:
    return f"{tissue}_{SUBSTRATES[s - 1]}"


def _law(tissue: str, j: int, km: list[float]) -> str:
    """The rate of flux ``j`` in ``tissue``, as callflux.m writes it."""
    y = {i: _sp(tissue, i) for i in range(1, 23)}
    K = {i: repr(float(km[i - 1])) for i in range(1, 27)}
    rs1, rs2 = f"rs1_{tissue}", f"rs2_{tissue}"
    ps1, ps2, ps22 = f"ps1_{tissue}", f"ps2_{tissue}", f"ps22_{tissue}"

    def mm(i, k=None):
        return f"{y[i]}/({k or K[i]} + {y[i]})"

    def bi(a, b, ka=None):
        ka = ka or K[a]
        prod = f"({y[a]}*{y[b]}/({ka}*{K[b]}))"
        return f"{prod}/(1 + {y[a]}/{ka} + {y[b]}/{K[b]} + {prod})"

    muscle = tissue == "muscle"
    laws = {
        1: f"{ps1}*({y[1]}/{K[1]})/(1 + {y[1]}/{K[1]})",
        2: f"{ps22}*({y[10]}/{K[10]})/(1 + {y[10]}/{K[10]})",
        3: f"{rs1}*{ps2}*{bi(12, 20)}",
        4: f"{ps1}*{rs2}*{mm(2)}",
        5: mm(12),
        6: mm(10),
        7: f"{ps1}*{mm(10)}",
        8: f"{ps22}*{bi(11, 20)}",
        9: (
            f"({y[2]}/(0.6 + {y[2]}))*({y[17]}/{y[16]})"
            f"/({K[24]}*0.1 + {y[17]}/{y[16]})"
            if muscle
            else f"{rs2}*{mm(2)}"
        ),
        10: (
            f"({y[3]}/(17.0 + {y[3]}))*({y[16]}/{y[17]})"
            f"/({K[23]} + {y[16]}/{y[17]})"
            if muscle
            else f"{rs1}*{mm(3)}"
        ),
        11: f"{ps1}*{mm(5)}",
        12: f"{rs2}*{mm(12)}",
        13: f"{rs1}*{mm(13)}",
        14: mm(2),
        15: mm(4),
        16: f"{rs1}*{bi(2, 15, '0.065' if muscle else None)}",
        17: f"{rs1}*{bi(6, 15)}",
        18: f"{rs2}*{ps1}*{mm(14)}",
        19: mm(7),
        20: f"{ps1}*{bi(13, 6)}",
        21: f"{rs1}*{ps2}*{bi(14, 20)}",
        22: f"{rs2}*{ps2}*{bi(8, 20)}",
        23: f"{ps2}*{mm(21)}",
        24: f"{ps1}*{mm(22)}",
        25: mm(18),
    }
    return laws[j]


def _control(tissue: str, j: int) -> str:
    spec = CONTROL.get(tissue, {}).get(j)
    if spec is None:
        return "1"
    terms = []
    for part in spec.split("*"):
        if part == "R":
            terms.append(f"RMR_{tissue}")
            continue
        boosts = []
        for b in part.split("+"):
            kind, i = b[0], int(b[1:])
            x = "(epi - epi0)" if kind == "E" else "GIR"
            boosts.append(f"c{i}*{x}^2/(c{i + 1} + {x}^2)")
        terms.append(f"(1 + active*({' + '.join(boosts)}))")
    return "*".join(terms)


class _Writer:
    def __init__(self, model_id: str, name: str):
        self.doc = libsbml.SBMLDocument(3, 2)
        self.m = self.doc.createModel()
        self.m.setId(model_id)
        self.m.setName(name)
        self._meta = 0

    def annotate(self, el, curie: str, qualifier=libsbml.BQB_IS):
        self._meta += 1
        el.setMetaId(f"meta_{self._meta}")
        cv = libsbml.CVTerm(libsbml.BIOLOGICAL_QUALIFIER)
        cv.setBiologicalQualifierType(qualifier)
        prefix, ident = curie.split(":")
        cv.addResource(f"http://identifiers.org/{prefix.lower()}/{curie}")
        el.addCVTerm(cv)

    def param(self, pid, value=None, constant=True, units=None, name=None):
        p = self.m.createParameter()
        p.setId(pid)
        p.setConstant(constant)
        if value is not None:
            p.setValue(float(value))
        if units:
            p.setUnits(units)
        if name:
            p.setName(name)
        return p

    def assign(self, pid, formula, units=None, name=None):
        self.param(pid, constant=False, units=units, name=name)
        rule = self.m.createAssignmentRule()
        rule.setVariable(pid)
        rule.setMath(_math(formula))

    def rate_rule(self, pid, formula):
        rule = self.m.createRateRule()
        rule.setVariable(pid)
        rule.setMath(_math(formula))


def _math(formula: str):
    ast = libsbml.parseL3Formula(formula)
    if ast is None:
        raise ValueError(
            f"unparseable: {formula}\n{libsbml.getLastParseL3Error()}"
        )
    return ast


def _names(node) -> set[str]:
    out = {node.getName()} if node.isName() else set()
    for i in range(node.getNumChildren()):
        out |= _names(node.getChild(i))
    return out


def _units(w: _Writer):
    def unit(uid, *parts):
        ud = w.m.createUnitDefinition()
        ud.setId(uid)
        for kind, exp, scale, mult in parts:
            u = ud.createUnit()
            u.setKind(kind)
            u.setExponent(exp)
            u.setScale(scale)
            u.setMultiplier(mult)

    unit("minute", (libsbml.UNIT_KIND_SECOND, 1, 0, 60.0))
    unit("mmol", (libsbml.UNIT_KIND_MOLE, 1, -3, 1.0))
    unit(
        "mM",
        (libsbml.UNIT_KIND_MOLE, 1, -3, 1.0),
        (libsbml.UNIT_KIND_LITRE, -1, 0, 1.0),
    )
    unit(
        "pM",
        (libsbml.UNIT_KIND_MOLE, 1, -12, 1.0),
        (libsbml.UNIT_KIND_LITRE, -1, 0, 1.0),
    )
    unit(
        "mmol_per_min",
        (libsbml.UNIT_KIND_MOLE, 1, -3, 1.0),
        (libsbml.UNIT_KIND_SECOND, -1, 0, 60.0),
    )
    unit(
        "L_per_min",
        (libsbml.UNIT_KIND_LITRE, 1, 0, 1.0),
        (libsbml.UNIT_KIND_SECOND, -1, 0, 60.0),
    )
    w.m.setTimeUnits("minute")
    w.m.setSubstanceUnits("mmol")
    w.m.setVolumeUnits("litre")
    w.m.setExtentUnits("mmol")


#: Anchors of the relative-intensity exercise layer that the published
#: papers do not print. Each is ours and is named as such where it is used.
RELATIVE = dict(
    # adrenaline plasma half-life (min): infusion studies and labelling put
    # it under 5 min with steady state in 10-15 min; 2.5 is the midpoint
    # this layer uses, to be swept over 1.5-4.
    epi_half_life=2.5,
    # the drive is held to the source's own scale: its steady state at 60%
    # VO2max (150 W for the reference male) equals the source's adrenaline
    # plateau, epi0 + gain, because the source's fitted boosts on muscle
    # glucose uptake and lipolysis assume that scale. Anchored to plasma
    # adrenaline instead (6x rest at maximum, Galbo et al. 1975) lactate
    # barely rises and FFA falls, so the state is an adrenergic drive.
    epi_ref_intensity=60.0,
    # secretion threshold and slope in %VO2max: no open source pins them;
    # 70 and 0.1/% are placeholders to sweep (55-80, 0.05-0.2).
    epi_threshold=70.0,
    epi_slope=0.1,
)


def _relative_intensity(w: "_Writer", sex: int) -> None:
    """Palumbo et al. 2018's exercise layer (PLoS Comput Biol 14:e1006073):
    intensity is a fraction of the subject's VO2max, oxygen uptake follows it
    with a first-order lag, the work rate the original drives read is derived
    from it by the ACSM leg-cycling equation, and the adrenergic drive is a
    state with secretion rising with intensity and first-order clearance, the
    form of Kildegaard et al. 2007. Its glucose-dependent secretion term is
    left out: it acts below about 3.5 mM blood glucose, outside this model's
    range. The drive keeps the source's scale (see ``RELATIVE``); threshold
    and slope are unanchored and are what a fit to data has to settle."""
    body = WEIGHTS[sex]["Wbody"]
    w.param("intensity", 0.0, name="target exercise intensity (% VO2max)")
    w.param("VO2max", 50.2, name="maximal O2 uptake (mL/kg/min)")
    w.param("body_weight", body, name="body weight (kg)")
    w.param("VO2_rest", 3.5, name="resting O2 uptake (mL/kg/min)")
    w.param("k_vo2", 0.8, name="O2 uptake rate constant (1/min)")
    w.param("WR_ref", 150.0, name="work rate the flow gains were set at (W)")
    w.param("PVO2max", 0.0, constant=False, name="O2 uptake (% VO2max)")
    w.rate_rule("PVO2max", "k_vo2*(intensity - PVO2max)")
    w.assign(
        "WR",
        "max(0, body_weight*(PVO2max*VO2max/100 - 2*VO2_rest)/10.8)",
        name="work rate (W), ACSM leg cycling",
    )
    w.param("active", 1.0)
    a = RELATIVE
    w.param("epi_half_life", a["epi_half_life"], units="minute")
    w.param("epi_ref_intensity", a["epi_ref_intensity"])
    w.param("epi_gain", 1100.0 if sex == 0 else 330.0)
    w.param("epi_threshold", a["epi_threshold"])
    w.param("epi_slope", a["epi_slope"])
    w.assign("k_epi", "ln(2)/epi_half_life")
    sig = "1/(1 + exp(epi_slope*(epi_threshold - {p})))"
    w.assign(
        "epi_secretion_max",
        "k_epi*epi_gain/("
        + sig.format(p="epi_ref_intensity")
        + " - "
        + sig.format(p="0")
        + ")",
    )
    w.assign(
        "epi_secretion",
        "k_epi*epi0 + epi_secretion_max*("
        + sig.format(p="PVO2max")
        + " - "
        + sig.format(p="0")
        + ")",
    )
    w.param(
        "epi",
        250.0 if sex == 0 else 150.0,
        constant=False,
        name="adrenergic drive (pM, on the source's adrenaline scale)",
    )
    w.rate_rule("epi", "epi_secretion - k_epi*epi")


def build(sex: int, exercise: str = "fixed") -> str:
    """The SBML document for one parameter set. ``exercise="fixed"`` is the
    source as published; ``"relative"`` replaces its fixed work-rate pulse
    with :func:`_relative_intensity`."""
    if exercise not in ("fixed", "relative"):
        raise ValueError(
            f"exercise is 'fixed' or 'relative', not {exercise!r}"
        )
    label = "male" if sex == 0 else "female"
    suffix = "" if exercise == "fixed" else "_relative"
    w = _Writer(
        f"abo2024_{label}{suffix}",
        f"Abo2024 - whole-body fuel metabolism during exercise ({label}"
        f"{', relative intensity' if suffix else ''})",
    )
    _units(w)
    notes = (
        "<body xmlns='http://www.w3.org/1999/xhtml'><p>Kim, Saidel &amp; "
        "Cabrera 2007 (Ann Biomed Eng 35:69, doi:10.1007/s10439-006-9201-x) "
        "as re-implemented by Abo, Casella &amp; Layton 2024 (Bull Math Biol "
        "86:17, doi:10.1007/s11538-023-01242-4), transcribed to SBML from "
        "github.com/Layton-Lab/Sex-Specific-Exercise-Model at commit 1536c28, "
        f"{label} parameter set. Time in minutes, concentrations in mM, "
        "hormones in pM, fluxes in mmol/min. Exercise is a work rate WR (W) "
        "between tstart and tend; afterwards every exercise drive decays as "
        "exp(-(t - tend)/5), as in the source.</p></body>"
    )
    w.m.setNotes(notes)

    y0 = initial_state(sex)
    weights = WEIGHTS[sex]
    density = dict(
        brain=1.04, heart=1.05, muscle=1.05, gi=1.06, liver=1.08, adipose=0.923
    )
    mass = dict(
        brain="WB", heart="WH", muscle="WM", gi="WG", liver="WL", adipose="WA"
    )
    sizes = {"blood": 5.5 if sex == 0 else 5.0}
    sizes.update({t: weights[mass[t]] * density[t] for t in TISSUES})
    for comp, size in sizes.items():
        c = w.m.createCompartment()
        c.setId(comp)
        c.setName(
            {"gi": "gastrointestinal tract", "muscle": "skeletal muscle"}.get(
                comp, comp
            )
        )
        c.setSize(size)
        c.setConstant(True)
        c.setSpatialDimensions(3)
        c.setUnits("litre")
        w.annotate(c, UBERON[comp])

    def species(comp, s, value, fixed=False):
        sid = f"{comp}_{SUBSTRATES[s - 1]}"
        sp = w.m.createSpecies()
        sp.setId(sid)
        sp.setName(SUBSTRATES[s - 1])
        sp.setCompartment(comp)
        sp.setInitialConcentration(float(value))
        sp.setSubstanceUnits("mmol")
        sp.setHasOnlySubstanceUnits(False)
        sp.setBoundaryCondition(fixed)
        sp.setConstant(fixed)
        w.annotate(
            sp,
            CHEBI[SUBSTRATES[s - 1]],
            (
                libsbml.BQB_IS_VERSION_OF
                if SUBSTRATES[s - 1] in CLASSES
                else libsbml.BQB_IS
            ),
        )

    for s in range(1, 10):
        species("blood", s, y0[s - 1], fixed=s in (8, 9))
    for i, t in enumerate(TISSUES):
        for s in range(1, 23):
            species(t, s, y0[22 * (i + 1) + s - 1])

    c0 = _round_sig(C0[sex])
    for i, v in enumerate(c0, start=1):
        w.param(f"c{i}", v)
    w.param(
        "epi0", 250.0 if sex == 0 else 150.0, name="basal epinephrine (pM)"
    )
    w.param("glucagon0", y0[154])
    w.param("insulin0", y0[155])
    w.param("glucagon", y0[154], constant=False, name="arterial glucagon (pM)")
    w.param("insulin", y0[155], constant=False, name="arterial insulin (pM)")
    if exercise == "fixed":
        w.param("WR", 150.0, name="work rate (W)")
        w.param("tstart", 10.0, units="minute")
        w.param("tend", 80.0, units="minute")
        w.param("epi_gain", 1100.0 if sex == 0 else 330.0)
        w.assign("active", "piecewise(1, (time > tstart) && (WR > 0), 0)")
        w.assign(
            "recovery", "piecewise(1, time < tend, exp(-(time - tend)/5))"
        )
        w.assign(
            "epi",
            "epi0 + active*epi_gain*(1 - exp(-(time - tstart)/30))*recovery",
            name="arterial epinephrine (pM)",
        )
        drive = "active*{sig}*(1 - exp(-(time - tstart)/{tau}))*recovery"
        work = "active*WR*{gamma}*recovery"
    else:
        _relative_intensity(w, sex)
        drive = "{sig}*WR/WR_ref"
        work = "WR*{gamma}"
    q_rest = {
        "brain": 0.75,
        "heart": 0.25,
        "muscle": 0.9,
        "gi": 1.1,
        "adipose": 0.36,
    }
    q_ha, q_other = (0.4, 1.74) if sex == 0 else (0.25, 1.39)
    w.param("Q_brain", q_rest["brain"], units="L_per_min")
    w.param("Q_adipose", q_rest["adipose"], units="L_per_min")
    w.param("Q_hepatic_artery", q_ha, units="L_per_min")
    for t, sig, tau in (
        ("heart", "c1", "c4"),
        ("muscle", "c2", "c47"),
        ("gi", "c3", "c4"),
    ):
        w.assign(
            f"Q_{t}",
            f"{q_rest[t]} + " + drive.format(sig=sig, tau=tau),
            units="L_per_min",
        )
    mr = dict(
        brain=15.20,
        heart=7.33,
        muscle=10.82,
        gi=3.04,
        liver=13.92,
        adipose=2.74,
    )
    for t in ("brain", "gi", "liver", "adipose"):
        w.param(f"MR_{t}", mr[t])
    w.assign("MR_heart", f"{mr['heart']} + " + work.format(gamma="c48"))
    w.assign("MR_muscle", f"{mr['muscle']} + " + work.format(gamma="c5"))
    w.assign("RMR_heart", f"MR_heart/{mr['heart']}")
    w.assign("RMR_muscle", f"MR_muscle/{mr['muscle']}")
    w.assign("GIR", "glucagon/insulin - glucagon0/insulin0")
    g = "blood_GLC"
    w.assign(
        "phi",
        f"piecewise(1, {g} < 2.5, 1 - ({g} - 2.5)^2/25, " f"{g} <= 7.5, 0)",
    )
    w.assign(
        "psi",
        f"piecewise(0, {g} < 2.5, 1 - ({g} - 7.5)^2/25, " f"{g} <= 7.5, 1)",
    )
    w.rate_rule(
        "glucagon",
        "glucagon*(phi*(c6 - c7*(glucagon - glucagon0) "
        "- c8*(insulin - insulin0)) - c12)",
    )
    w.rate_rule(
        "insulin",
        "insulin*(psi*(c6 - c9*(glucagon - glucagon0) "
        "- c10*(insulin - insulin0) - c11*(epi - epi0)) - c12)",
    )

    for t in TISSUES:
        km = list(KM[t])
        if t == "muscle" and sex == 1:
            km[6] = 14.8 * 1.28
        km += [
            km[15] / km[16],
            km[16] / km[15],
            km[17] / km[18],
            km[18] / km[17],
        ]
        nad, nadh, atp, adp = (_sp(t, i) for i in (16, 17, 18, 19))
        w.assign(f"rs1_{t}", f"({nad}/{nadh})/({km[22]!r} + {nad}/{nadh})")
        w.assign(f"rs2_{t}", f"({nadh}/{nad})/({km[23]!r} + {nadh}/{nad})")
        w.assign(f"ps1_{t}", f"({atp}/{adp})/({km[24]!r} + {atp}/{adp})")
        w.assign(f"ps2_{t}", f"({adp}/{atp})/({km[25]!r} + {adp}/{atp})")
        w.assign(
            f"ps22_{t}",
            f"({adp}/{atp})^2/({km[25]!r}^2 + " f"({adp}/{atp})^2)",
        )
        vmax = list(VMAX[t])
        if t == "muscle" and sex == 1:
            vmax[8] *= 0.85
        if t == "adipose" and sex == 1:
            vmax[18] *= 1.2
        for j in range(1, 26):
            v = vmax[j - 1]
            if j < 25 and not v:
                continue
            vid = f"Vmax_{t}_f{j}"
            if j == 25:
                w.assign(vid, f"2*MR_{t}", units="mmol_per_min")
            else:
                w.param(vid, v, units="mmol_per_min")
            r = w.m.createReaction()
            r.setId(f"{t}_f{j}")
            r.setName(f"{NAMES[j]} ({t})")
            r.setReversible(False)
            for name, coeff in STOICH[j].items():
                sid = f"{t}_{name}"
                ref = r.createReactant() if coeff < 0 else r.createProduct()
                ref.setSpecies(sid)
                ref.setStoichiometry(abs(coeff))
                ref.setConstant(True)
            law = r.createKineticLaw()
            law.setMath(_math(f"{vid}*{_control(t, j)}*({_law(t, j, km)})"))

    def exchange(t, s, rate):
        sub = SUBSTRATES[s - 1]
        r = w.m.createReaction()
        r.setId(f"uptake_{t}_{sub}")
        r.setName(f"blood to {t} exchange, {sub}")
        r.setReversible(True)
        a = r.createReactant()
        a.setSpecies(f"blood_{sub}")
        a.setStoichiometry(1)
        a.setConstant(True)
        b = r.createProduct()
        b.setSpecies(f"{t}_{sub}")
        b.setStoichiometry(1)
        b.setConstant(True)
        r.createKineticLaw().setMath(_math(rate))

    for t in ("brain", "heart", "muscle", "gi", "adipose"):
        part = PARTITION[t][sex]
        for s in EXCHANGED[t]:
            sub = SUBSTRATES[s - 1]
            exchange(t, s, f"Q_{t}*(blood_{sub} - {part[s - 1]!r}*{t}_{sub})")
    part, part_gi = PARTITION["liver"][sex], PARTITION["gi"][sex]
    for s in EXCHANGED["liver"]:
        sub = SUBSTRATES[s - 1]
        if s in EXCHANGED["gi"]:
            portal = (
                f"(Q_gi*{part_gi[s - 1]!r}*gi_{sub} + Q_hepatic_artery*"
                f"blood_{sub})/(Q_gi + Q_hepatic_artery)"
            )
        else:
            portal = f"blood_{sub}"
        exchange(
            "liver",
            s,
            f"(Q_gi + Q_hepatic_artery)*({portal} - "
            f"{part[s - 1]!r}*liver_{sub})",
        )

    for s in range(1, 8):
        if not UR_OTHER[s - 1]:
            continue
        sub = SUBSTRATES[s - 1]
        r = w.m.createReaction()
        r.setId(f"other_tissues_{sub}")
        r.setName(f"uptake by other tissues, {sub} (negative: release)")
        r.setReversible(True)
        a = r.createReactant()
        a.setSpecies(f"blood_{sub}")
        a.setStoichiometry(1)
        a.setConstant(True)
        r.createKineticLaw().setMath(_math(repr(float(UR_OTHER[s - 1]))))

    species_ids = {
        w.m.getSpecies(i).getId() for i in range(w.m.getNumSpecies())
    }
    for i in range(w.m.getNumReactions()):
        r = w.m.getReaction(i)
        declared = {
            ref.getSpecies()
            for ref in (*r.getListOfReactants(), *r.getListOfProducts())
        }
        read = _names(r.getKineticLaw().getMath()) & species_ids
        for sid in sorted(read - declared):
            r.createModifier().setSpecies(sid)

    errors = w.doc.checkInternalConsistency()
    if errors:
        raise ValueError(w.doc.getErrorLog().toString())
    return libsbml.writeSBMLToString(w.doc)


def main():
    for sex, label in ((0, "male"), (1, "female")):
        for exercise, suffix in (("fixed", ""), ("relative", "_relative")):
            path = HERE / f"abo2024_{label}{suffix}.xml"
            path.write_text(build(sex, exercise))
            print(path)


if __name__ == "__main__":
    main()
