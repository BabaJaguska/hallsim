"""Each edge's declared symbolic form reproduces its derivative and assign."""

import jax.numpy as jnp
import numpy as np
import pytest
import sympy

from hallsim.models.bistable_latch import BistableLatch
from hallsim.models.clamp_edge import ClampEdge
from hallsim.models.gain_edge import GainEdge
from hallsim.models.forcing import PulseSource, StepSource
from hallsim.models.gated_removal import GatedRemoval
from hallsim.models.hill_edge import HillEdge
from hallsim.models.running_integral import RunningIntegral
from hallsim.models.saturating_removal import SaturatingRemoval
from hallsim.process import PortRole
from hallsim.sbml_math import TIME

EDGES = {
    "hill_flux": HillEdge(basal=0.2, hi=1.7, K=(0.6,), n=(3.0,)),
    "hill_two_sources": HillEdge(
        basal=0.1, hi=2.0, K=(0.6, 1.1), n=(3.0, 1.5), sources=("s1", "s2")
    ),
    "hill_level": HillEdge(
        basal=0.3, hi=0.9, K=(0.4,), n=(2.0,), mode="level"
    ),
    "gain_flux": GainEdge(offset=0.2, gain=1.3, mode="flux"),
    "gain_level": GainEdge(offset=-0.2, gain=0.7, mode="level"),
    "clamp": ClampEdge(k_clamp=2.5),
    "gated_removal": GatedRemoval(k_remove=0.8, K=0.5, n=3.0),
    "saturating_removal": SaturatingRemoval(
        alpha=0.1, eta=0.05, beta=1.4, K=0.3, tau_scale=2.0
    ),
    "running_integral": RunningIntegral(power=1.5, tau=None),
    "leaky_integral": RunningIntegral(power=2.0, tau=3.0),
    "pulse_on": PulseSource(amplitude=1.3, t_start=0.2, t_end=1.0),
    "pulse_off": PulseSource(amplitude=1.3, t_start=0.9, t_end=None),
    "step": StepSource(after=2.0, before=0.5, t_step=0.4),
    "latch": BistableLatch(
        k_trigger=1.2,
        K_trig=0.7,
        n_trig=2.0,
        k_feedback=0.9,
        K_fb=0.4,
        n_fb=4.0,
        k_decay=0.25,
        k_output=1.1,
    ),
}


@pytest.mark.parametrize("name", sorted(EDGES), ids=sorted(EDGES))
def test_symbolic_form_matches_numeric(name):
    proc = EDGES[name]
    rng = np.random.default_rng(1)
    t = 0.7
    ports = proc.ports_schema()
    state = {
        p: float(v) for p, v in zip(ports, rng.uniform(0.05, 1.5, len(ports)))
    }
    subs = {sympy.Symbol(p): sympy.Float(v) for p, v in state.items()}
    subs[TIME] = sympy.Float(t)
    jstate = {p: jnp.asarray(v) for p, v in state.items()}

    try:
        derivative = proc.derivative(t, jstate)
    except NotImplementedError:  # a pure source: ASSIGNED ports only
        derivative = {}
    numeric = {p: float(v) for p, v in derivative.items()}
    symbolic: dict[str, float] = {}
    for channel in proc.reaction_channels() or ():
        law = float(sympy.sympify(channel.rate_law).xreplace(subs).evalf())
        for port, coeff in channel.stoichiometry:
            symbolic[port] = symbolic.get(port, 0.0) + coeff * law
    assert set(symbolic) == set(numeric), (symbolic, numeric)
    for port in numeric:
        assert symbolic[port] == pytest.approx(
            numeric[port], rel=1e-9, abs=1e-12
        )

    assigned = {p: float(v) for p, v in proc.assign(t, jstate).items()}
    rules = {
        p: float(sympy.sympify(e).xreplace(subs).evalf())
        for p, e in proc.assignment_rules()
    }
    assert set(rules) == set(assigned)
    for port in assigned:
        assert rules[port] == pytest.approx(
            assigned[port], rel=1e-9, abs=1e-12
        )
    declared = {
        p for p, spec in ports.items() if spec.role is PortRole.ASSIGNED
    }
    assert declared == set(rules)
