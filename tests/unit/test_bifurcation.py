"""Bifurcation analysis validated on the analytic Hopf normal form.

    x' = mu x - omega y +/- x (x^2 + y^2)
    y' = omega x + mu y +/- y (x^2 + y^2)

Hopf at mu=0; the cubic sign fixes criticality: ``-`` supercritical
(l1<0), ``+`` subcritical (l1>0). These have closed-form answers, so they
pin the first Lyapunov coefficient's sign and the Hopf locator.
"""

import jax.numpy as jnp
import pytest
import numpy as np

from hallsim.bifurcation import (
    field_from_composite,
    equilibrium,
    first_lyapunov_coefficient,
    hopf_scan,
    leading_complex_pair_re,
    spectrum,
)

OMEGA = 1.3


def _normal_form(mu, sign):
    def f(s):
        x, y = s[0], s[1]
        r2 = x * x + y * y
        return jnp.array(
            [
                mu * x - OMEGA * y + sign * x * r2,
                OMEGA * x + mu * y + sign * y * r2,
            ]
        )

    return f


def test_equilibrium_at_origin():
    eq = equilibrium(_normal_form(0.3, -1.0), [0.05, -0.05])
    assert eq is not None
    assert np.allclose(eq, [0.0, 0.0], atol=1e-8)


def test_spectrum_matches_analytic_pair():
    # Jacobian at the origin is [[mu, -omega], [omega, mu]] -> mu +/- i*omega.
    ev = spectrum(_normal_form(-0.2, -1.0), [0.0, 0.0])
    assert np.allclose(sorted(ev.real), [-0.2, -0.2])
    assert np.allclose(sorted(np.abs(ev.imag)), [OMEGA, OMEGA])


def test_leading_pair_crosses_zero_with_mu():
    f_of = lambda m: _normal_form(m, -1.0)  # noqa: E731
    assert leading_complex_pair_re(f_of(-0.1), [0.0, 0.0]) < 0
    assert leading_complex_pair_re(f_of(+0.1), [0.0, 0.0]) > 0


def test_supercritical_l1_negative():
    l1 = first_lyapunov_coefficient(_normal_form(0.0, -1.0), [0.0, 0.0])
    assert l1 < 0


def test_subcritical_l1_positive():
    l1 = first_lyapunov_coefficient(_normal_form(0.0, +1.0), [0.0, 0.0])
    assert l1 > 0


def test_hopf_scan_locates_and_classifies():
    params = np.linspace(-0.25, 0.25, 51)
    sup = hopf_scan(
        lambda m: _normal_form(m, -1.0), params, x0_guess=[0.02, 0.0]
    )
    assert len(sup) == 1
    h = sup[0]
    assert abs(h.param) < 1e-3  # Hopf sits at mu=0
    assert abs(h.omega - OMEGA) < 1e-2
    assert h.supercritical is True

    sub = hopf_scan(
        lambda m: _normal_form(m, +1.0), params, x0_guess=[0.02, 0.0]
    )
    assert len(sub) == 1
    assert sub[0].supercritical is False


def test_hopf_scan_no_false_positive_when_no_crossing():
    # mu stays negative -> stable focus throughout, no Hopf.
    params = np.linspace(-0.5, -0.1, 21)
    assert (
        hopf_scan(
            lambda m: _normal_form(m, -1.0), params, x0_guess=[0.02, 0.0]
        )
        == []
    )


# --- conserved moieties (P0.16) -------------------------------------------
# A <-> B with A + B conserved. The Jacobian [[-K1, K2], [K1, -K2]] is
# singular at every state, so the raw Newton has no unique solution however
# good the seed is; the fixed point on the leaf through y0 is analytic.

K1, K2 = 0.7, 0.3
TOTAL = 4.0
A_STAR = TOTAL * K2 / (K1 + K2)


def _reversible(s):
    return jnp.array([-K1 * s[0] + K2 * s[1], K1 * s[0] - K2 * s[1]])


LAWS = np.array([[1.0, 1.0]]) / np.sqrt(2.0)


def test_conserved_model_has_no_equilibrium_without_laws():
    assert equilibrium(_reversible, [1.0, 3.0]) is None


def test_laws_recover_the_analytic_fixed_point_on_the_leaf():
    eq = equilibrium(_reversible, [1.0, 3.0], laws=LAWS)
    assert eq is not None
    assert np.allclose(eq, [A_STAR, TOTAL - A_STAR], atol=1e-9)
    assert np.isclose(eq.sum(), TOTAL, atol=1e-9)  # stayed on the leaf


def test_y_ref_selects_the_leaf_not_the_seed():
    eq = equilibrium(_reversible, [9.0, 9.0], laws=LAWS, y_ref=[1.0, 3.0])
    assert np.isclose(eq.sum(), TOTAL, atol=1e-9)


def test_leaf_spectrum_drops_the_conserved_zero():
    eq = equilibrium(_reversible, [1.0, 3.0], laws=LAWS)
    raw = spectrum(_reversible, eq)
    assert len(raw) == 2 and abs(raw[0]) < 1e-12  # leading raw mode is a zero
    leaf = spectrum(_reversible, eq, laws=LAWS)
    assert len(leaf) == 1
    assert np.isclose(leaf[0].real, -(K1 + K2))


@pytest.mark.slow
def test_dallepezze_equilibrium_needs_its_conservation_laws():
    """The model P0.16 was filed against. Six laws over 23 states: without
    them the search reports no equilibrium for a model that has one, and the
    raw spectrum's leading mode is a conserved zero rather than the rate the
    fixed point actually relaxes at."""
    from hallsim.composite import Composite
    from hallsim.sbml_import import process_from_sbml
    from hallsim.steady_state import conservation_laws
    from demos.models.multi_hallmark import (
        DP14_IRRADIATION_RATE_NAME,
        DP14_SBML_PATH,
    )

    proc = process_from_sbml(
        str(DP14_SBML_PATH),
        name="dp14",
        parameters={DP14_IRRADIATION_RATE_NAME: 0.0},
    )
    comp = Composite(
        processes={"dp14": proc},
        topology={"dp14": {n: f"dp14/{n}" for n in proc._species_names}},
        validate=False,
        semantic_validation=False,
    )
    keys = comp.store_keys()
    y0 = np.asarray(comp.initial_state_vec(keys))
    f, _ = field_from_composite(comp)

    assert equilibrium(f, y0) is None

    laws = conservation_laws(comp, y0)
    assert laws.shape[0] == 6
    eq = equilibrium(f, y0, laws=laws)
    assert eq is not None

    # The late-senescence fixed point, docs/dallepezze2014-critique.md §4.
    at = {k: float(eq[i]) for i, k in enumerate(keys)}
    assert np.isclose(at["dp14/SA_beta_gal"], 9.0315, atol=1e-3)
    assert np.isclose(at["dp14/DNA_damage"], 7.2781, atol=1e-3)
    assert np.isclose(at["dp14/ROS"], 19.943, atol=1e-2)

    assert abs(spectrum(f, eq)[0].real) < 1e-9  # raw: a conserved zero leads
    leaf = spectrum(f, eq, laws)
    assert len(leaf) == 17
    assert np.isclose(leaf[0].real, -0.0730, atol=1e-3)
