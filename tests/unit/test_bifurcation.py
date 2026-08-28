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
    codim1_scan,
    critical_eigenvalue,
    fold_coefficient,
    spectrum,
    unstable_dim,
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


def test_critical_eigenvalue_crosses_zero_with_mu():
    f_of = lambda m: _normal_form(m, -1.0)  # noqa: E731
    assert critical_eigenvalue(f_of(-0.1), [0.0, 0.0]).real < 0
    assert critical_eigenvalue(f_of(+0.1), [0.0, 0.0]).real > 0
    # A Hopf is the complex case: the crossing pair keeps its frequency.
    assert (
        abs(abs(critical_eigenvalue(f_of(0.0), [0.0, 0.0]).imag) - OMEGA)
        < 1e-9
    )


def test_unstable_dim_counts_the_pair():
    assert unstable_dim(_normal_form(-0.1, -1.0), [0.0, 0.0]) == 0
    assert unstable_dim(_normal_form(+0.1, -1.0), [0.0, 0.0]) == 2


def test_supercritical_l1_negative():
    l1 = first_lyapunov_coefficient(_normal_form(0.0, -1.0), [0.0, 0.0])
    assert l1 < 0


def test_subcritical_l1_positive():
    l1 = first_lyapunov_coefficient(_normal_form(0.0, +1.0), [0.0, 0.0])
    assert l1 > 0


def test_scan_locates_and_classifies_a_hopf():
    params = np.linspace(-0.25, 0.25, 51)
    sup = codim1_scan(
        lambda m: _normal_form(m, -1.0), params, x0_guess=[0.02, 0.0]
    )
    assert len(sup) == 1
    h = sup[0]
    assert h.kind == "hopf"
    assert abs(h.param) < 1e-3  # Hopf sits at mu=0
    assert abs(h.omega - OMEGA) < 1e-2
    assert h.supercritical is True

    sub = codim1_scan(
        lambda m: _normal_form(m, +1.0), params, x0_guess=[0.02, 0.0]
    )
    assert len(sub) == 1
    assert sub[0].supercritical is False


def test_scan_no_false_positive_when_no_crossing():
    # mu stays negative -> stable focus throughout, no Hopf.
    params = np.linspace(-0.5, -0.1, 21)
    assert (
        codim1_scan(
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


# --- folds ----------------------------------------------------------------
# The three real-crossing normal forms, each with a closed-form quadratic
# coefficient a = 1/2 f'' and transversality s = <p, df/dmu>. No complex pair
# is ever involved in any of them.


def _scalar(g):
    return lambda s: jnp.array([g(s[0])])


def _scalar_family(g):
    return lambda mu: (lambda s: jnp.array([g(mu, s[0])]))


def test_saddle_node_coefficients():
    # x' = mu + x^2 at mu=0: a = 1, and mu enters directly so s = 1.
    a, q, p = fold_coefficient(_scalar(lambda x: 0.0 + x * x), [0.0])
    assert np.isclose(a, 1.0)
    assert np.isclose(abs(float(q[0] * p[0])), 1.0)


def test_fold_coefficient_rejects_a_complex_crossing():
    with pytest.raises(ValueError, match="complex"):
        fold_coefficient(_normal_form(0.0, -1.0), [0.0, 0.0])


def test_scan_finds_the_saddle_node_where_the_branch_ends():
    # x' = mu + x^2: stable branch x = -sqrt(-mu) exists only for mu < 0 and
    # ends at mu = 0. The grid deliberately straddles it without landing on
    # it, so the location comes from the refinement, not the sampling.
    params = np.linspace(-0.25, 0.25, 50)
    assert not np.any(np.isclose(params, 0.0))
    found = codim1_scan(
        _scalar_family(lambda mu, x: mu + x * x), params, x0_guess=[-0.5]
    )
    assert len(found) == 1
    b = found[0]
    assert b.kind == "fold"
    assert abs(b.param) < 1e-3
    assert b.omega == 0.0
    assert np.isclose(b.coefficient, 1.0, atol=1e-3)
    assert abs(b.transversality) > 0.5  # mu enters f directly
    assert b.degenerate is False


def test_transcritical_has_no_parameter_transversality():
    # x' = mu*x - x^2. The branch x=0 persists for every mu, so nothing
    # vanishes; the crossing shows up as the unstable dimension changing.
    # a = -1 but df/dmu = x = 0 at the crossing, which is what makes it
    # transcritical rather than a saddle-node.
    found = codim1_scan(
        _scalar_family(lambda mu, x: mu * x - x * x),
        np.linspace(-0.25, 0.25, 50),
        x0_guess=[0.0],
    )
    assert len(found) == 1
    b = found[0]
    assert b.kind == "fold"
    assert abs(b.param) < 1e-2
    assert np.isclose(b.coefficient, -1.0, atol=1e-3)
    assert abs(b.transversality) < 1e-6
    assert b.degenerate is True


def test_pitchfork_has_no_quadratic_coefficient():
    # x' = mu*x - x^3: f'' = 0 at the origin, so a = 0 and the saddle-node
    # classification does not hold.
    found = codim1_scan(
        _scalar_family(lambda mu, x: mu * x - x * x * x),
        np.linspace(-0.25, 0.25, 50),
        x0_guess=[0.0],
    )
    assert len(found) == 1
    assert found[0].kind == "fold"
    assert abs(found[0].coefficient) < 1e-6
    assert found[0].degenerate is True


def test_scan_reports_a_newton_failure_as_nothing():
    # No equilibrium anywhere: x' = 1 + x^2. Losing the branch is not a fold.
    found = codim1_scan(
        _scalar_family(lambda mu, x: 1.0 + mu * 0.0 + x * x),
        np.linspace(-0.25, 0.25, 20),
        x0_guess=[0.0],
    )
    assert found == []
