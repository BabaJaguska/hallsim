"""Bifurcation analysis validated on the analytic Hopf normal form.

    x' = mu x - omega y +/- x (x^2 + y^2)
    y' = omega x + mu y +/- y (x^2 + y^2)

Hopf at mu=0; the cubic sign fixes criticality: ``-`` supercritical
(l1<0), ``+`` subcritical (l1>0). These have closed-form answers, so they
pin the first Lyapunov coefficient's sign and the Hopf locator.
"""

import jax.numpy as jnp
import numpy as np

from hallsim.bifurcation import (
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
        hopf_scan(lambda m: _normal_form(m, -1.0), params, x0_guess=[0.02, 0.0])
        == []
    )
