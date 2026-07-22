"""Tests for the XPPAUT .ode importer (hallsim.xpp_import)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hallsim.composite import Composite
from hallsim.scheduler import Scheduler
from hallsim.xpp_import import (
    _XPP_MATH,
    UnsupportedXPPFeatureError,
    XPPParseError,
    XPPProcess,
    _xpp_expr_to_python,
    process_from_xpp,
)


def _ev(expr, **vars):
    """Translate an XPP expression and evaluate it over the JAX namespace."""
    ns = dict(_XPP_MATH)
    ns.update({k.lower(): jnp.asarray(float(v)) for k, v in vars.items()})
    return float(eval(_xpp_expr_to_python(expr), ns))


def _write(tmp_path, text, name="model.ode"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def _run(proc, t_span, macro_dt, y0=None):
    """Run a standalone XPPProcess: map every state to its own store path."""
    topo = {proc._name: {s: s for s in proc._state_names}}
    comp = Composite(processes={proc._name: proc}, topology=topo)
    return Scheduler().run(
        comp, t_span=t_span, macro_dt=macro_dt, save_dt=macro_dt, y0=y0
    )


# --- expression translation -------------------------------------------------


def test_power_operator():
    assert _ev("a^2 + b^n", a=3, b=2, n=3) == pytest.approx(9 + 8)


def test_power_is_right_associative():
    # 2^(3^2) = 2^9 = 512, not (2^3)^2 = 64
    assert _ev("2^3^2") == pytest.approx(512)


def test_case_insensitive():
    # Heav and SIN resolve despite the casing; heav(1)*sin(0) = 0
    assert _ev("Heav(X) * SIN(t)", X=1, t=0) == pytest.approx(0.0)


def test_unary_minus_binds_below_power():
    # -x^2 = -(x^2) = -4, not (-x)^2 = 4
    assert _ev("-x^2", x=2) == pytest.approx(-4.0)


def test_if_then_else():
    assert _ev("if(x>0)then(a)else(b)", x=1, a=5, b=9) == pytest.approx(5)
    assert _ev("if(x>0)then(a)else(b)", x=-1, a=5, b=9) == pytest.approx(9)


def test_logical_and_precedence():
    # The reason a real parser is needed: '&' must not bind tighter than '>'.
    assert _ev("if(x>0 & y>0)then(1)else(0)", x=1, y=1) == pytest.approx(1)
    assert _ev("if(x>0 & y>0)then(1)else(0)", x=1, y=-1) == pytest.approx(0)


def test_logical_or():
    assert _ev("if(x>0 | y>0)then(1)else(0)", x=-1, y=2) == pytest.approx(1)
    assert _ev("if(x>0 | y>0)then(1)else(0)", x=-1, y=-1) == pytest.approx(0)


def test_nested_if_then_else():
    assert _ev("if(x>0)then(if(y>0)then(1)else(2))else(3)", x=1, y=1) == 1
    assert _ev("if(x>0)then(if(y>0)then(1)else(2))else(3)", x=1, y=-1) == 2
    assert _ev("if(x>0)then(if(y>0)then(1)else(2))else(3)", x=-1, y=0) == 3


def test_bad_expression_raises():
    with pytest.raises(XPPParseError):
        _xpp_expr_to_python("a +* b")


# --- parsing edge cases -----------------------------------------------------


def test_no_odes_is_error(tmp_path):
    with pytest.raises(XPPParseError):
        process_from_xpp(_write(tmp_path, "par a=1\ndone\n"))


def test_wiener_rejected(tmp_path):
    src = "x'=w\nwiener w\ndone\n"
    with pytest.raises(UnsupportedXPPFeatureError):
        process_from_xpp(_write(tmp_path, src))


def test_dae_rejected(tmp_path):
    src = "x'=-x\n0=y-x\ndone\n"
    with pytest.raises(UnsupportedXPPFeatureError):
        process_from_xpp(_write(tmp_path, src))


def test_unknown_param_override(tmp_path):
    src = "par k=1\nx'=-k*x\ndone\n"
    with pytest.raises(XPPParseError):
        process_from_xpp(_write(tmp_path, src), parameters={"nope": 3.0})


# --- numeric round-trips ----------------------------------------------------


def test_decay_matches_analytic(tmp_path):
    src = "# linear decay\npar k=0.5\ninit x=2\nx' = -k*x\ndone\n"
    proc = process_from_xpp(_write(tmp_path, src))
    assert isinstance(proc, XPPProcess)
    res = _run(proc, t_span=(0.0, 4.0), macro_dt=0.01)
    x_final = float(res.get("x")[-1])
    assert x_final == pytest.approx(2.0 * jnp.exp(-0.5 * 4.0), abs=1e-3)


def test_harmonic_oscillator_conserves_energy(tmp_path):
    # dx/dt = y ; dy/dt = -x   →   x^2 + y^2 conserved
    src = "init x=1, y=0\nx' = y\ndy/dt = -x\ndone\n"
    proc = process_from_xpp(_write(tmp_path, src))
    res = _run(proc, t_span=(0.0, 10.0), macro_dt=0.01)
    energy = res.get("x") ** 2 + res.get("y") ** 2
    assert float(jnp.max(jnp.abs(energy - 1.0))) < 1e-2


def test_dx_dt_and_prime_forms_agree(tmp_path):
    a = process_from_xpp(_write(tmp_path, "par k=0.3\nx'=-k*x\n", "a.ode"))
    b = process_from_xpp(_write(tmp_path, "par k=0.3\ndx/dt=-k*x\n", "b.ode"))
    assert a._ode_py == b._ode_py


def test_user_function_and_fixed_var(tmp_path):
    # fixed variable + user function + ^ all in the RHS
    src = (
        "par k=1.0\n"
        "init x=2\n"
        "f(u) = u^2 / (1 + u^2)\n"
        "gain = k * f(x)\n"
        "x' = -gain\n"
        "done\n"
    )
    proc = process_from_xpp(_write(tmp_path, src))
    res = _run(proc, t_span=(0.0, 1.0), macro_dt=0.01)
    x = res.get("x")
    assert jnp.all(jnp.isfinite(x))
    assert float(x[-1]) < 2.0  # gain is positive → x decays


# --- calibration surface & batching ----------------------------------------


def test_parameters_surface_and_override(tmp_path):
    src = "par k=0.5, b=1.0\nx'=-k*x + b\ndone\n"
    proc = process_from_xpp(_write(tmp_path, src), parameters={"k": 0.9})
    assert proc.parameters["k"] == 0.9
    assert proc.parameters["b"] == 1.0
    fields = {p.field for p in proc.calibratable_params()}
    assert "parameters.k" in fields
    assert "parameters.b" in fields


def test_derivative_is_batched_and_differentiable(tmp_path):
    src = "par k=0.5\nx'=-k*x\ndone\n"
    proc = process_from_xpp(_write(tmp_path, src))

    # shape-polymorphic: a batched port value yields a batched derivative
    batched = proc.derivative(0.0, {"x": jnp.array([1.0, 2.0, 3.0])})
    assert batched["x"].shape == (3,)
    assert jnp.allclose(batched["x"], -0.5 * jnp.array([1.0, 2.0, 3.0]))

    # differentiable w.r.t. state
    g = jax.grad(lambda x: proc.derivative(0.0, {"x": x})["x"])(2.0)
    assert float(g) == pytest.approx(-0.5)


def test_reconciled_to_rescales_rate(tmp_path):
    # native unit = hours; canonical = seconds → rate scales by 1/3600
    src = "par k=1.0\nx'=-k*x\ndone\n"
    proc = process_from_xpp(_write(tmp_path, src), native_time_seconds=3600.0)
    rec = proc.reconciled_to(1.0)
    d_native = proc.derivative(0.0, {"x": 1.0})["x"]
    d_canonical = rec.derivative(0.0, {"x": 1.0})["x"]
    assert float(d_canonical) == pytest.approx(float(d_native) / 3600.0)
