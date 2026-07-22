"""Tests for SBMLProcess timed-intervention mechanisms."""

import jax.numpy as jnp
import pytest

from hallsim.composite import Composite
from hallsim.models.multi_hallmark import GZ06_PSI_NAME, GZ06_SBML_PATH
from hallsim.sbml_import import process_from_sbml
from hallsim.scheduler import Scheduler


def _gz06(psi):
    return process_from_sbml(
        str(GZ06_SBML_PATH), name="gz06", parameters={GZ06_PSI_NAME: psi}
    )


def _solo_ys(proc, t_end=30.0):
    comp = Composite(
        processes={"gz06": proc},
        topology={"gz06": {n: f"gz06/{n}" for n in proc._species_names}},
        validate=False,
        semantic_validation=False,
    )
    res = Scheduler().run(
        comp,
        t_span=(0.0, t_end),
        macro_dt=t_end,
        y0=comp.initial_state_vec(),
        save_dt=t_end / 30.0,
    )
    return res.ys


class TestWithParamStep:
    def test_rejects_unknown_param(self):
        with pytest.raises(KeyError, match="not an SBML constant"):
            _gz06(0.9).with_param_step("not_a_param", 5.0, 0.1)

    def test_records_step(self):
        p = _gz06(0.9).with_param_step(GZ06_PSI_NAME, 5.0, 0.1)
        assert p._param_steps == ((GZ06_PSI_NAME, 5.0, 0.1),)

    def test_step_never_fires_holds_value_before(self):
        # t_step past the horizon → the constant stays at value_before the
        # whole run, matching a plain process pinned to value_before.
        stepped = _gz06(0.9).with_param_step(GZ06_PSI_NAME, 1e9, 0.2)
        pinned = _gz06(0.2)
        assert jnp.allclose(_solo_ys(stepped), _solo_ys(pinned), atol=1e-6)

    def test_step_at_zero_holds_configured_value(self):
        # t_step=0 → the configured parameters value is in force from t=0,
        # matching a plain process pinned to that value (value_before unused).
        stepped = _gz06(0.9).with_param_step(GZ06_PSI_NAME, 0.0, 0.2)
        pinned = _gz06(0.9)
        assert jnp.allclose(_solo_ys(stepped), _solo_ys(pinned), atol=1e-6)

    def test_step_changes_trajectory_midway(self):
        # A mid-run step must diverge from both endpoints' pinned runs.
        stepped = _gz06(0.9).with_param_step(GZ06_PSI_NAME, 15.0, 0.2)
        assert not jnp.allclose(_solo_ys(stepped), _solo_ys(_gz06(0.2)))
        assert not jnp.allclose(_solo_ys(stepped), _solo_ys(_gz06(0.9)))
