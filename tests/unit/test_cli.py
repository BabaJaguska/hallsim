"""The ``simulate`` CLI surface stays wired to what it dispatches into.

``simulate calibrate`` once imported a demo function that had been renamed away,
so the command raised ImportError for anyone who ran it — nothing here caught
it. These tests are cheap: they check the wiring, not the science.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from hallsim.cli import simulate

DEMOS = Path(__file__).resolve().parents[2] / "demos"


@pytest.mark.parametrize("name", sorted(simulate.commands))
def test_command_help_works(name):
    """Every registered command exposes help without blowing up."""
    result = CliRunner().invoke(simulate, [name, "--help"])
    assert result.exit_code == 0, result.output


def test_multi_hallmark_dispatch_targets_exist():
    """The subcommands the CLI offers resolve to handlers the demo defines.

    The CLI reaches into ``demos/`` at call time, so a rename there breaks the
    command silently until someone runs it.
    """
    sys.path.insert(0, str(DEMOS))
    try:
        from multi_hallmark_calibrate import _COMMANDS
    finally:
        sys.path.remove(str(DEMOS))

    offered = simulate.commands["multi-hallmark"].params[0].type.choices
    # "calibrate" is "run" with args.calibrate set, not its own handler.
    assert {c for c in offered if c != "calibrate"} <= set(_COMMANDS)
    assert "run" in _COMMANDS


def test_clamp_options_name_real_config_keys():
    """``simulate clamp`` forwards its options as config overrides, so an
    option whose name drifts from the demo's config key silently stops
    applying."""
    sys.path.insert(0, str(DEMOS))
    try:
        from clamp_setpoint import DEFAULTS
    finally:
        sys.path.remove(str(DEMOS))

    offered = {p.name for p in simulate.commands["clamp"].params} - {"help"}
    assert offered and offered <= set(DEFAULTS)


class TestVerbosity:
    """The CLI is the documented entry point, and it configured no logging at
    all: every `log.info` was dropped and every `log.warning` arrived through
    `logging.lastResort` as bare stderr with no level or logger name."""

    @staticmethod
    def _levels(argv):
        import logging

        CliRunner().invoke(simulate, argv + ["info"])
        return (
            logging.getLogger().level,
            logging.getLogger("hallsim").getEffectiveLevel(),
        )

    def test_default_is_warnings(self):
        import logging

        root, hallsim = self._levels([])
        assert root == logging.WARNING
        assert hallsim == logging.WARNING

    def test_verbose_lifts_hallsim_only(self):
        """-v means "what HallSim decided", not JAX's backend probing, so the
        root stays where it was."""
        import logging

        root, hallsim = self._levels(["-v"])
        assert hallsim == logging.INFO
        assert root == logging.WARNING

    def test_vv_reaches_debug(self):
        import logging

        assert self._levels(["-vv"])[1] == logging.DEBUG

    def test_quiet_silences_both(self):
        import logging

        root, hallsim = self._levels(["-q"])
        assert root == logging.ERROR
        assert hallsim == logging.ERROR

    def test_a_handler_is_installed_and_names_level_and_logger(self):
        """Without one, records reached stderr via `logging.lastResort` as
        bare text: no level, no logger name, nothing to filter on."""
        import logging

        CliRunner().invoke(simulate, ["info"])
        handlers = logging.getLogger().handlers
        assert handlers, "no handler installed; records fall to lastResort"
        fmt = handlers[0].formatter._fmt
        assert "%(levelname)s" in fmt and "%(name)s" in fmt


class TestTheFittedSetIsOneList:
    """``fitted`` names exactly the set a fit uses; a saved fit is scored
    with the set it had. There is no second list describing the set by
    subtraction — that produced a checkpoint indexed by names it lacked."""

    def _build(self, **kw):
        sys.path.insert(0, str(DEMOS))
        try:
            from multi_hallmark_calibrate import build_problem
        finally:
            sys.path.remove(str(DEMOS))
        return build_problem(**kw)

    def test_fitted_names_exactly_the_set(self):
        full = set(self._build().param_refs)
        chosen = ("CDKN1A_transcr", "alpha_x_control")
        assert set(chosen) < full
        assert set(self._build(fitted=chosen).param_refs) == set(chosen)

    def test_a_name_outside_the_default_set_is_refused(self):
        with pytest.raises(KeyError, match="not in the fitted set"):
            self._build(fitted=("not_a_parameter",))
