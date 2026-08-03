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
