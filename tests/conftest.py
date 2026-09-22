import os
from pathlib import Path

import pytest

# Point reporter_wiring's reference-table loaders at the bundled subset
# fixtures, so the suite runs offline without the full vendored data/ tables
# (which are gitignored and absent in CI). A real HALLSIM_DATA_DIR wins.
os.environ.setdefault(
    "HALLSIM_DATA_DIR",
    str(Path(__file__).parent / "fixtures" / "refdata"),
)


@pytest.fixture(autouse=True)
def _fresh_routing_memory():
    """The Scheduler shares routing verdicts across instances for the whole
    process. A test that measures cold routing must not inherit a verdict
    from whichever test ran before it, so each test starts with none."""
    from hallsim import scheduler

    scheduler._RUNG_MEMORY.clear()
    yield
    scheduler._RUNG_MEMORY.clear()
