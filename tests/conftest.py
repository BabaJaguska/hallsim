import os
from pathlib import Path

# Point reporter_wiring's reference-table loaders at the bundled subset
# fixtures, so the suite runs offline without the full vendored data/ tables
# (which are gitignored and absent in CI). A real HALLSIM_DATA_DIR wins.
os.environ.setdefault(
    "HALLSIM_DATA_DIR",
    str(Path(__file__).parent / "fixtures" / "refdata"),
)
