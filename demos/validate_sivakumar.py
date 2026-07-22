#!/usr/bin/env python
"""Run hallsim validation layer on all 5 Sivakumar2011 SBML models."""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hallsim.sbml_import import process_from_sbml
from hallsim.validation import CompositeValidator

MODEL_DIR = Path(__file__).parent.parent / "models" / "sivakumar2011"

models = {
    "EGF (394)": MODEL_DIR / "egf_BIOMD0000000394.xml",
    "Shh (395)": MODEL_DIR / "shh_BIOMD0000000395.xml",
    "Notch (396)": MODEL_DIR / "notch_BIOMD0000000396.xml",
    "Wnt (397)": MODEL_DIR / "wnt_BIOMD0000000397.xml",
    "Crosstalk (398)": MODEL_DIR / "crosstalk_BIOMD0000000398.xml",
}

validator = CompositeValidator()

for label, sbml_path in models.items():
    print(f"\n{'='*60}")
    print(f"{label}: {sbml_path.name}")
    print("=" * 60)
    proc = process_from_sbml(str(sbml_path), name=label)
    species = list(proc.ports_schema().keys())

    processes = {label: proc}
    topology = {label: {s: s for s in species}}

    report = validator.validate(processes, topology)
    print(report.summary())
    if not report.is_valid:
        print("  ** VALIDATION FAILED **")
