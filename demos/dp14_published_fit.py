"""Verify DallePezze 2014 against the fitting data its own paper deposited.

The check that licenses every later finding about a model: does the deposit
still reproduce the fit the publication claims? The paper reports
chi2 = 70.4278 over 127 points and 14 observables; the deposited PottersWheel
workbook (PLOS supplement s028) carries those points with their standard
deviations, so the number is recomputable rather than quotable.

    python demos/dp14_published_fit.py

Observables are not states: each is ``scale_<obs> x sum(species)`` declared as
an SBML assignment rule, so the observable map is read out of the source rather
than hand-listed, and the scales come from the model's own fitted parameters.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from hallsim.composite import Composite
from hallsim.intake import published_fit_chi2
from hallsim.sbml_import import process_from_sbml
from demos.models.multi_hallmark import DP14_SBML_PATH

log = logging.getLogger("hallsim.demo.dp14_fit")

ROOT = Path(__file__).resolve().parent.parent
FITTING_XLS = (
    ROOT / "data" / "dallepezze2014" / "pcbi.1003728.s028_fitting_data.xls"
)
#: The paper's stated chi2 (Table 1 / Text S1).
REPORTED_CHI2 = 70.4278


def observable_map(xml_path=DP14_SBML_PATH):
    """``{obs_name: (scale_param, (species, ...))}`` from the SBML's own
    assignment rules — the paper's observables in terms of model states."""
    xml = Path(xml_path).read_text()
    species = set(re.findall(r'<species\b[^>]*\bid="([^"]+)"', xml))
    out = {}
    for rule in re.finditer(
        r'<assignmentRule[^>]*variable="([^"]+)".*?</assignmentRule>',
        xml,
        re.S,
    ):
        name = rule.group(1)
        if not name.endswith("_obs"):
            continue
        refs = re.findall(r"<ci>\s*([^<\s]+)\s*</ci>", rule.group(0))
        scale = next((r for r in refs if r.startswith("scale")), None)
        members = tuple(r for r in refs if r in species)
        if scale and members:
            out[name] = (scale, members)
    return out


def load_observations(path=FITTING_XLS):
    """``{obs_name: (times, values, sds)}`` from the deposited workbook.

    Columns run ``Time (days) | Stimulus | <obs> | stdCol-<obs> | ...``; NaN
    marks a timepoint that observable was not measured at, and those are
    dropped rather than imputed.
    """
    import xlrd

    sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
    header_row = next(
        r
        for r in range(sheet.nrows)
        if str(sheet.row_values(r)[0]).startswith("Time")
    )
    header = [str(c).strip() for c in sheet.row_values(header_row)]
    rows = np.array(
        [
            [_num(c) for c in sheet.row_values(r)]
            for r in range(header_row + 1, sheet.nrows)
            if str(sheet.row_values(r)[0]).strip()
        ]
    )
    times = rows[:, 0]

    out = {}
    for i, name in enumerate(header):
        if name.startswith(("Time", "Stimulus", "stdCol-")) or not name:
            continue
        sd_col = header.index(f"stdCol-{name}")
        keep = ~np.isnan(rows[:, i]) & ~np.isnan(rows[:, sd_col])
        if keep.any():
            out[name] = (times[keep], rows[keep, i], rows[keep, sd_col])
    return out


def _num(cell):
    try:
        return float(cell)
    except (TypeError, ValueError):
        return float("nan")


def verify(reported: float | None = REPORTED_CHI2):
    """Score the deposit against its own fitting data. Returns a FitReport."""
    obs_map = observable_map()
    data = load_observations()
    proc = process_from_sbml(str(DP14_SBML_PATH), name="dp14")
    comp = Composite(
        processes={"dp14": proc},
        topology={"dp14": {n: f"dp14/{n}" for n in proc._species_names}},
        validate=False,
        semantic_validation=False,
    )
    observations, scales = {}, {}
    for name, (t, y, sd) in data.items():
        if name not in obs_map:
            log.warning("no assignment rule for observable %r — skipped", name)
            continue
        scale_param, members = obs_map[name]
        key = tuple(f"dp14/{m}" for m in members)
        observations[key] = (t, y, sd)
        scales[key] = float(proc.parameters[scale_param])
    return published_fit_chi2(
        comp, observations, scales=scales, reported=reported, save_dt=0.01
    )


def run_demo():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = verify()
    print(report)
    return report


if __name__ == "__main__":
    run_demo()
