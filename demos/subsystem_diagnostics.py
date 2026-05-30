"""Simulate each multi-hallmark subsystem solo on its native clock and plot.

Each of the three composed SBML models runs alone (no coupling, no
reconciliation) over a window sized to its own native time unit, so the
intrinsic dynamics — and their wildly different timescales — are visible
side by side:

- DP14 (days)    : senescence signalling / damage / mitophagy
- GZ06 (hours)   : p53–Mdm2 oscillator
- NFKB (seconds) : NF-κB limit-cycle oscillator

Figures are written to ``outputs/subsystem_diagnostics/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hallsim.composite import Composite  # noqa: E402
from hallsim.scheduler import Scheduler  # noqa: E402
from hallsim.sbml_import import process_from_sbml  # noqa: E402
from hallsim.models.multi_hallmark import (  # noqa: E402
    DP14_SBML_PATH,
    DP14_IRRADIATION_RATE_NAME,
    DP14_IRRADIATION_RATE_DEFAULT,
    DP14_MTOR_PHOS_RATE_NAME,
    DP14_MTOR_PHOS_RATE_DEFAULT,
    GZ06_SBML_PATH,
    GZ06_PSI_NAME,
    GZ06_PSI_DEFAULT,
    NFKB_SBML_PATH,
)

OUT = Path(__file__).parent.parent / "outputs" / "subsystem_diagnostics"


def run_solo(proc, t_end, n_save):
    comp = Composite(
        processes={proc._name: proc},
        topology={},
        validate=False,
        semantic_validation={"check_semantics": False},
    )
    res = Scheduler().run(
        comp,
        t_span=(0.0, t_end),
        macro_dt=t_end,
        y0=comp.initial_state_vec(),
        save_dt=t_end / n_save,
    )
    return np.asarray(res.ts), res


def plot_subsystem(title, unit, ts, res, species, fname):
    fig, ax = plt.subplots(figsize=(9, 5))
    for path, label in species:
        ax.plot(ts, np.asarray(res.get(path)), label=label, lw=1.5)
    ax.set_xlabel(f"native time ({unit})")
    ax.set_ylabel("species value")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=120)
    plt.close(fig)
    print(f"  wrote {OUT / fname}")


def main():
    # DP14 — native unit: days. Carries the composite's hallmark-exposed
    # rate constants at their defaults.
    dp14 = process_from_sbml(
        str(DP14_SBML_PATH),
        name="dp14",
        parameters={
            DP14_MTOR_PHOS_RATE_NAME: DP14_MTOR_PHOS_RATE_DEFAULT,
            DP14_IRRADIATION_RATE_NAME: DP14_IRRADIATION_RATE_DEFAULT,
        },
    )
    ts, res = run_solo(dp14, t_end=50.0, n_save=400)
    plot_subsystem(
        "DP14 (DallePezze 2014) — native unit: days",
        "days",
        ts,
        res,
        [
            ("dp14/mTORC1_pS2448", "mTORC1_pS2448"),
            ("dp14/CDKN1A", "CDKN1A (p21)"),
            ("dp14/DNA_damage", "DNA_damage"),
            ("dp14/ROS", "ROS"),
            ("dp14/Mito_mass_new", "Mito_mass_new"),
        ],
        "dp14.png",
    )

    # GZ06 — native unit: hours. p53–Mdm2 oscillator at psi default.
    gz06 = process_from_sbml(
        str(GZ06_SBML_PATH),
        name="gz06",
        parameters={GZ06_PSI_NAME: GZ06_PSI_DEFAULT},
    )
    ts, res = run_solo(gz06, t_end=100.0, n_save=600)
    plot_subsystem(
        "GZ06 (Geva-Zatorsky 2006) — native unit: hours",
        "hours",
        ts,
        res,
        [("gz06/x", "p53 (x)"), ("gz06/y", "Mdm2 (y)")],
        "gz06.png",
    )

    # NFKB — native unit: seconds. NF-κB limit-cycle oscillator.
    nfkb = process_from_sbml(str(NFKB_SBML_PATH), name="nfkb")
    ts, res = run_solo(nfkb, t_end=30000.0, n_save=600)
    plot_subsystem(
        "NFKB (Ihekwaba 2004) — native unit: seconds",
        "seconds",
        ts,
        res,
        [
            ("nfkb/NFkBn", "NF-κB nuclear"),
            ("nfkb/IkBa", "IκBα protein"),
            ("nfkb/IkBat", "IκBα transcript"),
            ("nfkb/IKK", "IKK"),
        ],
        "nfkb.png",
    )


if __name__ == "__main__":
    main()
