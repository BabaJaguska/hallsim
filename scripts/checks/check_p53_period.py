"""Is the 0.30 d p53 period in the composite_population figure real, or a
save-grid artefact?

The figure measures the period as the median spacing of local maxima on the
save grid (save_dt=0.02 d), which quantises every period to a multiple of
0.02 d -- and 0.30 d is exactly 15 grid steps. This runs the SAME composite
(nominal params, one cell) on a 10x finer grid and measures the period three
ways: grid-snapped peaks (the figure's estimator), parabola-interpolated
peaks, and the FFT dominant frequency. It also reports the period per
oscillation over time, because the composite drives GZ06's psi from DP14's
DNA_damage -- so the period is not expected to be constant.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from hallsim.hallmarks import with_hallmarks  # noqa: E402
from hallsim.models.multi_hallmark import (  # noqa: E402
    build_multi_hallmark_composite,
)
from hallsim.io import outdir  # noqa: E402
from hallsim.scheduler import Scheduler  # noqa: E402

OUT = outdir("checks")
T_END = 15.0
FINE_DT = 0.002
FIG_DT = 0.02


def peak_times(t, x, interpolate):
    """Interior local maxima; parabola-interpolated when asked."""
    i = np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])) + 1
    i = i[x[i] > 0.15 * x.max()]
    if not interpolate:
        return t[i]
    a, b, c = x[i - 1], x[i], x[i + 1]
    shift = 0.5 * (a - c) / (a - 2 * b + c)
    return t[i] + shift * (t[1] - t[0])


def fft_period(t, x):
    """Dominant period from the detrended spectrum, parabola-refined."""
    y = x - x.mean()
    f = np.fft.rfftfreq(len(y), t[1] - t[0])
    p = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    k = int(np.argmax(p[1:])) + 1
    a, b, c = np.log(p[k - 1]), np.log(p[k]), np.log(p[k + 1])
    return 1.0 / (f[k] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0]))


def main():
    comp = with_hallmarks(
        build_multi_hallmark_composite(),
        {"Genomic Instability": 0.5, "Deregulated Nutrient Sensing": 0.5},
    )
    res = Scheduler().run(
        comp,
        (0.0, T_END),
        macro_dt=5.0,
        save_dt=FINE_DT,
        y0=comp.initial_state_vec(),
    )
    t = np.asarray(res.ts)
    x = np.asarray(res.get("gz06/x"))
    dmg = np.asarray(res.get("dp14/DNA_damage"))

    fine = np.diff(peak_times(t, x, interpolate=True))
    stride = int(round(FIG_DT / FINE_DT))
    coarse = np.diff(peak_times(t[::stride], x[::stride], interpolate=False))

    report = {
        "fine_dt": FINE_DT,
        "fig_dt": FIG_DT,
        "n_peaks": int(len(fine) + 1),
        "period_interp_median_d": float(np.median(fine)),
        "period_interp_cv_pct": float(fine.std() / fine.mean() * 100),
        "period_gridsnap_median_d": float(np.median(coarse)),
        "period_gridsnap_cv_pct": float(coarse.std() / coarse.mean() * 100),
        "period_fft_d": float(fft_period(t, x)),
        "period_interp_first_d": float(fine[0]),
        "period_interp_last_d": float(fine[-1]),
        "period_interp_median_h": float(np.median(fine) * 24),
        "dna_damage_start": float(dmg[0]),
        "dna_damage_max": float(dmg.max()),
        "dna_damage_end": float(dmg[-1]),
    }
    print(json.dumps(report, indent=2))
    (OUT / "p53_period_check.json").write_text(json.dumps(report, indent=2))

    pt = peak_times(t, x, interpolate=True)
    np.savez(OUT / "p53_period_check.npz", t=t, x=x, dmg=dmg, peaks=pt)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(t, x, lw=0.7, color="#1f2937")
    ax[0].plot(pt, np.interp(pt, t, x), "o", ms=2.5, color="#d97706")
    ax[0].set_ylabel("p53 (gz06/x)")
    ax[0].set_title(
        f"period: interp median {np.median(fine)*24:.2f} h, "
        f"grid-snapped {np.median(coarse)*24:.2f} h, "
        f"FFT {fft_period(t, x)*24:.2f} h"
    )
    ax[1].plot(pt[1:], fine * 24, "-o", ms=3, color="#d97706",
               label="period (interp)")
    ax[1].axhline(FIG_DT * 15 * 24, ls="--", lw=0.9, color="0.5",
                  label="figure value 0.30 d")
    tw = ax[1].twinx()
    tw.plot(t, dmg, lw=1.0, color="#2563eb", alpha=0.6)
    tw.set_ylabel("DP14 DNA_damage", color="#2563eb")
    ax[1].set_ylabel("period (h)")
    ax[1].set_xlabel("time (days)")
    ax[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "p53_period_check.png", dpi=150)


if __name__ == "__main__":
    main()
