#!/usr/bin/env python
"""Can the stiffness verdict be reached without the full spectrum?

    python scripts/compare_abscissa_methods.py

`classify_spectrum` needs two numbers from a Jacobian: the spectral abscissa
`max(-Re lambda)`, which decides the solver, and `max|Im lambda|`, which sets an
anti-aliased `save_dt`. It gets them from `np.linalg.eigvals`, which computes
every eigenvalue of a non-symmetric matrix.

Three questions, over the cached corpus from ``build_jacobian_corpus.py``:

1. **Soundness.** ``omega = -lambda_min((J + J^T)/2)`` is the numerical
   abscissa. For any eigenvalue with unit eigenvector ``v``,
   ``Re lambda = v*((J + J*)/2)v``, so it is bounded by that matrix's extreme
   eigenvalues -- hence ``omega >= alpha`` always. A violation is a bug in the
   derivation, not an unusual model.

2. **Verdict impact.** Stiff iff ``alpha * dt > max_explicit_substeps``. Rather
   than pick a ``dt``, report the crossing point ``dt* = threshold / alpha`` for
   each estimator; the interval between them is the band of ``macro_dt`` where
   they would route differently.

3. **Does the shipped ARPACK path recover the abscissa?** Above 512 states
   `_extremal_eigenvalues` asks for ``which="LM"`` -- largest *magnitude*. For an
   oscillator the largest-magnitude modes are the imaginary ones, so the fastest
   decay need not be in the returned subset. Forced here on every model by
   lowering ``DENSE_JACOBIAN_MAX_DIM``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla

import hallsim  # noqa: F401
from hallsim import stiffness as stiff_mod
from hallsim.config import DEFAULT_MAX_EXPLICIT_SUBSTEPS as THRESH

CACHE = Path.home() / ".cache" / "hallsim" / "jacobians"
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "jacobian_corpus"

#: Measured in VCC/scratch/2026-09-04_response_programs/_cycle_average_batch.jsonl
OSCILLATORY = {243, 453, 560, 407, 489, 720, 452, 581, 954, 704}
SETTLED = {140, 488, 287, 582}
NON_NORMAL = {632}  # P0.33: block-triangular, ||J[damage, cycle]|| == 0

log = logging.getLogger("abscissa")


def alpha_exact(J):
    """What classify_spectrum computes today: fastest decay over decaying modes."""
    decay = -np.linalg.eigvals(J).real
    pos = decay[decay > 0]
    return float(pos.max()) if pos.size else 0.0


def omega_sym(J):
    """Numerical abscissa: rigorous upper bound, one symmetric eigenvalue."""
    S = (J + J.T) / 2.0
    lo = sla.eigh(S, subset_by_index=[0, 0], eigvals_only=True)[0]
    return float(max(-lo, 0.0))


def alpha_arpack(J):
    """The abscissa the shipped iterative path would report, forced on."""
    n = J.shape[0]
    if n < 5:  # ARPACK needs k < n-1; below that the dense path always runs
        return None, None
    from scipy.sparse.linalg import eigs

    k = max(1, min(stiff_mod.ITERATIVE_EIGS_K, n - 2))
    try:
        ev = eigs(J.astype(float), k=k, which="LM",
                  return_eigenvectors=False, maxiter=n * 100)
    except Exception as exc:
        return None, f"{type(exc).__name__}"
    decay = -ev.real
    pos = decay[decay > 0]
    return (float(pos.max()) if pos.size else 0.0), float(np.abs(ev.imag).max())


def label_of(key):
    if not key.startswith("BIOMD"):
        return "vendored"
    acc = int(key.replace("BIOMD", ""))
    if acc in OSCILLATORY:
        return "oscillatory"
    if acc in SETTLED:
        return "settled"
    if acc in NON_NORMAL:
        return "non-normal"
    return "unclassified"


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s: %(message)s")
    files = sorted(CACHE.glob("*.npz"))
    if not files:
        sys.exit(f"no corpus at {CACHE}; run build_jacobian_corpus.py first")

    rows = []
    for f in files:
        J = np.load(f, allow_pickle=True)["J"]
        if J.ndim != 2 or J.shape[0] != J.shape[1] or not np.all(np.isfinite(J)):
            continue
        t = time.perf_counter()
        a = alpha_exact(J)
        t_exact = time.perf_counter() - t
        t = time.perf_counter()
        w = omega_sym(J)
        t_omega = time.perf_counter() - t
        t = time.perf_counter()
        a_arp, im_arp = alpha_arpack(J)
        t_arp = time.perf_counter() - t
        im_exact = float(np.abs(np.linalg.eigvals(J).imag).max())

        rows.append({
            "key": f.stem, "n": int(J.shape[0]), "class": label_of(f.stem),
            "alpha_exact": a, "omega_sym": w, "alpha_arpack": a_arp,
            "max_abs_im_exact": im_exact,
            "max_abs_im_arpack": im_arp if isinstance(im_arp, float) else None,
            "t_exact_s": t_exact, "t_omega_s": t_omega, "t_arpack_s": t_arp,
            "dt_cross_exact": THRESH / a if a > 0 else None,
            "dt_cross_omega": THRESH / w if w > 0 else None,
        })

    (OUT / "abscissa_methods.json").write_text(json.dumps(rows, indent=1))

    # --- 1. soundness -----------------------------------------------------
    tol = 1e-9
    viol = [r for r in rows
            if r["omega_sym"] < r["alpha_exact"] * (1 - tol) - tol]
    print(f"\n  === 1. soundness: omega >= alpha ===")
    print(f"  Jacobians            : {len(rows)}")
    print(f"  violations           : {len(viol)}")
    for r in viol[:5]:
        print(f"    {r['key']:22s} n={r['n']:4d} omega={r['omega_sym']:.6g} "
              f"< alpha={r['alpha_exact']:.6g}")

    ratios = np.array([r["omega_sym"] / r["alpha_exact"] for r in rows
                       if r["alpha_exact"] > 0])
    print(f"  omega/alpha          : median {np.median(ratios):.3f}  "
          f"p90 {np.percentile(ratios, 90):.3f}  max {ratios.max():.3g}")
    print(f"  exact within 1%      : {(ratios < 1.01).sum()}/{ratios.size}")

    # --- 2. verdict impact ------------------------------------------------
    print(f"\n  === 2. verdict, threshold = {THRESH} substeps ===")
    both = [r for r in rows if r["dt_cross_exact"] and r["dt_cross_omega"]]
    band = np.array([r["dt_cross_exact"] / r["dt_cross_omega"] for r in both])
    print(f"  models with a crossing point : {len(both)}")
    print(f"  dt band width (exact/omega)  : median {np.median(band):.3f}  "
          f"p90 {np.percentile(band, 90):.3f}  max {band.max():.3g}")
    print(f"  band within 10%              : {(band < 1.1).sum()}/{band.size}")

    # --- 3. does ARPACK LM recover the abscissa? --------------------------
    print(f"\n  === 3. shipped ARPACK which='LM' vs exact ===")
    print(f"  {'class':>14} {'n':>4} {'arpack/exact':>13} {'im arpack/exact':>16}")
    for cls in ("oscillatory", "settled", "non-normal", "unclassified",
                "vendored"):
        sel = [r for r in rows if r["class"] == cls
               and isinstance(r["alpha_arpack"], float)
               and r["alpha_exact"] > 0]
        if not sel:
            continue
        rat = np.array([r["alpha_arpack"] / r["alpha_exact"] for r in sel])
        imr = np.array([(r["max_abs_im_arpack"] or 0.0)
                        / max(r["max_abs_im_exact"], 1e-300) for r in sel])
        miss = int((rat < 0.99).sum())
        print(f"  {cls:>14} {len(sel):4d} "
              f"median {np.median(rat):.3f}, under-reports {miss:2d} "
              f"  median {np.median(imr):.3f}")

    # --- timing -----------------------------------------------------------
    big = [r for r in rows if r["n"] >= 32]
    if big:
        te = sum(r["t_exact_s"] for r in big)
        to = sum(r["t_omega_s"] for r in big)
        print(f"\n  === timing, n >= 32 ({len(big)} models) ===")
        print(f"  eigvals total {te * 1e3:8.1f}ms   omega total "
              f"{to * 1e3:8.1f}ms   speedup {te / max(to, 1e-12):.1f}x")


if __name__ == "__main__":
    main()
