"""Composite schematic for the flagship multi-hallmark model.

A clean, light wiring diagram — rounded blocks, sans-serif, colored module
titles, italic ``reporters:`` captions. Two hallmark dials feed the dp14
spine, which drives the gz06 p53 oscillator (ψ) and the ih04 NF-κB module
(IKK) through three literature-grounded coupling edges; six gene reporters
read it out. Vector PDF + PNG, print-ready.

    .venv_hallsim/bin/python demos/multi_hallmark_schematic.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
})

C_DP = "#2563eb"    # blue   — dp14 (DallePezze)
C_GZ = "#5b3fc4"    # violet — gz06 (Geva-Zatorsky)
C_NF = "#c0392b"    # red    — ih04 (Ihekwaba NF-κB)
C_DIAL = "#374151"  # slate  — hallmark dials
INK = "#111827"
DIM = "#475569"     # darker slate for legible sub-text
BODY = "#1f2937"    # near-black component lines

F_DP = "#eef4fd"    # light tints
F_GZ = "#f3f0fb"
F_NF = "#fdf0ef"
F_DIAL = "#f4f5f7"


def block(ax, x, y, w, h, edge, fill, r=0.11, lw=2.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3,
        mutation_aspect=1.0))


def arrow(ax, p0, p1, color, rad=0.0, lw=2.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
        mutation_scale=15, linewidth=lw, color=color, zorder=2,
        shrinkA=2, shrinkB=4))


def elabel(ax, x, y, lines, color, rot):
    ax.text(x, y, lines, fontsize=9.8, color=color, ha="center", va="center",
            rotation=rot, rotation_mode="anchor", fontweight="bold",
            zorder=5, linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.14", fc="#ffffff", ec="none"))


def reporters(ax, x, y, text, color, fs=9.8):
    ax.text(x, y, text, fontsize=fs, color=color, ha="center", va="center",
            style="italic", zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(12.8, 5.9))
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 5.9)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- hallmark dials (inputs, left) --------------------------------
    block(ax, 0.4, 3.45, 2.5, 1.02, "#94a3b8", F_DIAL, r=0.10, lw=1.7)
    ax.text(1.65, 4.17, "Genomic Instability", fontsize=11.5, color=INK,
            fontweight="bold", ha="center")
    ax.text(1.65, 3.86, "severity 0-1", fontsize=9.2, color=DIM, ha="center")
    ax.text(1.65, 3.63, "→ etoposide dose", fontsize=9.2, color=DIM,
            ha="center")

    block(ax, 0.4, 1.1, 2.5, 1.18, "#94a3b8", F_DIAL, r=0.10, lw=1.7)
    ax.text(1.65, 2.02, "Deregulated", fontsize=11.5, color=INK,
            fontweight="bold", ha="center")
    ax.text(1.65, 1.77, "Nutrient Sensing", fontsize=11.5, color=INK,
            fontweight="bold", ha="center")
    ax.text(1.65, 1.47, "severity 0-1", fontsize=9.2, color=DIM, ha="center")
    ax.text(1.65, 1.24, "→ mTORC1 (rapamycin)", fontsize=9.2, color=DIM,
            ha="center")

    # ---- dp14 spine (center) ------------------------------------------
    block(ax, 3.7, 1.95, 2.6, 2.0, C_DP, F_DP)
    ax.text(5.0, 3.62, "dp14", fontsize=15, color=C_DP, fontweight="bold",
            ha="center")
    ax.text(5.0, 3.30, "BIOMD582", fontsize=9.4, color=DIM, ha="center")
    for k, line in enumerate(["mTOR · AMPK · FoxO3a",
                              "mitophagy · ROS · DNA damage", "CDKN1A"]):
        ax.text(5.0, 2.98 - 0.30 * k, line, fontsize=9.6, color=BODY,
                ha="center")
    reporters(ax, 5.0, 1.64, "reporters:  CDKN1A · EIF4EBP1 · CYCS · HMOX1",
              C_DP, fs=9.2)

    # ---- gz06 (top-right) ---------------------------------------------
    block(ax, 8.9, 3.72, 3.1, 1.2, C_GZ, F_GZ)
    ax.text(10.45, 4.60, "gz06", fontsize=15, color=C_GZ, fontweight="bold",
            ha="center")
    ax.text(10.45, 4.28, "BIOMD157", fontsize=9.4, color=DIM, ha="center")
    ax.text(10.45, 4.02, "p53–Mdm2 oscillator", fontsize=9.6, color=BODY,
            ha="center")
    reporters(ax, 10.45, 3.50, "reporters:  DDB2", C_GZ)

    # ---- ih04 (bottom-right) ------------------------------------------
    block(ax, 8.9, 0.9, 3.1, 1.2, C_NF, F_NF)
    ax.text(10.45, 1.78, "ih04", fontsize=15, color=C_NF, fontweight="bold",
            ha="center")
    ax.text(10.45, 1.46, "BIOMD230", fontsize=9.4, color=DIM, ha="center")
    ax.text(10.45, 1.20, "NF-κB / IκBα", fontsize=9.6, color=BODY,
            ha="center")
    reporters(ax, 10.45, 0.68, "reporters:  NFKBIA", C_NF)

    # ---- dials → spine ------------------------------------------------
    arrow(ax, (2.9, 3.82), (3.7, 3.28), C_DIAL, rad=-0.14)
    arrow(ax, (2.9, 1.88), (3.7, 2.55), C_DIAL, rad=0.14)

    # ---- three coupling edges -----------------------------------------
    arrow(ax, (6.3, 3.45), (8.9, 4.25), C_GZ, rad=0.14)
    elabel(ax, 7.6, 4.12, "DNA damage → ψ\n(ATM→p53)", C_GZ, 17)

    arrow(ax, (6.3, 2.8), (8.9, 1.82), C_DP, rad=-0.11)
    elabel(ax, 7.6, 2.44, "mTOR → IKK", C_DP, -17)

    arrow(ax, (6.3, 2.3), (8.9, 1.32), C_NF, rad=-0.14)
    elabel(ax, 7.55, 1.66, "DNA damage → IKK", C_NF, -17)

    fig.tight_layout(pad=0.2)
    out = Path(__file__).resolve().parent.parent / "outputs" / \
        "multi_hallmark_calibrate"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"composite_schematic.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    print(f"wrote composite_schematic.png/.pdf -> {out}", flush=True)


if __name__ == "__main__":
    main()
