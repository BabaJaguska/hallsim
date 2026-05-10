"""Run ssGSEA on GSE248823 → produce per-condition pathway delta vectors.

Pipeline:
1. Parse the GSE248823 series matrix (Affymetrix HTA 2.0, 30,293 probes,
   20 samples).
2. Parse the GPL17586 platform annotation, extract the primary gene
   symbol per probe (first SYMBOL in the gene_assignment column).
3. Aggregate probe-level expression → gene-level via mean across probes
   that map to the same gene symbol.
4. Run ssGSEA (gseapy) over a curated set of pathway gene sets
   covering the seven HallSim PathwayMapper outputs.
5. Compute per-condition deltas and save them as CSV for the
   concordance demo.

Held-out validation:
- Δ_DDIS: mean(D14 etoposide, REPs 1-2) − mean(D00 etoposide control, REPs 1-2)
- Δ_OIS:  mean(D07 RAS, REPs 1-2) − mean(D00 RAS control, REPs 1-2)
- Δ_DDIS_rapamycin: mean(D14 etoposide+rapa) − mean(D14 etoposide alone)

Calibrating on DDIS and evaluating on OIS gives a held-out r number; the
two arms drive senescence via different upstream mechanisms (DDR vs.
oncogene), so generalization across them is the legitimate test.

Pathway gene sets are curated from MSigDB Hallmark v2023.2 (≈40-200
canonical genes per pathway). They are hardcoded so this script runs
offline without network access; replacing them with the full MSigDB
GMT is straightforward.

Usage::

    .venv_hallsim/bin/python scripts/run_ssgsea.py
    .venv_hallsim/bin/python scripts/run_ssgsea.py --max-iter 5000 --plot
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import gseapy
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "FibroblastsDNA_dmg_Rapamycin"
SERIES_MATRIX = DATA_DIR / "GSE248823_series_matrix.txt"
PLATFORM = DATA_DIR / "GPL17586-45144.txt"
OUT_CSV = ROOT / "data" / "ssgsea_deltas.csv"
OUT_NES_CSV = ROOT / "data" / "ssgsea_nes_per_sample.csv"


# Sample positions in the series matrix (0-indexed columns of the
# 20-sample expression matrix). Determined from !Sample_title metadata.
SAMPLE_COLS = {
    "ETOPOSIDE_D00": [0, 1],  # control for DDIS arm
    "ETOPOSIDE_D07": [2, 3],
    "ETOPOSIDE_D14": [4, 5],  # DDIS late time
    "ETOPOSIDE_RAPAMYCIN_D07": [6, 7],
    "ETOPOSIDE_RAPAMYCIN_D14": [8, 9],  # DDIS+rapa rescue
    "RAS_D00": [10, 11],  # control for OIS arm
    "RAS_D04": [12, 13],
    "RAS_D07": [14, 15],  # OIS late time
    "RAS_DMOG_D04": [16, 17],
    "RAS_DMOG_D07": [18, 19],  # OIS+DMOG rescue
}


# Curated pathway gene sets derived from MSigDB Hallmark v2023 + select
# KEGG/Reactome members. Each list is the canonical core for the pathway;
# trimmed to ~30-80 well-characterized genes for tractability and
# overlap-stability with the WI38/HTA-2 platform's ~21k gene-symbol
# coverage. These align 1:1 with the HallSim PathwayMapper output names.
PATHWAY_GENE_SETS: dict[str, list[str]] = {
    "mtorc1_signaling": [
        # HALLMARK_MTORC1_SIGNALING core: PI3K-AKT-mTOR axis + downstream effectors
        "MTOR", "RPTOR", "MLST8", "DEPTOR", "RHEB", "RPS6", "RPS6KB1",
        "RPS6KB2", "EIF4EBP1", "EIF4EBP2", "EIF4E", "EIF4G1", "AKT1",
        "AKT2", "AKT3", "PIK3CA", "PIK3CB", "PIK3R1", "PTEN", "TSC1",
        "TSC2", "RRAGA", "RRAGB", "RRAGC", "RRAGD", "LAMP1", "LAMP2",
        "ULK1", "MYC", "HIF1A", "SREBF1", "SREBF2", "FASN", "ACACA",
        "GOT1", "PSAT1", "PHGDH", "SLC7A5", "SLC7A11", "ATP5A1",
        "EPRS", "MARS", "GARS",
    ],
    "glycolysis": [
        "HK1", "HK2", "GPI", "PFKL", "PFKM", "PFKP", "ALDOA", "ALDOB",
        "ALDOC", "TPI1", "GAPDH", "GAPDHS", "PGK1", "PGAM1", "ENO1",
        "ENO2", "ENO3", "PKM", "LDHA", "LDHB", "SLC16A1", "SLC16A3",
        "SLC2A1", "SLC2A3", "SLC2A4", "PFKFB1", "PFKFB2", "PFKFB3",
        "PFKFB4", "ME1", "ME2", "PGM1", "GALK1", "ADPGK", "G6PD",
        "PGD", "GFPT1", "GMPPA", "MDH1", "MDH2", "FBP1",
    ],
    "oxphos": [
        # Complex I - NADH dehydrogenase
        "NDUFA1", "NDUFA2", "NDUFA3", "NDUFA4", "NDUFA5", "NDUFA6",
        "NDUFA7", "NDUFA8", "NDUFA9", "NDUFA10", "NDUFB1", "NDUFB2",
        "NDUFB3", "NDUFB4", "NDUFB7", "NDUFB10", "NDUFS1", "NDUFS2",
        "NDUFS3", "NDUFS7", "NDUFS8", "NDUFV1", "NDUFV2",
        # Complex II
        "SDHA", "SDHB", "SDHC", "SDHD",
        # Complex III
        "UQCRC1", "UQCRC2", "UQCRH", "CYC1", "UQCRFS1",
        # Complex IV
        "COX4I1", "COX5A", "COX5B", "COX6A1", "COX6B1", "COX6C",
        "COX7A1", "COX7A2", "COX7B", "COX8A",
        # Complex V (ATP synthase)
        "ATP5F1A", "ATP5F1B", "ATP5F1C", "ATP5F1D", "ATP5F1E",
        "ATP5MG", "ATP5PB", "ATP5PD",
        # Other mito core
        "TOMM20", "TIMM13", "VDAC1", "VDAC2", "ACO2", "IDH3A",
        "IDH3B", "OGDH", "DLST",
    ],
    "nfkb_signaling": [
        # HALLMARK_TNFA_SIGNALING_VIA_NFKB core
        "NFKB1", "NFKB2", "RELA", "RELB", "REL", "NFKBIA", "NFKBIB",
        "NFKBIE", "IKBKB", "IKBKG", "CHUK", "TNF", "TNFAIP3",
        "TNFRSF1A", "TNFRSF1B", "IL6", "IL1A", "IL1B", "CXCL8",
        "CXCL1", "CXCL2", "CXCL3", "CCL2", "CCL5", "CCL20", "ICAM1",
        "VCAM1", "SELE", "PTGS2", "BIRC3", "BCL3", "JUN", "JUNB",
        "FOS", "FOSL1", "EGR1", "EGR2", "EGR3", "DUSP1", "DUSP2",
        "DUSP4", "DUSP5", "ATF3", "PLAU", "PLAUR", "MMP3", "MMP9",
        "CSF2", "CSF1", "IFNGR2", "GADD45B",
    ],
    "ros_pathway": [
        # HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY
        "SOD1", "SOD2", "SOD3", "CAT", "GPX1", "GPX2", "GPX3", "GPX4",
        "GPX7", "GSR", "GCLC", "GCLM", "NQO1", "NFE2L2", "KEAP1",
        "HMOX1", "TXN", "TXN2", "TXNRD1", "TXNRD2", "PRDX1", "PRDX2",
        "PRDX3", "PRDX4", "PRDX5", "PRDX6", "MGST1", "MGST2", "GSS",
        "GSTA1", "GSTM1", "GSTP1", "G6PD", "PGD", "FTH1", "FTL",
        "MT1A", "MT1X", "MT2A", "EGLN1",
    ],
    "senescence": [
        # KEGG_CELLULAR_SENESCENCE + canonical senescence/SASP markers
        "CDKN1A", "CDKN2A", "CDKN2B", "TP53", "MDM2", "RB1", "RBL1",
        "RBL2", "E2F1", "E2F2", "E2F3", "CCNE1", "CCNE2", "CCND1",
        "CDK2", "CDK4", "CDK6", "ATM", "ATR", "CHEK1", "CHEK2",
        "GADD45A", "GADD45B", "BBC3", "BAX", "BCL2", "IGFBP3",
        "IGFBP7", "GLB1", "LMNB1", "MMP1", "MMP3", "MMP10", "IL6",
        "IL8", "CXCL1", "CCL2", "SERPINE1", "PAI1", "PLAU", "TIMP1",
        "TIMP2", "TGFB1", "TNFRSF10B", "FAS", "FASLG", "BTG2", "ZFP36L1",
        "DDB2", "POLK", "RRM2B", "TRIAP1", "RPS27L",
    ],
    "autophagy": [
        # KEGG_AUTOPHAGY + GO macroautophagy core
        "ATG3", "ATG4A", "ATG4B", "ATG4C", "ATG4D", "ATG5", "ATG7",
        "ATG10", "ATG12", "ATG13", "ATG14", "ATG16L1", "ATG16L2",
        "BECN1", "BECN2", "MAP1LC3A", "MAP1LC3B", "MAP1LC3B2",
        "MAP1LC3C", "GABARAP", "GABARAPL1", "GABARAPL2", "SQSTM1",
        "ULK1", "ULK2", "AMBRA1", "WIPI1", "WIPI2", "PIK3C3",
        "PIK3R4", "VPS11", "VPS16", "VPS18", "VPS33A", "VPS33B",
        "RAB7A", "TFEB", "TFE3", "FOXO3", "FOXO1", "DRAM1", "DRAM2",
        "BNIP3", "BNIP3L", "NBR1", "OPTN", "TAX1BP1", "CALCOCO2",
        "STX17", "VAMP8", "SNAP29", "RUBCN",
    ],
}


# ── Data loading ────────────────────────────────────────────────────


def load_expression_matrix() -> pd.DataFrame:
    """Read the GSE248823 series-matrix expression block.

    Returns a (n_probes, n_samples) DataFrame indexed by probe ID with
    GSM accessions as columns.
    """
    print(f"  reading {SERIES_MATRIX.name} ...")
    # Find the data block by line offset
    with open(SERIES_MATRIX) as f:
        lines = f.readlines()
    start = next(
        i for i, ln in enumerate(lines) if ln.startswith("!series_matrix_table_begin")
    )
    end = next(
        i for i, ln in enumerate(lines) if ln.startswith("!series_matrix_table_end")
    )
    # Header is the line right after "begin" — quoted GSM IDs separated by tabs
    data_block = "".join(lines[start + 1 : end])
    from io import StringIO

    df = pd.read_csv(
        StringIO(data_block),
        sep="\t",
        header=0,
        index_col=0,
        quotechar='"',
    )
    # Drop any all-NaN rows
    df = df.dropna(how="all")
    print(f"    -> {df.shape[0]} probes × {df.shape[1]} samples")
    return df


def load_probe_to_gene() -> dict[str, str]:
    """Parse GPL17586 → primary gene symbol per probe.

    Reads the gene_assignment column and pulls the first SYMBOL from
    blocks of the form ``ACCESSION // SYMBOL // DESC // CYTO // ENTREZ``.
    """
    print(f"  reading {PLATFORM.name} ...")
    df = pd.read_csv(PLATFORM, sep="\t", comment="#", low_memory=False)
    print(f"    -> {df.shape[0]} platform entries; columns: {list(df.columns)[:6]}...")

    probe_to_gene: dict[str, str] = {}
    n_unmapped = 0
    for probe, raw in zip(df["ID"], df["gene_assignment"].fillna("")):
        if not raw or raw == "---":
            n_unmapped += 1
            continue
        # Take the first '///'-separated block, then field [1] of '//' split.
        first_block = raw.split("///", 1)[0]
        parts = [p.strip() for p in first_block.split("//")]
        if len(parts) < 2 or not parts[1] or parts[1] == "---":
            n_unmapped += 1
            continue
        symbol = parts[1].strip()
        # Reject obvious non-symbols (RP11-, ENST-, etc.) by filtering
        # to alphanumeric leading
        if not symbol[0].isalpha():
            n_unmapped += 1
            continue
        probe_to_gene[probe] = symbol
    print(
        f"    -> mapped {len(probe_to_gene)} probes, "
        f"{n_unmapped} unmapped ({100 * n_unmapped / df.shape[0]:.1f}%)"
    )
    return probe_to_gene


def aggregate_to_genes(
    expr: pd.DataFrame, probe_to_gene: dict[str, str]
) -> pd.DataFrame:
    """Aggregate (probe × sample) → (gene × sample) by mean."""
    print("  aggregating probe-level expression to gene-level ...")
    # Keep only probes with a mapped gene
    common = expr.index.intersection(probe_to_gene.keys())
    expr = expr.loc[common].copy()
    expr["__gene__"] = [probe_to_gene[p] for p in expr.index]
    gene_expr = expr.groupby("__gene__").mean(numeric_only=True)
    print(f"    -> {gene_expr.shape[0]} unique gene symbols")
    return gene_expr


# ── ssGSEA ──────────────────────────────────────────────────────────


def run_ssgsea(
    gene_expr: pd.DataFrame, *, max_iter: int = 1000
) -> pd.DataFrame:
    """Run ssGSEA via gseapy → (sample × pathway) NES matrix.

    Pathways are the seven HallSim PathwayMapper outputs; gene sets are
    the curated lists in PATHWAY_GENE_SETS.
    """
    print("  running gseapy.ssgsea ...")
    # Coverage diagnostic per pathway
    print("    pathway gene-set coverage (intersection with platform):")
    for name, genes in PATHWAY_GENE_SETS.items():
        present = sum(1 for g in genes if g in gene_expr.index)
        print(f"      {name:<22}  {present}/{len(genes)}  "
              f"({100 * present / len(genes):.1f}%)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = gseapy.ssgsea(
            data=gene_expr,
            gene_sets=PATHWAY_GENE_SETS,
            sample_norm_method="rank",
            min_size=5,
            max_size=2000,
            permutation_num=0,  # ssGSEA proper, no permutation
            outdir=None,
            verbose=False,
        )
    nes = result.res2d.pivot_table(
        index="Term", columns="Name", values="NES"
    )
    # Order pathways canonically
    from hallsim.pathway_mapper import PATHWAY_ORDER

    nes = nes.reindex(list(PATHWAY_ORDER))
    print(f"    -> NES matrix: {nes.shape[0]} pathways × {nes.shape[1]} samples")
    return nes


# ── Δ_data computation ──────────────────────────────────────────────


def compute_deltas(nes: pd.DataFrame) -> pd.DataFrame:
    """Per-pathway delta vectors for each preprint scenario."""
    samples = list(nes.columns)
    sample_by_pos = {i: samples[i] for i in range(len(samples))}

    def grp_mean(positions):
        cols = [sample_by_pos[i] for i in positions]
        return nes[cols].mean(axis=1)

    deltas = pd.DataFrame(
        {
            "DDIS_D14_vs_D00": grp_mean(SAMPLE_COLS["ETOPOSIDE_D14"])
            - grp_mean(SAMPLE_COLS["ETOPOSIDE_D00"]),
            "OIS_D07_vs_D00": grp_mean(SAMPLE_COLS["RAS_D07"])
            - grp_mean(SAMPLE_COLS["RAS_D00"]),
            "DDIS_RAPA_vs_DDIS_D14": grp_mean(
                SAMPLE_COLS["ETOPOSIDE_RAPAMYCIN_D14"]
            )
            - grp_mean(SAMPLE_COLS["ETOPOSIDE_D14"]),
            "OIS_DMOG_vs_OIS_D07": grp_mean(SAMPLE_COLS["RAS_DMOG_D07"])
            - grp_mean(SAMPLE_COLS["RAS_D07"]),
        }
    )
    return deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save NES heatmap to data/ssgsea_nes_heatmap.png",
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()

    print("=" * 72)
    print("HallSim — ssGSEA on GSE248823 (DDIS + OIS arms)")
    print("=" * 72)

    print("\n[1/5] Load expression matrix ...")
    expr = load_expression_matrix()

    print("\n[2/5] Load platform annotation ...")
    probe_to_gene = load_probe_to_gene()

    print("\n[3/5] Aggregate probes → genes ...")
    gene_expr = aggregate_to_genes(expr, probe_to_gene)

    print("\n[4/5] Run ssGSEA ...")
    nes = run_ssgsea(gene_expr, max_iter=args.max_iter)

    print("\n[5/5] Compute deltas ...")
    deltas = compute_deltas(nes)
    print()
    print(deltas.round(3).to_string())

    OUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    deltas.to_csv(OUT_CSV)
    nes.to_csv(OUT_NES_CSV)
    print(f"\n  saved {OUT_CSV.relative_to(ROOT)}")
    print(f"  saved {OUT_NES_CSV.relative_to(ROOT)}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 5))
            im = ax.imshow(
                nes.values.astype(float),
                aspect="auto",
                cmap="RdBu_r",
                vmin=-2,
                vmax=2,
            )
            ax.set_yticks(range(len(nes.index)))
            ax.set_yticklabels(nes.index)
            ax.set_xticks(range(len(nes.columns)))
            # short sample labels
            inv = {v[0]: k for k, v in SAMPLE_COLS.items()}
            inv2 = {v[1]: k for k, v in SAMPLE_COLS.items()}
            sample_short = []
            for i, gsm in enumerate(nes.columns):
                cond = inv.get(i) or inv2.get(i) or gsm
                sample_short.append(f"{cond}\n[{i}]")
            ax.set_xticklabels(sample_short, rotation=45, ha="right",
                               fontsize=7)
            plt.colorbar(im, ax=ax, label="NES")
            ax.set_title("ssGSEA NES — GSE248823")
            plt.tight_layout()
            out = ROOT / "data" / "ssgsea_nes_heatmap.png"
            plt.savefig(out, dpi=150)
            print(f"  saved {out.relative_to(ROOT)}")
        except ImportError:
            print("  matplotlib not available; skipping plot")

    print("\n" + "=" * 72)
    print("Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
