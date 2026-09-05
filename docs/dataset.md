# Calibration dataset — GSE248823

The dataset the multi-hallmark **demo** is calibrated and evaluated against.
Loaded by
[`demos/multi_hallmark_calibrate.py`](../demos/multi_hallmark_calibrate.py)
from `data/FibroblastsDNA_dmg_Rapamycin/`.

The demo exists to exercise the calibration machinery — held-out arms,
fold-change loss, gene reporters, gradients through a stiff multi-group solve
— on real published models and a real public dataset. It is not a senescence
result, and its concordance score is not a claim about HallSim; see the caveat
at the end and P0.14 in [known-problems.md](known-problems.md).

## Source

- **Accession:** [GSE248823](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE248823) (GEO).
- **Study:** Tighanimine et al. 2024, *Nature Metabolism* 6:323–342 —
  "A homoeostatic switch causing glycerol-3-phosphate and
  phosphoethanolamine accumulation triggers senescence by rewiring lipid
  metabolism." DOI 10.1038/s42255-023-00972-y.
- **Platform:** GPL17586 — Affymetrix Human Transcriptome Array 2.0.
  A **microarray**: values are normalized **log2 intensities**, so the
  only calibration-comparable quantity is a fold-change (a same-gene ratio
  cancels the probe-specific scale). See
  [`docs/calibration.md`](calibration.md).
- **Cells:** WI-38 human fibroblasts (*Homo sapiens*).
- **Size:** 20 arrays = 10 conditions × 2 biological replicates.

## Arms

Timepoints are in **days**. The two senescence triggers run on different
clocks: etoposide sampled at D00 / D07 / D14, RAS at D00 / D04 / D07.

| Arm | Trigger | Timepoints | Arrays |
|-----|---------|-----------|--------|
| Etoposide (DDIS) | DNA-damage-induced senescence | D00, D07, D14 | 6 |
| Etoposide + rapamycin | + mTOR inhibitor | D07, D14 | 4 |
| RAS (OIS) | oncogene (HRAS)-induced senescence | D00, D04, D07 | 6 |
| RAS + DMOG | + DMOG (prolyl-hydroxylase inhibitor / hypoxia-mimetic) | D04, D07 | 4 |

## Sample → column mapping

Series-matrix column indices (0-based, after the ID column), matching
`SAMPLE_POSITION_GROUPS` in the demo. Each group is the two biological
replicates for one condition.

| Columns | Sample title | Arm · timepoint |
|---------|--------------|-----------------|
| 0, 1   | `WI38_…_ETOPOSIDE_D00`            | Etoposide · D00 |
| 2, 3   | `WI38_…_ETOPOSIDE_D07`            | Etoposide · D07 |
| 4, 5   | `WI38_…_ETOPOSIDE_D14`            | Etoposide · D14 |
| 6, 7   | `WI38_…_ETOPOSIDE_RAPAMYCIN_D07`  | Etoposide+rapa · D07 |
| 8, 9   | `WI38_…_ETOPOSIDE_RAPAMYCIN_D14`  | Etoposide+rapa · D14 |
| 10, 11 | `WI38_…_RAS_D00`                  | RAS · D00 |
| 12, 13 | `WI38_…_RAS_D04`                  | RAS · D04 |
| 14, 15 | `WI38_…_RAS_D07`                  | RAS · D07 |
| 16, 17 | `WI38_…_RAS_DMOG_D04`             | RAS+DMOG · D04 *(unused)* |
| 18, 19 | `WI38_…_RAS_DMOG_D07`             | RAS+DMOG · D07 *(unused)* |

## What the multi-hallmark demo uses

| Composite arm | Definition (condition vs reference) | Role |
|---------------|-------------------------------------|------|
| `DDIS_vs_ctrl` | etoposide D07, D14 **vs** etoposide D00 | **fit** (the only arm in the loss) |
| `RAPA_vs_ctrl` | etoposide+rapa D07, D14 **vs** etoposide D00 | held-out (rapamycin effect) |
| `RAS_vs_ctrl` | RAS D04, D07 **vs** RAS D00 | held-out (transfer to a different trigger) |

- The **RAS + DMOG** arm (4 arrays) is **not used** — DMOG is a metabolic
  perturbation outside the composite's scope.
- Every arm is normalised **within itself**, to its own day 0
  (`normalization="baseline"`), so the model is asked to reproduce `X_t / X_0`
  along each arm rather than a cross-arm contrast. The rapamycin culture's day 0
  *is* etoposide D00 — rapamycin is not added until day 2 — so that arm
  normalises to `ETOPOSIDE_D00` too. The drug contrast (rapa vs no-rapa) is
  recovered afterwards by differencing the two within-arm curves.
- **Replicates are averaged** (mean of log2 intensities) into each
  condition *before* the fold-change, so every reporter contributes one
  measured Δ per timepoint — concordance is over **n = 5 reporters**, not
  the 2 replicates.

## Reporters

Five mechanistic observables ↔ five canonical reporter genes
([`hallsim.gene_reporters.MULTI_HALLMARK_REPORTERS`](../src/hallsim/gene_reporters.py)):

<!-- reporters:start — checked against MULTI_HALLMARK_REPORTERS by
     tests/unit/test_gene_reporters.py; edit the code, then this list. -->

| Gene | Store path |
|---|---|
| `CDKN1A` | `dp14/CDKN1A` |
| `GLB1` | `dp14/SA_beta_gal` |
| `BNIP3` | `dp14/FoxO3a` |
| `DDB2` | `gz06/x` |
| `MDM2` | `gz06/y0` |
| `FAS` | `k14/CD95_level` |

<!-- reporters:end -->

Per-reporter summaries and rationale are in
[calibration.md](calibration.md#gene-reporters).

## Caveat

**There is no time-matched untreated arm.** Etoposide is sampled D00/D07/D14
and RAS D00/D04/D07, each arm normalised to its own day 0; no untreated culture
is measured at D07 or D14. Nothing in this dataset can distinguish a trigger-
driven trajectory from one a culture would have followed anyway, which is the
same missing measurement that let DallePezze 2014's spontaneous senescence go
unnoticed — see [senescence-model-rebuild.md](senescence-model-rebuild.md) §6.
A dataset with a time-matched untreated arm is a prerequisite for any
concordance number that claims the perturbation caused the change.

**Candidates located (2026-08-29), none yet ingested or verified:**

| accession | contrast | why it matters |
|---|---|---|
| GSE63577 + GSE77682 (Marthandan) | MRC-5 and HFF, young (PD32/PD16) **vs 20 Gy at 120 h** | MRC-5 at 20 Gy is DallePezze's *own* cell line and *own* dose — the closest external check on DP14 that exists |
| GSE63577 | MRC-5 PD32 **vs** PD72 | replicative senescence, a second route to the same phenotype |
| GSE222400 | WI-38, doxorubicin, D0/1/2/3/4/6/8/16 vs untreated control | same cell line as GSE248823, eight timepoints, and a control arm |

Two reality checks land immediately and both cut against DP14: at 120 h after
20 Gy, MRC-5 `GLB1` moves **+0.16 log2** where the model has SA-β-gal rising
about tenfold, and `CDKN2A` moves **−0.94**, down.

**Verify before use.** In GSE222400 the per-sample files are already
differential tables, and the `Untreated-control_1` file is *not* a self-contrast
of zero — it reports `MKI67` −5.51 at `padj` 0.000. Whatever those log2FCs are
referenced to, it is not what the filename says, and the contrast structure has
to be established before any of it is fitted against.

Two biological replicates and 2–3 timepoints per arm is thin — it
constrains rather than fully resolves the dynamics. Single-cell RNA-seq
would be the preferred modality for mechanistic inference of this kind;
this dataset is used for its accessibility, topical alignment
(gerotherapeutic modulation of senescence), and its two-arm
±intervention design.
