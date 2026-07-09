# Calibration dataset — GSE248823

The transcriptomic dataset the flagship composite is calibrated and
evaluated against. Loaded by
[`demos/multi_hallmark_calibrate.py`](../demos/multi_hallmark_calibrate.py)
from `data/FibroblastsDNA_dmg_Rapamycin/`.

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

## What the flagship uses

| Composite arm | Definition (condition vs reference) | Role |
|---------------|-------------------------------------|------|
| `DDIS_vs_ctrl` | etoposide D07, D14 **vs** etoposide D00 | **fit** (the only arm in the loss) |
| `RAPA_vs_DDIS` | etoposide+rapa D07, D14 **vs** time-matched etoposide D07, D14 | held-out (rapamycin effect) |
| `RAS_vs_ctrl` | RAS D04, D07 **vs** RAS D00 | held-out (transfer to a different trigger) |

- The **RAS + DMOG** arm (4 arrays) is **not used** — DMOG is a metabolic
  perturbation outside the composite's scope.
- The two references illustrate that a fold-change denominator is a
  *choice*: `DDIS_vs_ctrl` uses a **baseline timepoint** (D00);
  `RAPA_vs_DDIS` uses a **time-matched comparator condition** (isolating
  the drug effect at each day).
- **Replicates are averaged** (mean of log2 intensities) into each
  condition *before* the fold-change, so every reporter contributes one
  measured Δ per timepoint — concordance is over **n = 6 reporters**, not
  the 2 replicates.

## Reporters

Six mechanistic observables ↔ six canonical reporter genes
([`hallsim.gene_reporters.MULTI_HALLMARK_REPORTERS`](../src/hallsim/gene_reporters.py)):
CDKN1A, DDB2, HMOX1, NFKBIA, CYCS, EIF4EBP1.

## Caveat

Two biological replicates and 2–3 timepoints per arm is thin — it
constrains rather than fully resolves the dynamics. Single-cell RNA-seq
would be the preferred modality for mechanistic inference of this kind;
this dataset is used for its accessibility, topical alignment
(gerotherapeutic modulation of senescence), and its two-arm
±intervention design.
