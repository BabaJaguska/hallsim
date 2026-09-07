# Rejection registry

Every deposit screened for a HallSim slot and rejected, with the class of
failure and one line of reason. Recorded here rather than in prose so it can be
counted: the distribution of *why* candidate models fail is a finding about the
field, and it is what the framework's claim rests on — that it rejects fast,
for stated reasons, before a reviewer is spent.

It also stops repeat screening. Four separate sessions re-derived the same
verdict on the same NF-κB deposits because the reasoning lived in ten
`docs/review-*.md` files and nothing indexed it.

Parsed by `hallsim.rejections`; the format is checked by
`tests/unit/test_rejections.py`. One row per line, pipes inside a cell are not
allowed, and `class` must come from the closed vocabulary below.

## Failure classes

| class | meaning |
|---|---|
| `consumes-not-emits` | takes in the quantity it was wanted to supply. Mentioning is not emitting. |
| `wrong-formalism` | qualitative/Boolean, a drawn map with no rate laws, or written for stochastic simulation and read as an ODE |
| `no-dynamic-range` | the readout is saturated across the range the coupling edge would drive |
| `not-identifiable` | its parameters cannot be estimated from the data the benchmark has |
| `cell-type-mismatch` | cell type, stimulus or receptor complement does not transfer to the target |
| `numerically-unusable` | does not import, does not solve, or has no usable gradient |
| `no-provenance` | parameters or citations do not support what the model asserts |

A deposit usually fails several ways. `class` is the one that decided it —
the cheapest check that would have been sufficient on its own.

## Rejections

| id | model | slot | class | reason | evidence |
|---|---|---|---|---|---|
| BIOMD0000000230 | Ihekwaba2004 NF-κB sensitivity | inflammation | consumes-not-emits | emits no SASP effector; NF-κB is an input to the arm we needed it to drive | docs/review-ihekwaba2004-wetlab.md |
| BIOMD0000000794 | Benary2019 NF-κB via β-TrCP | inflammation | consumes-not-emits | clean intake, but emits only feedback inhibitors, never an effector | session 2026-09-05 |
| BIOMD0000000151 | Singh2006 IL-6 signal transduction | inflammation | consumes-not-emits | produces IL6 receptor complexes only; free IL6 is an input | session 2026-09-06 |
| MODEL1712240002 | Bekkar2018 IL1B secretion | inflammation | consumes-not-emits | 15 of 16 transitions have no inputs; one DNF over 13 args, and IL1B is one of them | session 2026-09-06 |
| BIOMD0000000560 | Hui2016 articular cartilage | inflammation | wrong-formalism | deposited for Gillespie (substance unit `item`, 65/65 integral ICs); read as an ODE it deletes the ALK1 arm | docs/review-hui2016-maths.md |
| MODEL2302140001 | Wu2010 rheumatoid arthritis map | inflammation | wrong-formalism | 254 reactions, 0 kinetic laws, 0 parameters — a drawn map | docs/design-pathway-maps.md |
| MODEL2406110001 | Mechanotransduction and inflammation | inflammation | wrong-formalism | emits the whole SASP panel as arrows; 324 reactions, 0 rate laws | docs/design-pathway-maps.md |
| MODEL2604100001 | Optotransduction pathway | inflammation | wrong-formalism | 305 reactions and 0 kinetic laws; emits IL1A, IL1B, IL6 and IL8 as drawn arrows | session 2026-09-06 |
| MODEL2307090001 | Aghakhani2022 breast CAF | inflammation | wrong-formalism | SBML-qual: 465 qualitative species, 403 transitions, empty core model | session 2026-09-06 |
| MODEL2212220001 | Aghakhani2022 RA synovial fibroblast | inflammation | wrong-formalism | SBML-qual: 359 qualitative species, 345 transitions, no rate law anywhere; a human fibroblast model in the wrong formalism | session 2026-09-06 |
| MODEL2307190001 | Singh2023 RA synovial fibroblast | inflammation | wrong-formalism | SBML-qual: 321 qualitative species, 233 transitions; parses to an empty core model | session 2026-09-06 |
| MODEL2408030001 | Zerrouk2024 RA multicellular | inflammation | wrong-formalism | SBML-qual: 0 core species and 0 reactions, so the produced-species screen read it as a negative until `no-reactions` existed | session 2026-09-06 |
| MODEL2307180001 | Zerrouk2023 M1 synovial macrophage | inflammation | wrong-formalism | SBML-qual; also a macrophage rather than the fibroblast the benchmark measures | session 2026-09-06 |
| MODEL2307180002 | Zerrouk2023 M2 synovial macrophage | inflammation | wrong-formalism | SBML-qual; the M2 companion to MODEL2307180001, same objection | session 2026-09-06 |
| MODEL2312140001 | Sizek2023 MiDAS senescence | senescence | numerically-unusable | deposit contains no .xml or .sbml file at all | session 2026-09-06 |
| BIOMD0000000534 | Dwivedi2014 IL-6 QSP, healthy volunteer | inflammation | no-dynamic-range | pSTAT3 spends 87.9% of its achievable response by 21x ksynthIL6Gut; measured SASP IL-6 induction is 3.3-37x, so the whole plausible range saturates it | docs/review-dwivedi2014-wetlab.md |
| BIOMD0000000535 | Dwivedi2014 anti-IL6 antibody arm | inflammation | not-identifiable | 3 benchmark timepoints give rank 3 against 19 module parameters | docs/review-dwivedi2014-maths.md |
| BIOMD0000000536 | Dwivedi2014 sgp130 arm | inflammation | numerically-unusable | administers zero drug after import; Dose is dead | docs/review-dwivedi2014-maths.md |
| BIOMD0000000537 | Dwivedi2014 anti-IL6R antibody arm | inflammation | not-identifiable | same module, same rank-3 limit | docs/review-dwivedi2014-maths.md |
| BIOMD0000000873 | Soni2018 IL-6 M2 macrophage | inflammation | numerically-unusable | integrates and is tolerance-insensitive, but the gradient is non-finite | session 2026-09-06 |
| MODEL1911130003 | Boer1985 macrophage-T cell | inflammation | no-provenance | 0% ontology coverage, no declared time unit, population-level not cell-level | session 2026-09-06 |
| BIOMD0000000524 | Kallenberger2014 CD95L apoptosis | apoptosis | cell-type-mismatch | CD95L-induced extrinsic death in HeLa; irradiated fibroblasts die by the intrinsic route, which the deposit does not contain | docs/design-fate-architecture.md |
