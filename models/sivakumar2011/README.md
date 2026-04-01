# Sivakumar et al. 2011 — Neural Stem Cell Signaling Models

**Paper:** "A systems biology approach to model neural stem cell regulation by
notch, shh, wnt, and EGF signaling pathways"

**Authors:** Sivakumar KC, Dhanesh SB, Shobana S, James J, Mundayoor S

**Journal:** OMICS: A Journal of Integrative Biology, 2011 Oct; 15(10):729-737

**DOI:** 10.1089/omi.2011.0011 | **PMID:** 21978399

---

## Overview

The paper models four signaling pathways that regulate neural stem cell (NSC)
differentiation, plus a crosstalk model that integrates them. Each pathway was
modeled as a system of ODEs with mass-action and Michaelis-Menten kinetics,
simulated in COPASI. The key finding is that **individual pathway models do not
match experimental data** — only the integrated crosstalk model reproduces
biologically realistic dynamics.

All concentrations are in arbitrary units (dimensionless). Time units are not
explicitly stated in the paper but the models reach steady state within ~100
time units.

Each pathway model also includes a drug perturbation baked into the SBML
initial conditions.

---

## Model 1: EGF Receptor Signaling (BIOMD0000000394)

**BioModels:** https://biomodels.org/BIOMD0000000397

**19 species, 10 reactions.** Compartment: cytoplasm (c2).

Models the EGFR -> Ras -> Raf-1 -> MEK1/2 -> ERK1/2 MAPK cascade, including
RKIP (Raf Kinase Inhibitory Protein) negative feedback and PP2A phosphatase
recycling.

**Drug:** Erlotinib (EGFR tyrosine kinase inhibitor), init=0.5

| Species ID | Name                        | Init |
|------------|-----------------------------|------|
| s3/s123    | EGF / EGFR                  | 5/5  |
| s124       | Ras                         | 5    |
| s23        | Raf-1 (inactive)            | 5    |
| s24        | Raf-1 (active)              | 0    |
| s25        | MEK1/2 (inactive)           | 5    |
| s26        | MEK1/2 (active)             | 0    |
| s27        | ERK1/2 (inactive)           | 5    |
| s28        | ERK1/2 (active)             | 0    |
| s29        | RKIP (inactive)             | 5    |
| s30        | RKIP (active)               | 0    |
| s21        | Akt (inactive)              | 5    |
| s22        | Akt (active)                | 0    |
| s34        | Mitogenesis/Differentiation | 1    |
| s127       | PKC                         | 2    |
| s147       | Complex(Grb2/../PLC)        | 5    |

**Key output:** ERK1/2 phosphorylation (s28) drives mitogenesis/differentiation.

---

## Model 2: Hedgehog/Shh Signaling (BIOMD0000000395)

**BioModels:** https://biomodels.org/BIOMD0000000395

**21 species, 12 reactions.** Compartments: cytoplasm (c1), nucleus (c5),
lipid raft (c4).

Models the Sonic Hedgehog pathway: Shh binds Patched (Ptch), releasing
Smoothened (Smo) inhibition. Smo activates the Costal2/Fused complex on
microtubules, which releases Cubitus interruptus (Ci/Gli) from the
Su(fu) repressor complex. Free Ci enters the nucleus, binds CBP, and
activates Hedgehog target genes.

**Drug:** SAG (Smoothened Agonist), init=0.5

| Species ID | Name                          | Init |
|------------|-------------------------------|------|
| s7         | Hedgehog (Shh)                | 5    |
| s1         | Patched                       | 5    |
| s21        | Complex(Patched/Hedgehog)     | 0    |
| s148       | Smoothened                    | 3    |
| s150       | Costal2/Fused complex         | 3    |
| s46        | Costal2/Fused/Smo complex     | 0    |
| s128       | Ci-repressor/Su(fu) complex   | 2.5  |
| s135       | Sap18                         | 2.5  |
| s70        | Cubitus interruptus (free)    | 0    |
| s71        | CBP                           | 2    |
| s158       | Complex(CBP/Cubitus)          | 0    |
| s75        | **Hedgehog target gene**      | 0    |

**Key output:** Hedgehog target gene activation (s75).

---

## Model 3: Notch Signaling (BIOMD0000000396)

**BioModels:** https://biomodels.org/BIOMD0000000396

**29 species, 16 reactions.** Compartments: cytoplasm (c1), nucleus (c5).

Models the Notch pathway: Delta/Serrate ligands bind Notch receptor, which
undergoes TACE-mediated (S2) and gamma-secretase-mediated (S3) cleavage to
release NICD (Notch Intracellular Domain). NICD enters nucleus, displaces
CoR from Su(H)/RBP-jk, recruits Mastermind and CoA, and activates E(spl)-C /
Hes-1 target genes. Includes Numb (asymmetric division), Neuralized (Delta
trafficking), Fringe (glycosylation), and Sel10/LNXp80 (NICD degradation).

**Drug:** DAPT (gamma-secretase inhibitor), init=0 in this model (but init=100
in the crosstalk model)

| Species ID | Name                               | Init |
|------------|------------------------------------|------|
| s1         | Notch                              | 5    |
| s7         | Delta                              | 5    |
| s48        | Serrate                            | 5    |
| s15        | Notch intracellular (transmembrane)| 5    |
| s19        | Notch TM (cleaved)                 | 0    |
| s24/s63    | NICD (nuclear / cytoplasmic)       | 0    |
| s26        | Su(H) / RBP-jk                    | 5    |
| s28        | CoR                                | 5    |
| s27        | Mastermind                         | 0.5  |
| s29        | CoA                                | 0.5  |
| s35        | Mastermind/Su(H)/CoA/NICD complex  | 0    |
| s75        | **E(spl)-C genes**                 | 0    |
| s25        | Numb                               | 0.64 |

**Key output:** E(spl)-C / Hes-1 gene activation (s75).

---

## Model 4: Wnt/beta-catenin Signaling (BIOMD0000000397)

**BioModels:** https://biomodels.org/BIOMD0000000397

**49 species, 29 reactions.** Compartments: cytoplasm (c1), nucleus (c3),
extracellular (c4).

The largest model. Wnt binds Frizzled + LRP5/6, activating the
Dishevelled/Beta-Arrestin/Frodo complex via CK2 and FRAT. This inhibits the
APC/Axin/GSK3-beta destruction complex. Free beta-catenin accumulates,
translocates to nucleus, and sequentially recruits TCF/Smad4 -> Bcl9 ->
Pygo -> SWI/SNF to activate Wnt target genes. Includes betaTrCP-mediated
ubiquitination and Siah-1/Ebi degradation pathways.

**Drug:** 6-bromoindirubin-3'-oxime (BIO, GSK3-beta inhibitor), init=0.5

| Species ID | Name                           | Init |
|------------|--------------------------------|------|
| s239       | Wnt (extracellular)            | 5    |
| s1         | Frizzled                       | 3    |
| s28        | LRP5/6                         | 3    |
| s107       | Dishevelled/Beta-Arrestin/Frodo| 3    |
| s36        | beta-catenin (cytoplasmic)     | 5    |
| s232       | beta-catenin (nuclear)         | 0    |
| s121       | APC/Axin/PP2A (destruction cx) | 4    |
| s252       | Full phosphorylation complex   | 5    |
| s174       | TCF/Smad4                      | 4    |
| s170       | Bcl9                           | 2    |
| s171       | Pygo                           | 2    |
| s172       | CBP                            | 2    |
| s173       | SWI/SNF                        | 2    |
| s195       | **Wnt Target Genes**           | 0    |
| s37        | GSK3-beta (free)               | 0    |
| s61        | betaTrCP                       | 2    |
| s304       | BIO (GSK3 inhibitor)           | 0.5  |

**Key output:** Wnt Target Gene activation (s195).

---

## Model 5: Crosstalk / Integration (BIOMD0000000398)

**BioModels:** https://biomodels.org/BIOMD0000000398

**22 species, 12 reactions.** Single compartment (default).

This is the integration model that wires the four pathways together. It
contains simplified representatives from each pathway and models their
cross-regulation. **This is the model the paper says is needed to match
experimental data.**

**Drugs (all at high dose):** DAPT=100, Erlotinib=100, BIO=100, SAG=0

| Species ID | Name                       | Pathway  | Init |
|------------|----------------------------|----------|------|
| s57        | Notch                      | Notch    | 5    |
| s53        | NICD                       | Notch    | 0    |
| s58        | Notch TM                   | Notch    | 0    |
| s68        | RBP-jk                     | Notch    | 5    |
| s72        | Complex NICD-RBP           | Notch    | 0    |
| s73        | **Hes-1**                  | Notch    | 0    |
| s81        | Shh                        | Shh      | 5    |
| s83        | Ptch1                      | Shh      | 5    |
| s85        | Complex Shh-Ptch1          | Shh      | 0    |
| s88        | Smo (Smoothened)           | Shh      | 5    |
| s96        | EGF                        | EGF      | 5    |
| s98        | EGFR                       | EGF      | 5    |
| s100       | Complex EGF-EGFR           | EGF      | 0    |
| s107       | Wnt                        | Wnt      | 5    |
| s109       | Frizzled                   | Wnt      | 5    |
| s111       | Complex Wnt-Frizzled       | Wnt      | 0    |
| s122       | Dishevelled                | Wnt      | 5    |
| s124       | FRAT-CK2                   | Wnt      | 5    |
| s135       | Complex Dsh-FRAT-CK2       | Wnt      | 0    |
| s142       | GSK3-beta                  | Wnt      | 5    |
| s144       | Beta-catenin               | Wnt      | 5    |
| s146       | Complex GSK3B-Bcatenin     | Wnt      | 0    |

**Key outputs:** Hes-1 (s73, Notch), Wnt-Frizzled (s111), GSK3B-Bcatenin (s146)

**Known cross-talk mechanisms in this model:**
- Notch <-> Wnt: GSK3-beta phosphorylates NICD; Dishevelled interacts with
  Notch signaling
- Shh <-> Wnt: Shh and Wnt synergize through GSK3-beta regulation
- EGF <-> Notch: EGFR signaling modulates Notch cleavage
- Shh <-> Notch: Hes-1 represses Shh targets; Gli activates Notch

---

## Reproducing the paper's Figure 1

Each sub-model corresponds to one panel of Figure 1 in the paper:
- **Fig 1A:** EGF pathway (BIOMD0000000394)
- **Fig 1B:** Shh pathway (BIOMD0000000395)
- **Fig 1C:** Wnt pathway (BIOMD0000000397)
- **Fig 1D:** Notch pathway (BIOMD0000000396)

The crosstalk model (BIOMD0000000398) integrates all four.

## Running the models

```bash
cd hallsim
source .venv_hallsim/bin/activate
python demos/run_all_sivakumar.py      # simulate all 5, generate plot
python demos/validate_sivakumar.py     # run hallsim validation layer
```

## sbmltoodejax note

The crosstalk model (398) triggers a bug in sbmltoodejax where MathML
namespace prefix `no` is emitted as bare Python (`no.sqrt(...)`) instead of
mapping to `jax.numpy`. This is patched in `hallsim/sbml_import.py`.
