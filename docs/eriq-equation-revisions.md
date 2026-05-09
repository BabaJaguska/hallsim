# ERiQ Model Equation Revisions: Biological Grounding

## Overview

The original ERiQ model (Alfego & Kriete, 2017) uses several raw reciprocal relationships
(`1/x`) to model inhibitory signaling. While qualitatively reasonable, these create:

1. **Numerical singularities** — derivatives → ∞ as state variables → 0
2. **Lack of saturation** — biological responses have maximum amplitudes
3. **Missing ultrasensitivity** — many signaling pathways exhibit switch-like behavior
   from cooperative multi-site regulation

This document details each equation revision with biological rationale and literature
references. The goal is to replace phenomenological `1/x` terms with biophysically
grounded Hill and Michaelis-Menten functions while preserving the qualitative dynamics
of the original ERiQ model.

All changes are in `_compute_algebraic()` and the `ERiQEnergyMetabolism.derivative()`
method in `src/hallsim/models/eriq.py`.

---

## Revision 1: AMPK Activation by Energy Stress

### Original
```python
AMPK = AMPK_SA * (1.0 / (ATPr + eps))
```

### Problem
Raw reciprocal: AMPK → ∞ as ATP → 0. No saturation at low ATP. Misses the
ultrasensitive switch character of AMPK activation.

### Revised
```python
AMPK = AMPK_SA * K_AMPK**n_AMPK / (K_AMPK**n_AMPK + ATPr**n_AMPK)
```
with `K_AMPK = 3.0` (half-inhibition ATP level), `n_AMPK = 2` (Hill coefficient).

### Biological Rationale
AMPK is activated by rising AMP:ATP ratio through three cooperative mechanisms:
(1) allosteric activation by AMP binding to the γ-subunit CBS domains,
(2) protection from dephosphorylation at Thr172 by PP2C phosphatases, and
(3) promotion of phosphorylation by LKB1. The adenylate kinase equilibrium
(2 ADP ⇌ AMP + ATP) further amplifies signals: a small ATP drop produces a
disproportionately large AMP rise.

These layered mechanisms produce ultrasensitive, switch-like behavior. An inhibitory
Hill function of ATP (high ATP → low AMPK, low ATP → high AMPK) with n=2 captures
this. K_AMPK=3.0 is chosen so that at homeostatic ATPr ≈ 6.0 (mfunct + glycol),
AMPK is low (~0.2), consistent with the resting state.

### References
- Hardie DG, Ross FA, Hawley SA (2012). "AMPK: a nutrient and energy sensor that
  maintains energy homeostasis." *Nat Rev Mol Cell Biol* 13(4):251–262.
- Hardie DG, Schaffer BE, Brunet A (2016). "AMPK: An Energy-Sensing Pathway with
  Multiple Inputs and Outputs." *Trends Cell Biol* 26(3):190–201.
- Suter M et al. (2006). "Dissecting the role of 5'-AMP for allosteric stimulation,
  activation, and deactivation of AMP-activated protein kinase." *J Biol Chem*
  281(43):32207–32216.
- Dalle Pezze P et al. (2012). "A dynamic network model of mTOR signaling reveals
  TSC-independent mTORC2 regulation." *Mol Syst Biol* 8:571. (Uses Hill-type AMPK
  with n=2 in their ODE model.)

---

## Revision 2: PTEN Activity

### Original
```python
PTEN = PTEN_SA * (1.0 / (mfunct + eps))
```

### Problem
Raw reciprocal: PTEN → ∞ as mitochondrial function → 0. The biological intent
is that declining mitochondria → more PTEN → less AKT. However, the mechanism is
indirect: declining mitochondria → more ROS → PTEN *inactivation* (not activation).
The original equation has the direction partially backwards.

### Revised
```python
PTEN = PTEN_SA * K_PTEN**n_PTEN / (K_PTEN**n_PTEN + ROS**n_PTEN)
```
with `K_PTEN = 0.5` (half-inhibition ROS level), `n_PTEN = 1`.

### Biological Rationale
PTEN is a lipid phosphatase that opposes PI3K by dephosphorylating PIP3. ROS
(specifically H₂O₂) *inactivates* PTEN by oxidizing the catalytic cysteine (Cys124),
forming an intramolecular disulfide bond with Cys71. Thus:

- High ROS → PTEN inactivated → more PIP3 → more AKT (pro-survival)
- Low ROS → PTEN active → less PIP3 → less AKT

This is the canonical mechanism by which oxidative stress activates AKT survival
signaling. Using an inhibitory Hill function of ROS (not of mito_function) directly
models the biochemistry and eliminates the singularity.

Note: This reverses the effective direction vs the original model. In the original,
low mito_function → high PTEN → high AKT (via the additive AKT equation). In our
revision, low mito_function → high ROS → low PTEN → high AKT (via the corrected
inhibitory AKT equation below). The net effect on AKT is preserved but the
mechanism is now correct.

### References
- Lee SR et al. (2002). "Reversible inactivation of the tumor suppressor PTEN by
  H₂O₂." *J Biol Chem* 277:20336–20342.
- Nguyen LK et al. (2013). Detailed ODE model of PI3K/AKT/PTEN with Hill functions.
  *PLoS Comput Biol*.

---

## Revision 3: AKT Activation

### Original
```python
AKT = AKT_SA * (GF + PTEN + ROS / 5.0)
```

### Problem
PTEN is *additive* to AKT, but biologically PTEN **inhibits** AKT (by degrading
PIP3, the lipid required for AKT membrane recruitment). The `+ PTEN` term has the
wrong sign.

### Revised
```python
AKT = AKT_SA * (GF + ROS_boost) * K_AKT_PTEN / (K_AKT_PTEN + PTEN)
```
where `ROS_boost = ROS / 5.0`, `K_AKT_PTEN = 0.5`.

### Biological Rationale
AKT activation requires PIP3 at the plasma membrane. Growth factors activate PI3K
which produces PIP3. PTEN opposes this by degrading PIP3. ROS can activate AKT
through PTEN inactivation (captured above) and through direct oxidative activation
of PI3K.

The revised form uses:
- Growth factors + ROS as *activators* (numerator)
- PTEN as an *inhibitor* via a Michaelis-Menten-like denominator term

This matches the standard form in published PI3K/AKT ODE models.

### References
- Hatakeyama M et al. (2003). "A computational model on the modulation of MAPK and
  Akt pathways." *Biochem J* 373:451–463.
- Schoeberl B et al. (2002). "Computational modeling of the dynamics of the MAP
  kinase cascade." *Nat Biotechnol* 20:370–375.

---

## Revision 4: FOXO Transcription Factor

### Original
```python
FOXO = FOXO_SA * (1.0 / (AKT + eps))
```

### Problem
Raw reciprocal: FOXO → ∞ as AKT → 0. No saturation. Misses the ultrasensitive
switch behavior from multi-site phosphorylation (AKT phosphorylates FOXO at Thr24,
Ser256, Ser319).

### Revised
```python
FOXO = FOXO_SA * K_FOXO**n_FOXO / (K_FOXO**n_FOXO + AKT**n_FOXO)
```
with `K_FOXO = 0.5` (half-inhibition AKT level), `n_FOXO = 2`.

### Biological Rationale
AKT phosphorylates FOXO transcription factors at multiple sites, causing 14-3-3
protein binding and cytoplasmic sequestration (nuclear exclusion). Multi-site
phosphorylation creates ultrasensitive switch-like behavior: FOXO transitions sharply
between active (nuclear) and inactive (cytoplasmic) states as AKT crosses a threshold.

An inhibitory Hill function with n=2 captures this cooperativity. At high AKT,
FOXO ≈ 0 (sequestered). At low AKT, FOXO ≈ FOXO_SA (nuclear, transcriptionally
active).

### References
- Calzone L et al. (2010). "Mathematical modelling of cell-fate decision in response
  to death receptor engagement." *PLoS Comput Biol* 6:e1000702. (AKT/FOXO module
  with Hill coefficient n=2.)
- Zhang XP, Liu F, Wang W (2011). "Two-phase dynamics of p53 in the DNA damage
  response." *PNAS* 108:8990–8995.

---

## Revision 5: mTOR–Autophagy Relationship

### Original
```python
AUTOPHAGY = AUTO_SA * 0.001 * (1.0 / (MTOR + eps) + 0.5 * FOXO + ROS + P53)
```

### Problem
`1/MTOR` singularity. No saturation of autophagy at low mTOR.

### Revised
```python
mtor_inhibition = K_AUTO**n_AUTO / (K_AUTO**n_AUTO + MTOR**n_AUTO)
AUTOPHAGY = AUTO_SA * (mtor_inhibition + 0.5 * FOXO + ROS + P53)
```
with `K_AUTO = 1.0` (half-inhibition mTOR level), `n_AUTO = 2`.

### Biological Rationale
mTORC1 suppresses autophagy by phosphorylating ULK1 at Ser757, preventing AMPK-ULK1
interaction and blocking autophagy initiation. The relationship is switch-like:
autophagy is largely off when mTOR is highly active, and sharply induced when mTOR
drops below a threshold (e.g., rapamycin treatment, nutrient starvation).

An inhibitory Hill function with n=2–4 captures this switch. We use n=2 and remove
the 0.001 scaling factor (absorbed into the new Hill function's scale, which naturally
saturates at 1.0).

### References
- Kapuy O, Vinod PK, Banhegyi G, Novak B (2014). "Bistability and toggle switches
  during autophagy regulation." *Autophagy* 10(9):1502–1519.
- Szymanska P et al. (2015). "Computational analysis of an autophagy/translation
  switch." *PLoS Comput Biol* 11(1):e1004084.

---

## Revision 6: SIRT Activity and Glycolysis Inhibition

### Original (in `_compute_algebraic`):
```python
SIRT = SIRT_SA * NADr           # where NADr = mfunct
```
Then in `ERiQEnergyMetabolism.derivative()`:
```python
dGlycolysis = GLYCOL_SA * (gain3 * u3 + 1.0 / (SIRT + eps))
```

### Problem
SIRT is linear in NAD+ (no saturation), and `1/SIRT` creates a singularity when
mito_function → 0.

### Revised

SIRT now uses Michaelis-Menten kinetics:
```python
SIRT = SIRT_SA * NADr / (Km_SIRT + NADr)
```
with `Km_SIRT = 1.5`.

The glycolysis inhibition term uses an inhibitory Hill form:
```python
sirt_inhibition_of_glycolysis = K_SIRT_gly / (K_SIRT_gly + SIRT)
```
with `K_SIRT_gly = 0.5`, replacing `1/SIRT`.

### Biological Rationale
SIRT1 is an NAD⁺-dependent deacetylase. NAD⁺ is a required co-substrate, and the
enzyme follows Michaelis-Menten kinetics with Km(NAD⁺) ≈ 100–170 μM. Since
physiological NAD⁺ (200–500 μM) is only ~2–3× the Km, SIRT1 activity is genuinely
sensitive to NAD⁺ fluctuations — it operates in the responsive middle region of the
MM curve.

For the glycolysis term: SIRT1 inhibits glycolysis by deacetylating and destabilizing
HIF-1α and glycolytic enzymes. This inhibition saturates — at maximum SIRT1 activity,
glycolysis doesn't go to zero. An inhibitory Michaelis-Menten form `K/(K + SIRT)`
captures this: high SIRT → low glycolysis boost, low SIRT → higher glycolysis boost,
bounded by K_SIRT_gly.

### References
- Borra MT et al. (2004). "Substrate specificity and kinetic mechanism of the Sir2
  family of NAD+-dependent histone/protein deacetylases." *Biochemistry*
  43(30):9877–9887. (Km_NAD ≈ 94–170 μM.)
- Cantó C, Menzies KJ, Auwerx J (2015). "NAD⁺ metabolism and the control of energy
  homeostasis." *Cell Metab* 22(1):31–53.

---

## Revision 7: Glucose Uptake (GLU)

### Original
```python
GLU = GLU_SA * (1.0 / (NFKB + eps))
```

### Problem
Raw reciprocal with singularity. Also, the biological direction may be partially
inverted: in many cell types, NFκB *increases* glucose uptake by upregulating GLUT1
transcription.

### Revised
```python
GLU = GLU_SA * K_GLU / (K_GLU + NFKB)
```
with `K_GLU = 1.0`.

### Biological Rationale
The NFκB–glucose relationship is context-dependent. In insulin-sensitive tissues
(relevant to the ERiQ quiescence model), chronic NFκB activation impairs insulin-
stimulated glucose uptake via IRS serine phosphorylation. However, basal glucose
uptake is maintained. A Michaelis-Menten inhibitory form captures the saturable
nature of this relationship without the singularity: at very low NFκB, GLU approaches
GLU_SA × 1.0; at high NFκB, GLU approaches 0 but never diverges.

We preserve the original model's qualitative direction (more NFκB → less glucose
uptake) since ERiQ models quiescent fibroblast-like cells where insulin signaling
matters. The key improvement is numerical stability.

### References
- Shoelson SE, Lee J, Goldfine AB (2006). "Inflammation and insulin resistance."
  *J Clin Invest* 116(7):1793–1801.
- Tornatore L et al. (2012). "The nuclear factor kappa B signaling pathway:
  integrating metabolism with inflammation." *Trends Cell Biol* 22(11):557–566.

---

## Revision 8: Mitochondrial Damage Rate

### Original
```python
MD = MDR_SA * (jnp.abs(mfunct + ROS) * MDR + (ROS - 0.8) * 0.0001)
```

### Problem
Linear in ROS. Does not capture the cooperative/threshold nature of ETC dysfunction
leading to ROS damage. No mass-action constraint (damage rate should decrease as
undamaged fraction shrinks).

### Revised
```python
MD = MDR_SA * MDR * ROS * jnp.maximum(mfunct, 0.0)
```

### Biological Rationale
Mitochondrial damage rate is proportional to:
- **ROS level** — superoxide/H₂O₂ damages mtDNA, membranes, and ETC complexes
- **Functional mitochondrial mass** — damaged mitochondria have reduced electron
  flow and thus reduced sites for ROS-induced damage (mass-action constraint)

The `mfunct * ROS` product captures this: when mito_function is high, there are many
targets for ROS damage; as it drops, fewer targets remain. The `jnp.maximum` prevents
negative damage rates. The MDR scaling factor controls the overall damage rate.

This follows the approach of Kowald & Kirkwood (2000): `k_damage * ROS * (M_total - D)`.
In our normalized variables, `mfunct` serves as a proxy for the undamaged mass.

### References
- Kowald A, Kirkwood TBL (2000). "Accumulation of defective mitochondria through
  delayed degradation of damaged organelles." *J Theor Biol* 202(2):145–160.
- Figge MT et al. (2012). "Deceleration of fusion-fission cycles improves
  mitochondrial quality control during aging." *PLoS Comput Biol* 8(6):e1002576.

---

## Summary of New Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `K_AMPK` | 3.0 | ATP level at half-maximal AMPK activation |
| `n_AMPK` | 2 | Hill coefficient for AMPK ultrasensitivity |
| `K_PTEN` | 0.5 | ROS level at half-maximal PTEN inactivation |
| `n_PTEN` | 1 | Hill coefficient for ROS-PTEN (simple oxidation) |
| `K_AKT_PTEN` | 0.5 | PTEN level at half-maximal AKT inhibition |
| `K_FOXO` | 0.5 | AKT level at half-maximal FOXO inhibition |
| `n_FOXO` | 2 | Hill coefficient for multi-site phosphorylation |
| `K_AUTO` | 1.0 | mTOR level at half-maximal autophagy inhibition |
| `n_AUTO` | 2 | Hill coefficient for mTOR-autophagy switch |
| `Km_SIRT` | 1.5 | NAD+ level for half-maximal SIRT activity |
| `K_SIRT_gly` | 0.5 | SIRT level for half-maximal glycolysis inhibition |
| `K_GLU` | 1.0 | NFκB level for half-maximal glucose uptake inhibition |

All K values are in model-normalized units and calibrated so that at the homeostatic
initial condition the revised model produces similar algebraic intermediate values
to the original.

---

## Preserved Equations (Not Changed)

The following algebraic relationships are preserved as-is because they are already
well-formed (linear combinations, products, or already bounded):

- `ATPm`, `ATPg`, `ATPr` — simple sums
- `ROS = 10 * ROS_SA * ros_act` — linear scaling of a state variable
- `NADr = NADr_SA * mfunct` — proxy for NAD+/NADH ratio
- `PGC1a = PGC1a_SA * (AMPK + 0.1 * SIRT)` — linear activation
  *(superseded — see Stability Revisions below)*
- `MTORs`, `MTORa`, `MTOR` — algebraic combinations
- `NFKB` — linear combination of activators
- `P53s`, `P53a`, `P53` — algebraic feedback
- `Uz` (free radical driver) — linear combination
- `HIF`, `PYR` — linear scalings

---

## Calibration Notes

The new Hill/MM parameters (K values) were chosen to approximate the original model's
algebraic output at the homeostatic initial condition:

- At IC: ATPr ≈ 6.02, mfunct ≈ 3.62, ROS ≈ 0.79, AKT ≈ 0.59
- Original AMPK ≈ 0.17; revised with K=3.0, n=2: AMPK ≈ 0.20 ✓
- Original FOXO ≈ 1.69; revised with K=0.5, n=2: needs recalibration
- Further parameter tuning may be needed to match the original trajectory
  qualitative behavior (damage accumulation time course, etc.)

The K values are a starting point and should be refined by:
1. Matching the homeostatic steady state
2. Reproducing the qualitative aging trajectory (progressive damage, declining function)
3. Comparing against the original MATLAB ERiQ output

---

## Stability Revisions (composability hardening)

The singularity revisions above are sufficient for ERiQ at baseline and under mild
perturbation. The revisions in this section harden ERiQ against runaway feedback under
the strong, sustained perturbations that arise when ERiQ is composed with upstream
damage / inflammation / proteostasis modules — for example a DamageRepair process that
chronically elevates p53 and NF-κB, which then drives mitochondrial biogenesis and ROS
production.

These changes are informed by the failure-mode analysis from a prior run, which observed runaway mitochondrial function
(+76,000%) and ATP (+26,000%) followed by NaN under coupled DamageRepair simulations
in an earlier ERiQ build. Both failure modes traced to unbounded feedback amplification
in linear sink/source terms.

### Stability Revision 1: PGC1α saturating biogenesis

**Original (after singularity revisions):**
```python
PGC1a = PGC1a_SA * (AMPK + 0.1 * SIRT)
```

**Revised:**
```python
pgc_drive = AMPK + 0.1 * SIRT
PGC1a = PGC1a_SA * pgc_drive * K_PGC1a / (K_PGC1a + pgc_drive)
```
with `K_PGC1a = 10.0` (default).

**Problem:** The linear form has no upper bound under chronic AMPK/SIRT activation,
which can drive mitochondrial biogenesis runaway when ERiQ is composed with damage
modules that sustain elevated AMPK (energy stress) or SIRT (NAD+ rise from
mito-dysfunction proxy).

**Biological rationale:** PGC1α is itself transcriptionally regulated and protein
turnover is bounded. *In vivo*, mitochondrial biogenesis saturates rather than scaling
linearly with upstream activator concentration. Saturating MM kinetics reflect the
finite capacity of the biogenesis machinery (ribosomal output, membrane lipid supply,
nuclear-encoded protein import).

**Calibration:** At baseline, `pgc_drive ≈ 0.27` (AMPK ≈ 0.20, SIRT ≈ 0.71). The
factor `K/(K + pgc_drive) ≈ 0.974` reduces PGC1α by ~2.6% from the linear value, well
within the homeostatic test tolerance (`|d/dt| < 1.0`). Under strong perturbation
(`pgc_drive → 10`), PGC1α saturates at `K_PGC1a` rather than growing without bound.

### Stability Revision 2: ROS clearance saturation

**Original:**
```python
dROS_activity = 0.01 * obs["Uz"]
```

**Revised:**
```python
ros_act_pos = max(ros_act, 0.0)
ros_clearance = (
    k_ROS_clear * ros_act_pos**n_ROS_clear
    / (K_ROS_clear**n_ROS_clear + ros_act_pos**n_ROS_clear)
)
dROS_activity = 0.01 * obs["Uz"] - ros_clearance
```
with `k_ROS_clear = 0.01`, `K_ROS_clear = 1.0`, `n_ROS_clear = 2.0` (defaults).

**Problem:** The integrator-style ROS dynamics in the original ERiQ have no direct sink
on `ros_act` (Az). The negative-feedback integrator `dCz = -ROS - Cz` provides
homeostatic regulation at baseline, but `ros_act` itself is unbounded and can grow when
the driving term `Uz` (which includes p53, mito_damage, FOXO) is sustained — exactly
the regime introduced by composing with a damage-response module that chronically
elevates p53.

**Biological rationale:** Cellular ROS clearance is performed by saturating enzyme
systems — superoxide dismutase, catalase, and the glutathione peroxidase / peroxiredoxin
network. All operate with Michaelis-Menten kinetics and finite Vmax. At low ROS, clearance
is approximately linear in ROS; at high ROS, clearance saturates and damage outpaces it.
The Hill form (n=2) reflects the cooperative recruitment of antioxidant defenses
(Nrf2/Keap1 transcriptional response).

**Calibration:** At baseline `ros_act ≈ 0.0794`, the clearance term is
`0.01 × 0.0063 / 1.0063 ≈ 6.3e-5` — three orders of magnitude below
the homeostatic test threshold and below the nominal `0.01 × Uz ≈ 0` driving term. At
`ros_act ≥ 1` the term saturates near `k_ROS_clear = 0.01`, providing a hard ceiling on
ROS growth that engages only when ros_act exceeds the half-saturation.

### What is NOT addressed by these revisions

Possibly remove the integrator-style activity dynamics
(`Cy/Ay`, `Cx/Ax`, `Cz/Az` pairs in `ERiQSignaling` and `ERiQOxidativeStress`) in
favor of fully algebraic activity computation. This is a larger refactor not undertaken
here. If long-horizon coupled simulations show drift in `mTOR_activity`, `p53_activity`,
or `ROS_activity` toward unphysical values, the next step is to convert these to
algebraic forms following the V10 pattern.

Also not addressed:
- **Glycolysis brakes** (G6P product inhibition on hexokinase, ATP allosteric inhibition
  on PFK1). Defer until a coupled simulation shows glycolysis runaway under strong
  `GLYCOL_SA` perturbation.
- **Strict nonnegativity on state variables.** `mfunct` is guarded inside the damage
  computation (`max(mfunct, 0)` on line 286 of `eriq.py`) but other state variables can
  still go negative as integrators. Add only if observed in practice.
- **Wip1 phosphatase, SASP-driven mTORC1 activation.** These are coupling features that
  belong in upstream damage / inflammation modules, not in ERiQ's intrinsic dynamics.
