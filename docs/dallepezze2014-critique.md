# The senescence model of Dalle Pezze et al. (2014) has no young state, and its intervention conclusion does not follow

Working basis for a critique paper. Every number below is either quoted from
the source article, read from the authors' own deposited files, or measured
against the deposit — provenance for each is in §8.

**Subject.** Dalle Pezze P, Nelson G, Otten EG, Korolchuk VI, Kirkwood TBL,
von Zglinicki T, Shanley DP. *Dynamic modelling of pathways to cellular
senescence reveals strategies for targeted interventions.* PLoS Comput Biol
2014;10(8):e1003728, and the deposit BIOMD0000000582.

---

## 1. The claim

The paper concludes that senescence, once underway, can be delayed but not
reversed:

> "Dynamic sensitivity analysis of the model showed the network stabilised at a
> new late state of cellular senescence… suggesting an unsatisfactory outcome
> for treatments aiming to delay or reverse cellular senescence at late time
> points. Combinatorial targeted interventions are therefore possible… but in
> the cases identified here, are only capable of delaying senescence onset."

That conclusion is drawn from a model in which **senescence is the only
possible outcome by construction**, from any initial condition, with or
without the damage that is supposed to cause it. The model has one attractor;
it is the senescent state; and the published "young" basal state is not a
state the model can occupy. The stability analysis offered as evidence for the
conclusion is a measurement of that single attractor, and would have returned
the same answer had no irradiation been modelled at all.

Three things need to be separated, because they have different standing:

| claim | standing |
|---|---|
| ROS and mTOR inhibition reduce senescence markers over ~2 weeks | **Supported.** Predicted in silico, confirmed in vitro (Figs. 2–4). Stands. |
| The model reproduces 20 Gy-irradiated MRC5 over 21 days | **Supported.** χ² reproduces to 0.08%. Stands, on that window. |
| The network *transitions* to a new late state, which is why intervention fails | **Not supported.** No transition exists in the model; there is one state and everything is already in its basin. |

---

## 2. Reproduction — the object under analysis is the right one

Before criticising, three independent checks that the deposit is the model the
paper analysed:

- **Byte identity.** The repository copy is md5 `359928d2ba6f8a698ae5577126e7946c`,
  identical to the live BioModels record.
- **Fit reproduces.** Running the deposit as published and scoring all 14
  observables at all 127 data points against the reported standard deviations
  gives **χ² = 70.3696** against the paper's stated **70.4278** — 0.08%.
- **Stability analysis reproduces.** trace(J) at the fixed point is
  **−9632.69** against the paper's Table 1 "Sum" of **−9617.50** (0.16%), and
  17 of the 19 tabulated Lyapunov exponents match the Jacobian eigenvalues,
  three to four significant figures.

Whatever is wrong is in the paper, not in the deposition or in our handling.

---

## 3. Finding A — the published basal state is not a state, and no parameterisation can make it one

This needs no simulation. Two model species have a two-term balance involving
four rate constants, so *any* rest state of *any* parameterisation leaving
those four alone must satisfy:

```
SA_beta_gal*  = (0.0701140 / 0.154821)  · ROS*  = 0.45287 · ROS*
DNA_damage*   = (0.118874 / 0.325725)   · ROS*  = 0.36495 · ROS*
```

The published initial condition sets `ROS = 10`, `SA_beta_gal = 0.81`,
`DNA_damage = 1`. Those three numbers are **mutually incompatible by factors
of 5.6 and 3.6**: `ROS = 10` demands `SA_beta_gal = 4.53` and
`DNA_damage = 3.65`; `SA_beta_gal = 0.81` demands `ROS = 1.79`.

The consequence is stronger than "the initial condition is off equilibrium."
It is that **no adjustment of the other 37 constants can make the published
basal state a rest state.** A 41-parameter scan (each constant × 0.1 and × 10,
Newton to the unirradiated fixed point, scored as relative distance from the
published basal values) drops the score from 13.66 at the published parameters
to a best of 1.18, and never to zero.

Measured at the published state, the unirradiated field is moving hard:

| species | y₀ | dy/dt at t=0 | τ = y/ẏ |
|---|---|---|---|
| Mitophagy | 10 | +67 384 | 1.5e-4 d |
| mTORC1_pS2448 | 10 | −2 882.6 | 3.5e-3 d |
| Mito_membr_pot_new | 12.12 | −3 384.3 | 3.6e-3 d |
| ROS | 10 | +22.89 | 0.44 d |
| DNA_damage | 1 | +0.863 | 1.16 d |
| SA_beta_gal | 0.81 | +0.576 | 1.41 d |

‖f(y₀)‖₂ = 67 593 with the irradiation route deleted. **Most of the senescence
signal in this model is the initial condition relaxing**, and it relaxes with
the irradiation reaction physically removed from the SBML.

The origin of the inconsistency is visible in the numbers themselves: nineteen
of twenty-three species are set to exactly 10.00, two to 25.00, and the only
three unconventional values (0.81, 1.00, 12.12) are exactly the three readouts
whose day-0 intensity was not normalised. The basal state is a normalisation
convention, not a solution.

**The paper does not claim otherwise.** Methods say the stress variables "were
set to reflect this initial basal level." No equilibration step is described,
and none was required by the fitting procedure, which pinned all 23 initial
conditions by hand (`pwAddX(..., 'fix', ...)`). The defect is not that the
authors asserted something false here — it is that the abstract's "new late
state" requires an old state to leave, and there is none.

---

## 4. Finding B — one attractor, and it is senescence

Monostability, established two independent ways on the conservation leaf
through y₀:

- **512 Newton seeds** on the leaf (orthonormal null-space projection,
  positive orthant): 332 converged, to **exactly one** non-negative fixed
  point.
- **64 random initial conditions**, integrated 200 days: all 64 land on the
  same point, maximum endpoint spread **1.7e-05**, `SA_beta_gal = 9.0315` for
  every one.

Jacobian spectrum at the fixed point: 0 eigenvalues with Re > 1e-8, 17
negative, 6 numerically zero (five conserved moieties plus an inert sink). No
saddle, no separatrix, nothing for a dose to push the system across.

That fixed point **is** the paper's late-senescence state:

| species | y₀ | y* | ratio |
|---|---|---|---|
| SA_beta_gal | 0.81 | 9.0315 | 11.15 |
| DNA_damage | 1 | 7.2781 | 7.28 |
| ROS | 10 | 19.943 | 1.99 |
| Mito_mass_old | 0 | 8.8607 | — |
| Mito_mass_new | 1 | 0.5449 | 0.54 |

Dosed and undosed trajectories converge to it identically:
‖y_dosed(400) − y_nodose(400)‖₂ = 1.8e-05, indistinguishable to four decimals
by day 100. Across **six decades of dose** (impulse 0 → 3207 γH2A.X foci) the
day-400 endpoint is identical to six significant figures.

And the young state is not merely a non-equilibrium — it is locally
**unstable**. Re λmax = **+0.1599 /day** at the published initial condition,
against −0.0730 /day at the fixed point. The undosed field stays locally
expanding for the first 4.5 days. Closing the ratchet loop on the model's own
quasi-steady chain gives loop gain **G = 1.509** at the young state and 0.652
at the senescent one: supercritical at young, subcritical at senescent.

**The abstract's transition is not a transition.** Senescence is the only
outcome of any initial condition. The dose selects the path, not the
destination.

---

## 5. Finding C — the dose is a brake, and the dose-response is inverted

Structural ablation isolates the causal route without argument: with
`mito_dysfunction = 0`, the dosed-minus-undosed difference at day 14 is
**exactly zero** (1e-10 to 1e-12) for ROS, SA_beta_gal, Mitophagy and both
mito-mass pools. The sole path is

```
Irradiation → DNA_damage → CDKN1A → mito_dysfunction (Mito_mass_new → old)
```

The mitochondrial-mass arm is a **positive feedback loop** — more new mass →
higher ψm(new) → faster AMPK-pT172 dephosphorylation → less AMPK-pT172 → less
mitophagy production and more mTORC1-pS2448 → more biogenesis → more new mass.
`CDKN1A` is that loop's only brake, and irradiation is the only thing that
raises CDKN1A.

So removing the dose removes the brake:

| | dosed | undosed |
|---|---|---|
| Mito_mass_new at day 5 | 1.43 | **4.01** |
| cumulative mito_dysfunction flux to day 14 | 9.93 | **11.74** |
| Mitophagy at day 14 | 12.60 | **17.31** |
| cumulative ROS production from the *new* ψm pool to day 14 | 635.8 | **1056.6** |

The 21-day dose-response is monotone **protective** across four decades:

| k₁₉ | γH2A.X foci at 5 min | SA_beta_gal (21 d) | Mito_mass_old (21 d) |
|---|---|---|---|
| 0 | 1 | 9.810 | 9.907 |
| 923.8 | 4.2 | 9.701 | 9.849 |
| **9237.72 (published)** | **33.1** | **9.140** | **9.331** |
| 92 377 | 322 | 7.331 | 7.871 |

More irradiation, less senescence. The loop gain is large enough to flip the
sign of the dose's effect on its own input: with the loop cut, dosed-minus-
undosed `DNA_damage` at day 14 is **+0.336**; intact, it is **−0.753**.

This is not a solver artefact. The 21-day endpoint is converged to eight
digits at rtol=1e-9, and loose (1e-3) versus tight (1e-11) tolerances differ
by at most 0.7% on any state.

---

## 6. Finding D — the evidence base contains no control

From the authors' own deposited fitting file (`pcbi.1003728.s028`, PottersWheel
Data Format 3.0.12, last saved by Piero Dalle Pezze 2014-04-01):

- **127 data points**, 14 observables, 18 timepoints over 21 days
- **`Stimulus = 1` on every single row.** Every measurement is irradiated.
- Methods: *"0 day timepoints were unirradiated."* Day 0 is the only
  unirradiated measurement in the study.
- **39 free parameters**, with all 23 initial conditions pinned by hand.

Noise per observable, median CV: ROS 10%, SA-β-gal 10%, γH2A.X 15%, JNK 20%,
CDKN1A 21%, mTOR 32%, ψm 35%, Akt 38%, CDKN1B 43%, FoxO3a-pS253 55%, FoxO3a
total 56%, AMPK 67%, Mito mass 80%, Mitophagy 90%. Mitophagy at day 1 is
**10.46 ± 17.39** — standard deviation larger than the mean.

**No unirradiated simulation is shown anywhere in the paper**, and no
unirradiated time course was measured. This is the mechanism by which the
defect survived peer review: the model's spontaneous senescence is invisible
without a control, because the data it is fitted to is also going to
senescence. The fit is good. The comparison that would have exposed the
problem was never made, by the authors or the referees.

---

## 7. Finding E — two stated conclusions rest on artefacts

**E1. "AMPK-driven biogenesis played a very minor role (k34)."** Reactions 30
and 31 are the same reaction, `Mito_mass_turnover → Mito_mass_new`, with rate
law `k · Mito_mass_turnover · mTORC1_pS2448` in both. `AMPK_pT172` appears
nowhere in reaction 31, despite the parameter being named
`mito_biogenesis_by_AMPK_pT172`. The pair is exactly structurally
non-identifiable:

```
max |f(k33+δ, k34−δ) − f(k33, k34)| over 64 random states = 4.4e-16
d(Mito_mass_old @21d)/dk33 = d(…)/dk34 = 1030.44   (ratio 1.000000000)
```

Fitted values are k33 = 0.0133620, k34 = 5.89e-05 — k34 is 0.44% of the
pair's sum, which is what an optimiser does to a redundant coordinate. The
conclusion is arithmetically correct and biologically empty: it reports that a
duplicate coordinate was driven to the floor. The same error is in the
authors' PottersWheel source, so it precedes deposition.

The 7-round MOTA identifiability protocol did not catch it for a structural
reason: k33 was frozen in round 4 and k34 assessed in round 5, and a
sequential freeze-then-assess protocol makes any exactly collinear pair look
identifiable once one member is frozen.

**E2. "The 19 computed Lyapunov exponents were all negative."** The system has
five exact conservation laws plus an inert sink, so **at least six exponents
are exactly zero**. The reduced dimension is 17–18, not 19; the count was
hand-set in Methods. Two of the tabulated exponents (−0.0136, −0.0052) have no
counterpart in the Jacobian spectrum — they are conserved directions whose
true exponent is zero, mis-estimated by finite-time averaging. The averaging
window ("start averaging after t, 21; overall time, 50") also sits inside the
transient: the slowest mode has a 13.7-day time constant, so at t = 21 the
trajectory is still ~1.5 time constants from the fixed point.

The *conclusion* — asymptotic stability — is correct, and we verified it
independently. The calculation offered as its proof is not.

---

## 8. What this invalidates, precisely

The paper's intervention conclusion is a three-step inference:

1. the network stabilises at a new late senescent state;
2. at that state, global parameter sensitivity falls, so signalling is noisy
   and interventions are weak;
3. therefore interventions can delay senescence onset but not reverse it.

Step 1 is not a property of the damage. It is the unique attractor of the
fitted vector field, reached from anywhere, dose or no dose. Step 2 is
generic: parameter sensitivity necessarily falls as any monostable system
approaches its attractor — it is a restatement of "the system is near
equilibrium," not a discovery about senescent cells. Step 3 therefore
transfers a structural property of a monostable ODE onto biology, and the
structural property was fixed by the choice of parameters, never tested
against a control, and cannot be tested within this model because no young
state exists in it.

A sharper way to put it: **the model cannot represent a cell that does not
senesce.** A claim about the limits of intervention, made with an instrument
that has no non-senescent outcome available, is not a result about
intervention.

A secondary inconsistency at the endpoint is worth reporting alongside. At day
21 the model gives a 26-fold fall in ψm per unit mitochondrion (data: 32-fold)
— near-complete depolarisation — in cells the paper's own Figure S3 shows
remain viable, and in a model whose energy sensor simultaneously reports
better-than-young energy status. Mass, potential and energy do not close on
one another.

---

## 9. What survives, and what a corrected analysis would look like

**Survives.** The ROS and mTOR intervention predictions, confirmed in vitro,
as transient statements about the first two weeks — which is what they were
tested against. The 0–21-day fit to irradiated MRC5. The convergence claim
itself.

**A young state is reachable, but not by removing the dose.** The 41-parameter
scan's best single change is instructive: raising `AMPK_T172_phos` tenfold
(identically, cutting `AMPK_pT172_dephos_by_Mito_membr_pot_new` tenfold) moves
the unirradiated rest state to SA-β-gal 1.303, DNA_damage 1.050, ROS 2.877,
Mito_mass_old 0.105 — a young cell. That is the positive-feedback loop broken
at its hinge. **A control arm in this model is a parameter change, not an
unexposed cell**, and that is itself a substantive claim the paper could have
made and did not.

**What would settle it experimentally.** An unirradiated MRC5 time course over
the same 21 days, on the same 14 readouts. If the untreated cells do not
approach the late-senescence state, the model is falsified as written and the
parameters must be refit with a control arm in the objective. If they do, the
"late state" is a property of the culture, not of the irradiation, and the
intervention conclusion needs restating in those terms. Either outcome is
publishable and neither was available in 2014, because the measurement was
never made.

**Recommended re-analysis.** Refit with (i) an unirradiated arm in the
objective, (ii) the basal state as a *fitted* rest state rather than a pinned
normalisation, (iii) k33/k34 merged to their identifiable sum, and (iv)
identifiability assessed jointly rather than by sequential freezing. Report
whether the young state survives as an attractor. That is the paper this
critique motivates.

---

## 10. Provenance

Primary sources under `data/dallepezze2014/` (see its `PROVENANCE.md`):
the authors' fitting dataset (PLOS s028), the PottersWheel-exported SBML
(s026), Text S1 (s025), and a text extraction of the article.

Measurements against the deposit are from the review reports
`docs/review-dallepezze2014-maths.md` and
`docs/review-dallepezze2014-physics.md` (gitignored raw evidence), produced by
the panel in `.claude/agents/`, and reproduced independently here for the fit
statistics, the rest residual, the dose-response scan and the data-file
audit. Screening and residual figures come from `hallsim.diagnostics` and
`hallsim.intake`; `intake.triage_sbml(582)` returns
‖f(y₀)‖/‖y₀‖ = **1.29e+03**, three orders of magnitude outside the other two
models in the same composite.
