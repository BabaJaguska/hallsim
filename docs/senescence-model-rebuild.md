# Repairing the senescence model — accepted findings and rebuild plan

An outside reviewer audited DallePezze 2014 (BIOMD0000000582) independently of
this repository and returned a verdict. This document records which parts of
that verdict we accept, on what evidence, which parts we qualify or reject, and
what the repair actually is.

It sits between two existing documents and does not repeat them:

- [dallepezze2014-critique.md](dallepezze2014-critique.md) — the case against
  the paper, with the measurements.
- [known-problems.md](known-problems.md) — P0.14, P0.16 and P1.13 are the open
  framework defects this model exposed.

---

## 1. Status of the outside review

Every quantitative claim in the review reproduces a measurement we had already
made independently. Where the numbers overlap they agree; where they differ,
they differ because the review stopped earlier.

| review claim | verdict | our evidence |
|---|---|---|
| Irradiation is a 5-minute acute pulse, then the field is autonomous again | accepted | `assignmentRule` on `Irradiation`, cutoff `0.003472` d = 5.0 min, `DNA_damaged_by_irradiation` = 9237.72 |
| ‖ẏ‖ = 67 593 /day at the published IC, mitophagy 67 384 /day | accepted, exact match | critique §3; `intake.triage_sbml(582)` → rest_residual 1.29e+03 |
| Untreated 21 d: DNA damage 1.00→7.39, ROS 10→19.34, old mito 0→9.91, ψm(new) 12.12→3.22, SA-β-gal 0.81→9.81 | accepted, exact match | critique §5, dose-response row `k₁₉ = 0`: SA-β-gal 9.810, Mito_mass_old 9.907 |
| Dosed and undosed converge (5.37e-10 relative, day 300) | accepted | ‖Δ‖ = 1.8e-05 at day 400; identical to 6 s.f. across six decades of dose |
| A pulse from the equilibrium returns to the equilibrium | accepted | same fact as monostability: 512 Newton seeds and 64 random ICs find one attractor |
| The common equilibrium is not a healthy state | accepted | it *is* the paper's late-senescence state: SA-β-gal 9.03, DNA damage 7.28, ROS 19.94, old mito 8.86 |
| No distinct irradiation-induced attractor was demonstrated | accepted | there is one attractor on the conservation leaf and everything is in its basin |
| Irreversibility was not demonstrated | accepted | no saddle, no separatrix, nothing for a dose to cross |
| "Intervention fails when started late" was never tested | accepted — **new to us**, see §3 | verified against the paper's own Methods |
| Low parameter sensitivity ≠ poor controllability | accepted, and the review states it better than we did | critique §8 step 2 |
| The model cannot prove interventions only delay senescence | accepted | critique §8 |

---

## 2. Accepted without qualification

**The forcing term is right; the resting state is not.** Irradiation enters as
an acute 5-minute input to a damage pool and then stops. That is a defensible
representation of an acute exposure, and the reviewer is correct to say so
before criticising. The defect is downstream of it.

**The published basal state is not a state of the model.** Accepted, and our
version is stronger: this is not an initialisation error that a refit could
absorb. Two species have a two-term balance, so any rest state of any
parameterisation leaving those four constants alone must satisfy
`SA_beta_gal = 0.45287·ROS` and `DNA_damage = 0.36495·ROS`. The published
`ROS = 10` demands SA-β-gal 4.53 against the published 0.81. A 41-parameter
scan (each × 0.1 and × 10) never reaches zero residual.

**The untreated model senesces on its own, and the dose is not what carries
it.** Accepted. This is the core of both audits and it is the finding that
disqualifies the paper's intervention conclusion.

**The four conclusions that do not follow.** Accepted as listed: no distinct
attractor, no irreversibility, no late-intervention test, and sensitivity is
not controllability. The control-theoretic framing — that ∂x/∂k and
finite-amplitude reachability under toxicity constraints are different
questions — is a better statement of our §8 step 2 and we adopt it.

**What survives.** Accepted, and identical to our §9: the 0–21 day fit to
irradiated MRC-5 (χ² reproduces to 0.08%), the in vitro-confirmed direction of
ROS and mTOR inhibition over the first two weeks, and the reduced fusion/fission
observation, which is an experimental result independent of the model.

---

## 3. Accepted and new to us — the intervention was never started late

Our critique attacked the inference from low sensitivity to weak control. It
did not check whether the *experiment* behind that claim varied treatment start
time. It did not, and this is verified from the paper's own text:

- SOD/CAT were applied "throughout the time course", fed three times a week
  from immediately post-irradiation; ψm was read at days 15, 18, 21.
- Torin1 likewise, with mitochondrial mass read at days 15, 18, 21.
- The paper's own summary — "the earliest time points were most effective,
  displaying a gradual loss of treatment efficacy" — is a statement about
  **readout time under continuous treatment**, not about treatment start time.

The in silico arm cannot express a late start at all. mTOR inhibition was
simulated by *lowering the initial protein levels* of mTORC1 and Akt ("since no
turnover was included for the species mTORC1 and Akt, it was sufficient to
decrease their initial protein levels"), and ROS inhibition by adding a
scavenger species present from t = 0. Both are initial-condition changes. There
is no `u_drug(t)` in the published model, so "start the drug on day 10" is not
a simulation the deposit can run.

The claim that late intervention fails therefore rests on a design that never
varied the one variable it is about. Any successor model must carry drugs as
time-dependent inputs with a start time, a washout, and a dose — not as
initial conditions.

---

## 4. Accepted structural diagnosis — what is missing

Accepted as listed. What the flagship composite already supplies is recorded
here so the plan does not rebuild it:

| missing module | review is right that dp14 lacks it | already in the composite | still needed |
|---|---|---|---|
| p16INK4a–CDK4/6–Rb–E2F arrest switch | yes — dp14 has p21/p27 and no Rb axis | nothing | **all of it.** This is the load-bearing gap |
| DDR: ATM/ATR, p53–MDM2, pulses | partly — dp14 has one linear damage pool | GZ06 p53–Mdm2 supplies pulses on `gz06/x` | repairable vs persistent lesion split |
| SASP / NF-κB / p38 / IL-6 / IL-8 / TGF-β | yes — the authors flag the abstraction themselves | Ihekwaba NF-κB module (`nfkb/`) | p38, the interleukins, autocrine feedback, cGAS–STING |
| Mitophagy as a flux, not a state | yes | nothing | flux formulation, lysosomal clearance, PINK1/Parkin |
| Cell-cycle output (E2F, EdU, clonogenic recovery) | yes — no proliferation readout exists | nothing | **all of it.** Markers alone cannot establish arrest |
| Population heterogeneity | yes — one deterministic average cell | nothing | queued in [roadmap.md](roadmap.md) |

The two entries marked **all of it** are the same gap seen twice: the model has
no representation of the cell cycle, so it has neither a mechanism for
committed arrest nor an observable that would show arrest was escaped. That is
why a senescence claim made with it cannot be checked.

---

## 5. Qualified or rejected

**5.1 "Require f(x_H, θ, c₀) = 0 for a healthy state" — qualified, and this one
matters.** Equilibration is state-dependent, not a blanket requirement.
Equilibrate a state only where its unperturbed biology is a stable fixed point
(p53–Mdm2, NF-κB, mTOR — equilibrating removes an arbitrary-IC transient). Do
**not** equilibrate states with a one-way slow drift or ratchet (SA-β-gal
accumulation, new→old mitochondrial mass, replicative age): their biological
baseline is a transient young state, and running them to steady state destroys
the transition the data measures. Measured burn, 2026-07-25: a 400-day
equilibration of this composite drove the young control SA-β-gal 0.8 → 9.0 and
erased the D00→D14 separation GSE248823 measures.

The correct requirement is a **timescale-split baseline** — fast signalling at
rest, slow ratchet states held at a young IC — with the split decided by
running the unperturbed model and classifying each state (settles to a bounded
value → equilibrate; drifts monotonically toward a distant attractor → hold),
not by literature. The review's own alternative, "or stable homeostatic cycle",
points the same way but is not the operative clause in its prescription.

**5.2 The stability number quoted is the wrong one.** The review reports
"largest Jacobian eigenvalue real part ≈ 6.3e-13, numerically zero; none
positive" and concludes the equilibrium is "locally non-expanding". That
eigenvalue is a conserved direction, not a dynamical mode — the system has five
conservation laws plus an inert sink, so at least six eigenvalues are exactly
zero. On the conservation leaf the attractor is genuinely asymptotically
stable, max Re λ = **−0.0730 /day**. The review's limitations section notices
the conserved directions but the headline number does not reflect it.

We had the same trap in our own tools (P0.16), now fixed. Measured on the
deposit: the raw spectrum over 23 states leads with **+4.9e-14**, which is the
number the review quotes; the leaf spectrum is 17 modes leading with
**−0.072969**. Same point, same Jacobian — the difference is only whether the
six conserved directions are counted as dynamics. Every stability statement in
the successor model must be made on the projected leaf.

**5.3 "Do not represent irradiation as a permanent parameter change" —
accepted as a design principle, rejected as a description of dp14.** dp14 does
not do this; it uses an acute pulse, which is why it has no memory of the
insult. The inversion is worth stating plainly: in *this* model a permanent
parameter change is the only thing that produces a young state at all. The best
single change found by the 41-parameter scan — `AMPK_T172_phos × 10`, which
breaks the mitochondrial positive-feedback loop at its hinge — gives an
unirradiated rest state of SA-β-gal 1.30, DNA damage 1.05, ROS 2.88. A control
arm in dp14 today is a *different model*, not an unexposed cell. That is the
diagnosis, not the fix.

**5.4 The proposed damage-state equations are a shape, not a model.** The
`Ḋ_r / Ḋ_p / Ṁ_D / Ṙ` sketch is the right shape — separate the acute physical
insult from persistent lesions — and Phase 3 below adopts it. As written it has
no arrest variable, no coupling to the cell cycle, and an unspecified `q(D_r)`.
It is not something to transcribe. Successor components come from a citable,
re-downloadable source (BioModels / CellML / JWS / a paper's supplement), which
is the repository rule.

---

## 6. What the outside review does not contain

The review's verdict is a subset of ours. These findings are not in it and are
not superseded by it:

1. **The dose is a brake and the dose–response is inverted.** With
   `mito_dysfunction = 0` the dosed-minus-undosed difference at day 14 is
   exactly zero, so the sole causal route is
   `Irradiation → DNA_damage → CDKN1A → mito_dysfunction`. CDKN1A is the only
   brake on a positive feedback loop, so removing the dose removes the brake.
   Across four decades of dose, 21-day SA-β-gal falls monotonically
   (9.810 → 7.331). More irradiation, less senescence. This is the single most
   damaging result and the review does not have it.
2. **The basal state is structurally impossible**, not merely off-equilibrium
   (§2). No parameterisation fixes it.
3. **The evidence base contains no control.** The authors' deposited fitting
   file has 127 rows and `Stimulus = 1` on every one. Day 0 is the only
   unirradiated measurement in the study. No unirradiated simulation appears
   anywhere in the paper. This is the mechanism by which the defect survived
   review.
4. **k33/k34 are the same reaction.** Identical rate law
   `k·Mito_mass_turnover·mTORC1_pS2448`; field invariant under `(k33+δ, k34−δ)`
   to 4.4e-16. `k34` is named `mito_biogenesis_by_AMPK_pT172` and never reads
   AMPK. Figure 6A's conclusion is an arbitrary split of one coordinate.
5. **The 19 Lyapunov exponents cannot all be negative.** At least six are
   exactly zero by conservation; two tabulated values have no counterpart in
   the Jacobian spectrum. The conclusion (asymptotic stability) is right; the
   calculation offered as its proof is not.

---

## 7. Rebuild plan

The recommendation is not to repair dp14 into a senescence model. dp14 cannot
represent a cell that does not senesce, and the part of it that is well
constrained — the mTOR/AMPK/Akt/FoxO3a and mitochondrial-turnover axis, fitted
to 14 observables over 21 days — is not the part that decides senescence.
**Keep dp14 as the nutrient-sensing / mitochondrial module. Build the arrest
and commitment axis new, and let it own the senescence claim.**

### Phase 0 — framework prerequisites (blockers, no biology)

These gate every measurement the later phases depend on. All are already in
[known-problems.md](known-problems.md).

| item | why it blocks |
|---|---|
| **P0.16, second half** — detect real eigenvalue crossings, not only Hopf pairs | a bistable switch is born at a saddle-node, which is a *real* crossing; `hopf_scan` looks for complex pairs and would report nothing at the fold Phase 2 exists to find. The projected solve (first half) is done — `equilibrium`, `spectrum` and `hopf_scan` take `laws=` and read stability on the leaf |
| **P1.13** — collinearity pass over rate laws at `Process` construction | catches k33/k34-class duplicates before a fit, from structure alone |
| **new: spontaneous-endpoint screen** | see below |

**The new screen is the generalisable fix for this entire failure.** Extend
`intake.triage_sbml` / `diagnostics.screen_composite` with an unperturbed
control run: integrate with every perturbation input zeroed over the
experiment window, and report what fraction of the perturbed excursion the
perturbation actually explains. dp14 would return ~0% — and would have returned
it at import, before any composite was built. This belongs in the framework and
not in a demo or a document: it must fire for the next model as well, whoever
imports it.

### Phase 1 — repairs to dp14 that stand on their own

1. Merge `k33`/`k34` into their identifiable sum. Structural, needs no data,
   removes the Figure-6A artefact.
2. Refit the basal state as a **fitted** rest state instead of 23 hand-pinned
   ICs, via `CalibrationProblem` with a rest residual over the *fast* states
   only (§5.1). The slow ratchet states stay at their young IC.
3. Report identifiability jointly, not by sequential freezing — the protocol
   that let an exactly collinear pair look identifiable.
4. Re-run `intake.triage_sbml` and the review panel. Acceptance: `rest_residual`
   drops by orders of magnitude and `rest_tau` on `Mitophagy` rises from 12.8 s
   to something above the save interval.

Expected outcome: the fit survives and the model is still monostable. That is
the point — Phase 1 makes dp14 a defensible mitochondrial module and shows that
no amount of repair inside it produces a young state.

### Phase 2 — the arrest switch (the actual new model)

The p16INK4a–CDK4/6–Rb–E2F axis, as a bistable switch with a proliferating
branch and an arrested branch.

- Source it, do not invent it. Run `hallsim.discovery.search_for_model` for
  restriction-point / Rb–E2F switch models; the Yao et al. 2008 bistable Rb–E2F
  switch is the primary candidate to look for, and whatever is found goes
  through `triage_sbml` and the three-agent panel before composition.
- Requirement: **two stable states separated by a separatrix**, confirmed on
  the projected leaf with the Phase-0 bifurcation tools, not asserted.
- Add a slow p16 / chromatin state coupled to Rb–E2F. This is what supplies
  history dependence.
- `models/bistable_latch.py` exists and is phenomenological. Use it only if no
  sourced mechanistic switch survives triage, and label the result
  phenomenological in every number it produces. A latch bolted on to make a
  trajectory persist is how the 2026-07-25 equilibration artefact nearly got
  institutionalised.
- Ship the cell-cycle observables with it: E2F activity, Rb phosphorylation,
  and a proliferation readout that maps to EdU incorporation. Without these
  there is no way to distinguish arrest from marker accumulation.

### Phase 3 — damage that persists

Adopt the review's split, with the arrest coupling it omits:

- repairable lesions `D_r` — fast repair;
- persistent / telomere-associated lesions `D_p` — slow or no repair, fed by
  `D_r` and by ROS;
- mitochondrial damage `M_D` as its own state.

`D_p` is the mechanism that carries the memory of a 5-minute pulse for 21 days
without changing a parameter. Route `D_p` to ATM → p53 (GZ06 already supplies
the pulses) → p21, and to the Phase-2 switch. Two damage pools also let the
model distinguish transient arrest from committed arrest, which one linear pool
cannot.

### Phase 4 — SASP

Extend the existing `nfkb/` module rather than adding a parallel one: p38, IL-6
and IL-8, TGF-β, autocrine feedback, and cGAS–STING for the radiation route.
Keep arrest, mitochondrial dysfunction and secretion **separable** — a
senomorphic that suppresses secretion without restoring proliferation is a
distinct outcome, and the model has to be able to represent it. Paracrine
propagation waits on the multi-cell roadmap item.

### Phase 5 — population

After Phase 2, never before. A population model built over a monostable
single-cell model averages the same wrong answer with more machinery. Once
there are two basins, the queued Gillespie / multi-cell work makes the fractions
(proliferating / transiently arrested / senescent / dead) the observable, which
is what a 20 Gy culture actually is.

### Phase 6 — data, including our own gap

Adopted from the review, in priority order: paired sham and irradiated time
courses on the same readouts; **randomised treatment-initiation times** with
washout arms and equal-cumulative-exposure arms; escape-from-arrest readouts
(EdU, clonogenic regrowth after washout) alongside the marker panel; joint
fitting of sham and treated arms with whole conditions held out.

**Our own calibration set has the same blind spot.** GSE248823 has no
time-matched untreated arm: etoposide is sampled D00/D07/D14 and RAS
D00/D04/D07, every arm normalises to its own day 0, and there is no untreated
D07 or D14. We therefore cannot falsify spontaneous senescence against our
current data either — the same measurement is missing that was missing in 2014.
Finding or generating a dataset with a time-matched untreated arm is a
prerequisite for any concordance number the successor model reports, and it
should be stated in [dataset.md](dataset.md) as a caveat, not discovered again
later.

---

## 8. Acceptance tests — what must be true before the successor is used

Each is a measurement, and each fails a model that has dp14's defect.

1. **A young state exists and is a state.** ‖f(y₀)‖/‖y₀‖ at the unperturbed
   baseline is small on the fast states; the slow ratchet states are held, not
   equilibrated, and that split is declared.
2. **The unperturbed model does not senesce.** Integrated over the experiment
   window with every perturbation zeroed, the endpoint stays near the young
   state. Enforced by the Phase-0 screen, not by inspection.
3. **Two basins, measured.** Distinct stable fixed points on the projected
   conservation leaf, with a separatrix between them, from
   `bifurcation.equilibrium` after P0.16.
4. **The dose selects the destination, not the path.** A sub-threshold pulse
   relaxes back; a supra-threshold pulse commits. Dose–response is monotone in
   the correct direction across at least three decades.
5. **The insult is carried by a state, not a parameter.** Arms differ only in
   `u(t)`. No arm changes a rate constant or an initial condition.
6. **Late intervention is expressible.** Drugs are `u_drug(t)` with a start
   time, so the day-0 / day-10 / day-18 comparison the 2014 paper claims is a
   simulation the model can actually run.
7. **Arrest is observed, not inferred.** A proliferation readout, not only
   SA-β-gal and ROS, and reversal is judged on it.
8. **No structurally redundant parameters** at construction (P1.13), and
   identifiability assessed jointly.
9. **Held-out arm, and a null baseline reported beside the score** (P1.6).

Tests 2, 3, 4 and 5 are the ones DallePezze 2014 fails, and each of the four
would have been cheap to run in 2014.
