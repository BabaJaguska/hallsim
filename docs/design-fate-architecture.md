# Why DallePezze 2014 and Kallenberger 2014 do not concatenate

Written 2026-09-05, after wiring Kallenberger into the multi-hallmark composite
and measuring what the edge delivers. Conclusion: the apoptosis arm is the
wrong module for this composite, and the reason is architectural rather than
numerical. Recorded so the next attempt starts from here.

## The claim we were building toward

DP14 has a single destination — damage in, senescence markers up — so
"the cell senesces" is an assumption of the model set. Adding an apoptosis
module was meant to give the damage signal a second destination and make the
fate a computed result.

## Why it does not work

**Different cell type, stimulus and pathway.** DP14 is 20 Gy irradiation of
MRC5 fibroblasts; Kallenberger is CD95-ligand-induced *extrinsic* apoptosis in
HeLa. For irradiation the relevant death route is *intrinsic*:
p53-killer → PUMA/BAX/BAK → MOMP → cytochrome c → caspase-9/3. Neither deposit
contains it.

**The edge we built is the wrong branch.** `fas_induction` drives Kallenberger's
CD95 receptor from GZ06's p53, cited to Owen-Schaub 1995 (wild-type p53 raises
surface Fas 3–4×). That citation supports the edge *existing*; it does not
support CD95 being how an irradiated fibroblast dies. Irradiation should not be
mapped to CD95L or caspase-8 without evidence that the ligand/receptor arm is
actually engaged.

**Our own measurements said so before the literature did.** The edge delivers
**1.13–1.15×** arm separation against a measured 3–4×, and delivers even that
only with an uncited receptor-turnover lag (τ = 3 h) added to rectify GZ06's
damage-blind mean p53. A pathway that needs an invented parameter to carry any
signal at all is a pathway that is not carrying the signal.

**Kallenberger is a timer, not a switch.** Bid is consumed irreversibly and
never resynthesised, so once any DISC forms the cleavage runs to completion:
`tBid` reaches ~99% of the Bid pool for *any* nonzero ligand dose given enough
time. Dose sets the rate, not the endpoint. Every threshold on it therefore
fires eventually in every cell — measured, SA-β-gal collapsed 9.27 → 1.78 at
doses where zero cells were committed at the 240-minute readout.

**Nothing routes.** Until `commitment=True` was added there was no edge from the
apoptosis arm back to the senescence arm, so a cell read as fully senescent and
fully apoptotic at once. Structurally: the k14 arm writes nothing that any
upstream process reads.

## What the architecture would have to be

Radiation/repair → ATM–p53 decision core → **two branches**:

- p21/p16–CDK2 → DP14 senescence maturation
- PUMA/BAX/BAK → MOMP → caspases

with Kallenberger optional, hanging off **tBID → MOMP** as an extrinsic arm and
with CD95L as an *independently controlled* input — never downstream of
irradiation.

The pieces that exist in the literature: Hat 2016 (p53-arrester/p53-killer with
the bifurcation structure that makes the switch), Tian 2012 (nuclear +
mitochondrial p53 → PUMA/BAX → cytochrome c → caspase-3, the intrinsic bridge),
Kollarovic 2016 (bistable hysteretic CDK2/p21 senescence commitment), Dolan 2015
(stochastic NHEJ repair with a slow senescence integrator). None is a single
calibrated model spanning mature senescence *and* both apoptosis routes in one
cell system.

## What we keep

`GatedRemoval` (`hallsim/models/gated_removal.py`) came out of this and is
correct independent of the biology: first-order removal of a state while a gate
is open, which is the cross-inhibition shape any fate competition needs
("MOMP/caspases ⊣ senescence accumulation"). Its trigger should be MOMP, not
tBID from a CD95 arm.

The composition machinery is untouched by any of this: three deposits, three
native clocks (days/hours/minutes), one reconciled axis, one differentiable
system.

## Where the code is

The wired composite is preserved on branch **`apoptosis-arm`** (commit
`a9063a9`): k14, the p53 → CD95 edge, the receptor-turnover lag, the
commitment return edge and the population fate split. Master carries the
two-model composite (DP14 + GZ06) until a SASP module replaces it.

`GatedRemoval` and `discovery.screen_produced_species` stay on master — they
are framework primitives, not biology.

## Decision

Apoptosis is a larger build than a readout axis. The route from here is either
the intrinsic module above — a multi-week job requiring recalibration in one
cell type — or a different third model.

DP14's authors named the gap themselves: the paper omits inflammatory
NF-κB/TNF-α/TGF-β signalling. GSE248823 moves those transcripts hard
(CCL2 +3.05, CXCL1 +2.68, IL6 +1.73 log2FC at D14), so an inflammation module
is both author-acknowledged and scoreable, which the apoptosis arm never was.

The constraint from the Ihekwaba 2004 failure (removed 2026-08-31) is that an
NF-κB core is not enough: it contributed nothing with both edges ablated, its
edges supplied 100% of its own IKK, and its only NF-κB-inducible transcript was
its own inhibitor. **A replacement must emit the SASP effectors the data
actually moves**, not just oscillate NF-κB/IκB.
