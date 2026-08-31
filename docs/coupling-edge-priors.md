# Priors for the cross-model coupling-edge strengths

> **Superseded 2026-08-31.** Both edges this note is about — `mtor_nfkb` and
> `ikkbeta_nfkb` — were deleted with the Ihekwaba 2004 constituent. The refereed
> findings against them, kept because they are the reason:
>
> - **The axis was inert.** 19/24 sign agreement whether both edges were live,
>   ablated, or scaled ×10; with both at zero the module was bit-identical
>   across all three arms.
> - **The `ikkbeta_nfkb` edge ran backwards.** `dp14/IKKbeta` is *higher in
>   control than in DDIS*, so it fired hardest where the perturbation was
>   absent. Confirmed by all three reviewers.
> - **§"Prior: anchored to the host module" is wrong twice.** IKK is not a
>   conserved pool — `v64` removes it at `k61·IKK` (t½ 96 min) with no source
>   anywhere, so `k_act` was the module's *entire* input rather than a
>   perturbation of it. And the quoted IKKβ band (ctrl 11.9 → DDIS 16.5) is
>   stale by ~5× and inverted.
> - **`dp14/IKKbeta` is not a kinase reading.** Both its constants are 1 and
>   marked "Assumed" in DallePezze's Table S4, making it a unity-gain low-pass
>   of `dp14/ROS` — the edge was ROS→NF-κB wearing a kinase's name. Ihekwaba's
>   `IKK` is `hasPart` {IKKβ, NEMO, IKKα} in µM; DP14's is `isVersionOf` IKBKB
>   and dimensionless.
>
> What generalises, and is now enforced in code: a coupling edge whose gate sits
> outside its driver's realised range, or whose driver is higher in the
> reference conditions than in any perturbed one, **raises** at
> `CalibrationProblem` construction (`check_hill_gates`). See
> `docs/known-problems.md` P0.18 and the 2026-08-31 diary entry.

Reference note for the preprint: how the strengths of the composite's
literature-derived coupling edges are bounded, and why they have no direct
off-the-shelf literature value.

The multi-hallmark demo has two `HillActivationEdge`s that inject a Hill-gated source
into the Ihekwaba 2004 NF-κB module's IKK pool:

- **`mtor_nfkb`** (`mtor_nfkb.k_act`): DP14 `mTORC1_pS2448` → `nfkb/IKK`.
- **`ikkbeta_nfkb`** (`ikkbeta_nfkb.k_act`): DP14 `IKKbeta` → `nfkb/IKK`.

Each contributes `d(IKK)/dt += k_act · H_act(signal; K, n)`, where
`H_act ∈ [0,1]` is a Hill gate. `k_act` is therefore the **maximum rate at
which the edge can raise the IKK pool** — a phenomenological rate in the
composite's units, not a directly measurable biochemical constant.

## The pathways are literature-backed; the strengths are not

The *existence and direction* of both edges are well established:

- **mTORC1 → IKK** (activating): Dan et al. 2008, *Genes Dev* (mTORC1/Raptor
  required for IKK activity); Laberge et al. 2015, *Nat Cell Biol* (mTOR →
  IL1A → NF-κB/SASP; rapamycin lowers it).
- **IKKβ → IKK** (activating): DP14's IKKβ is the same kinase (IKBKB) as
  the Ihekwaba signalosome pool; its defining catalytic role is to activate
  NF-κB (Karin & Ben-Neriah 2000, *Annu Rev Immunol*; primary: DiDonato et
  al. 1997, *Nature*). The genomic-instability drive reaches it because IKKβ
  is ROS-activated inside DP14 (DallePezze 2014's own reaction), not through
  a separate ROS→IKK edge — required for the NF-κB-dependent SASP in
  senescence (Salminen et al. 2012, *Cell Signal*).
  Replaces an earlier phenomenological `DNA_damage → IKK` gate. IKKβ's
  homeostatic band is narrow (ctrl 11.9 → DDIS 16.5), so `K=25/n=4` sits in
  the gate's low-occupancy foot — near-silent at baseline (`H≈0.05`, so the
  NF-κB module equilibrates) and rising super-linearly with the ROS-driven
  IKKβ increase (`H≈0.16` at DDIS). Not a free knob; anchored to IKKβ's
  measured operating range.

But **no paper reports `k_act` in our units.** The Hill-gated source is a
phenomenological abstraction of a multi-step signalling cascade; its
strength depends on the composite's IKK scale, not on a published rate.

## Prior: anchored to the host module (Ihekwaba 2004)

The correct, unit-consistent anchor is the module the edge writes into.
Ihekwaba 2004 (BIOMD0000000230) sets the IKK scale:

- IKK is a **conserved pool**, initial concentration **0.1** (matching the
  edges' port default), shuttling through IκB complexes rather than being
  produced.
- Free-IKK turnover rate constants: binding `k34 = 0.0225`, catalysis
  `k62 = 0.00407`, unbinding `k35 = 0.00125`; characteristic flux through
  the pool `k34·IKK·IkBa ≈ 4×10⁻⁴` (IKK-conc per native-time unit).

So the **natural scale for `k_act` is the IKK pool itself (~0.1)**: the edge
should be able to modulate IKK on the order of its own pool without
dominating the intrinsic dynamics. This gives a **weakly-informative,
Occam-style prior** — keep the edge near the host IKK scale, and let the
data pull it only as far as needed.

This is **implemented** as a log-normal MAP penalty: each edge's
`ParameterRef` sets `prior=0.1` (the host-IKK scale) with
`prior_sigma=0.5` (log10 decades), and `CalibrationProblem.prior_weight`
scales `Σ((log10 p − log10 prior)/prior_sigma)²` into the loss. Edge
clamps are tightened to (1e-4, 1.0) as a hard backstop. Without this the
under-constrained fit (8 params, 6 fit-arm reporters) drove
`damage_to_nfkb` toward its clamp; with it, the edge stays at ~0.1.

## Why not Konrath's numbers

Konrath 2023 (`MODEL2307130001`) *does* publish rate constants for the
genotoxic → IKK cascade (`SFM_k1 ≈ 9.8×10⁻⁸`, `SFM_k2 ≈ 0.053`,
`TM_k3 ≈ 1.5×10⁻⁴`). They confirm the pathway is quantifiable, but they are
in Konrath's **molecule-count** units (IKK ≈ 10⁵ molecules), not
transferable to Ihekwaba's normalised concentration pool (IKK = 0.1)
without a scaling that would itself be a guess. So the host-module scale,
not Konrath's absolute values, is the right anchor. (Konrath was evaluated
as a coupled model and dropped — dead-sink `pIKK`, seconds-vs-days clock —
see the diary; only its confirmation that the cascade is quantifiable
carries over.)

## Status

These are **order-of-magnitude priors, not measurements** — appropriate for
a phenomenological coupling strength with no direct literature value. As MAP
priors they keep the edges from dominating the host module; the data (the
NFKBIA reporter across the DDIS/rapamycin arms) sets the actual value within
that scale. `prior_weight` (the data-vs-prior trade-off) is a judgment call
with this little data — worth a sensitivity check rather than treating one
value as canonical.
