# Priors for the cross-model coupling-edge strengths

Reference note for the preprint: how the strengths of the composite's
literature-derived coupling edges are bounded, and why they have no direct
off-the-shelf literature value.

The flagship has two coupling edges that inject a Hill-gated source into
the Ihekwaba 2004 NF-κB module's IKK pool:

- **`MtorNFkBActivator`** (`mtor_nfkb.k_act`): DP14 `mTORC1_pS2448` → `nfkb/IKK`.
- **`DamageNFkBActivator`** (`damage_nfkb.k_act`): DP14 `DNA_damage` → `nfkb/IKK`.

Each contributes `d(IKK)/dt += k_act · H_act(signal; K, n)`, where
`H_act ∈ [0,1]` is a Hill gate. `k_act` is therefore the **maximum rate at
which the edge can raise the IKK pool** — a phenomenological rate in the
composite's units, not a directly measurable biochemical constant.

## The pathways are literature-backed; the strengths are not

The *existence and direction* of both edges are well established:

- **mTORC1 → IKK** (activating): Dan et al. 2008, *Genes Dev* (mTORC1/Raptor
  required for IKK activity); Laberge et al. 2015, *Nat Cell Biol* (mTOR →
  IL1A → NF-κB/SASP; rapamycin lowers it).
- **DNA damage → IKK** (activating), via the ATM → NEMO → IKK genotoxic
  axis: Wu et al. 2006, *Science*; Miyamoto 2011, *Cell Res*; required for
  the NF-κB-dependent SASP in senescence, Salminen et al. 2012, *Cell
  Signal*.

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
