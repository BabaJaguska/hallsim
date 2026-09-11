# Nazaret 2009 — BIOMD0000000232

Nazaret C, Heiske M, Thurley K, Mazat J-P. *Mitochondrial energetic
metabolism: a simplified model of TCA cycle with ATP production.*
J Theor Biol 2009;258(3):455–464. Curated BioModels entry.

TCA cycle + respiratory chain + F₁F₀-ATP synthase + adenine-nucleotide
translocase + proton leak, with the inner-membrane potential `DeltaPsi` as a
`rateRule` state and `ADP`/`NADH` as conserved-moiety assignment rules
(`ADP = At − ATP`, `NADH = Nt − NAD`). Native clock: seconds.

## The file

`nazaret2009_BIOMD0000000232.xml` is the deposited entry, byte-identical to
the BioModels download. Its seven rule-target parameters (`ATPcrit`,
`DeltaGtransport`, `DeltaPsi`, `JANT`, `JATP`, `Jleak`, `Jresp`) and two
boundary species (`ADP`, `NADH`) carry no `value=` attribute: the importer
evaluates their initial assignment and assignment rules at `t = 0`, the way
libRoadRunner and COPASI do, and `pytest -m conformance` checks the three
agree on it.

`DeltaPsi` is a `rateRule` state whose initial value the deposit's initial
assignment sets. The model is a steady-state model ("the existence of a
steady state is demonstrated"), so that seeds a relaxation rather than fixing
the answer; anything reading it should equilibrate first. `ADP` and `NADH`
are ASSIGNED ports, recomputed from `At − ATP` and `Nt − NAD` every step.
