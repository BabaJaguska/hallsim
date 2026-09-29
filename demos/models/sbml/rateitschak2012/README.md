# Rateitschak 2012 — BIOMD0000000585

Rateitschak K, Winter F, Lange F, Jaster R, Wolkenhauer O. *Parameter
identifiability and sensitivity analysis predict targets for enhancement of
STAT1 activity in pancreatic cancer and stellate cells.* PLoS Comput Biol
2012;8(12):e1002815. Curated BioModels entry, deposited as `MODEL1509240000`
under the accession `BIOMD0000000585`.

IFN-γ through IFNGR into the JAK/STAT1 cascade: cytoplasmic, nuclear and
dimerised STAT1 with SOCS1 feedback, an explicit `Ifng` input, and nuclear
STAT1 as the transcription-factor-activity readout. Native clock: minutes
(the paper quotes min⁻¹ and observes for nine hours). The deposited initial
condition is not a rest state — SOCS1 sits at 0.108 against 0.163 — so
equilibrate before applying anything.

## The file

`rateitschak2012_BIOMD0000000585.xml` is the deposited entry, byte-identical
to the BioModels download.

Eight species carry an `(observed)` suffix and state the deposit's own
observables; exactly two of them, `Stat1ex` and `Socs1ex`, carry an
`hgnc.symbol`, and those two are its entire ontology coverage. They are what
`derive_readouts` joins by identity, which is why the importer keeping
assignment-rule annotations is regression-tested against this file rather than
a synthetic one — a curated deposit states its observables exactly this way,
so a synthetic fixture cannot stand in for it.

It is vendored because those regressions otherwise pass only where the
BioModels cache is warm: EBI answers a laptop and returns 403 to GitHub's
runners.
