# Nazaret 2009 — BIOMD0000000232

Nazaret C, Heiske M, Thurley K, Mazat J-P. *Mitochondrial energetic
metabolism: a simplified model of TCA cycle with ATP production.*
J Theor Biol 2009;258(3):455–464. Curated BioModels entry.

TCA cycle + respiratory chain + F₁F₀-ATP synthase + adenine-nucleotide
translocase + proton leak, with the inner-membrane potential `DeltaPsi` as a
`rateRule` state and `ADP`/`NADH` as conserved-moiety assignment rules
(`ADP = At − ATP`, `NADH = Nt − NAD`). Native clock: seconds.

## Two files

| file | what it is |
|---|---|
| `nazaret2009_BIOMD0000000232.xml` | the deposited entry, byte-identical to the BioModels download |
| `nazaret2009_BIOMD0000000232_initialised.xml` | the same file with nine missing `value=` / `initialConcentration=` attributes supplied |

The deposited entry does not import: `sbmltoodejax` evaluates every symbol at
`t0` while generating the module, and seven rule-target parameters
(`ATPcrit`, `DeltaGtransport`, `DeltaPsi`, `JANT`, `JATP`, `Jleak`, `Jresp`)
and two boundary species (`ADP`, `NADH`) carry no value in the source. The
failure surfaces as `ValueError: None is not a valid value for jnp.array` from
inside the generated module, with no indication of which symbol is at fault.

## The nine attributes, and where each value comes from

Six are `assignmentRule` targets — their value is overwritten on the first
evaluation, so the number is a placeholder and cannot affect the trajectory:

    ATPcrit=0  DeltaGtransport=0  JANT=0  JATP=0  Jleak=0  Jresp=0

`DeltaPsi` is a `rateRule` target, so its value **is** an initial condition. It
is set to `150`, the model's own `DeltaPsim` reference parameter and the
potential the paper's operating point sits at. The model is a steady-state
model ("the existence of a steady state is demonstrated"), so this seeds a
relaxation rather than fixing the answer; anything reading it should
equilibrate first.

`ADP` and `NADH` follow from the deposited totals and initial values by the
model's own conservation rules:

    ADP  = At − ATP = 4.16 − 3.536 = 0.624
    NADH = Nt − NAD = 1.07 − 0.856 = 0.214

## One thing to know before reading `NADH` or `ADP`

The importer does not evaluate SBML assignment rules on boundary species: the
generated module computes `ADP = At − ATP` and `NADH = Nt − NAD` into its
internal `w` vector, but the store path `naz/ADP` / `naz/NADH` keeps its
initial value for the whole run. The **dynamics are unaffected** — every rate
law in this model spells the conservation out inline (`c[5] − y[8]`,
`c[11] − y[1]`) and never reads those two species — but the two trajectories
are stale constants. Read NADH as `NAD_total − NAD` instead;
`demos.models.mitochondrial_aging.MitoBioenergetics` does exactly that.
