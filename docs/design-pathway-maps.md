# Pathway maps are a knowledge source, not a model source

Written 2026-09-06, after a screen of 119 BioModels deposits returned the
whole SASP panel — `CCL2, IL1A, IL1B, IL6, IL8, MMP1, MMP13, MMP3` — from two
deposits that turned out to carry **zero kinetic laws**. Nothing in them is
integrable, and the first instinct (assign mass action and fit) is wrong by two
orders of magnitude. This is what they are good for instead.

Conclusion up front: **a map belongs in the validation layer, not the
Scheduler.** Its strongest use is refuting a coupling edge, which is the shape
of everything else in this repo that works.

## What is actually in one

Measured on the two deposits the screen surfaced:

| | Wu2010 RA map | Mechanotransduction & Inflammation |
|---|---|---|
| id | MODEL2302140001 | MODEL2406110001 |
| species / reactions | 450 / 254 | 251 / 324 |
| kinetic laws | **0** | **0** |
| species annotated | **0%** (0 UniProt) | **100%** (174 UniProt) |
| modifier types | CATALYSIS 222, **INHIBITION 58**, TRIGGER 46, UNKNOWN_CATALYSIS 10 | CATALYSIS 109, nothing else |
| reaction types | STATE_TRANSITION 158, **TRANSCRIPTION 49**, HETERODIMER_ASSOCIATION 24, TRANSLATION 10, TRANSPORT 7 | STATE_TRANSITION 324 |
| distinct PubMed ids | 28 | 38 |

Two things follow.

**The signs are present but invisible to libsbml.** SBO terms are absent on
every reaction and every modifier in both files. The semantics live in
CellDesigner's own namespace — `<celldesigner:modification type="INHIBITION">`,
`<celldesigner:reactionType>TRANSCRIPTION</...>` — which `libsbml` does not
surface through the core API. A reader that stops at SBO sees bare arrows and
concludes the map carries no direction, which is why the production screen
counted 254 drawn arrows as synthesis.

**Encoding quality is uneven in complementary ways.** Wu2010 has signs and no
node identities; Mechanotransduction has node identities and flattens every
modifier to CATALYSIS. Neither is individually sufficient, and a corpus-level
consumer has to expect both.

## The three defensible uses

### 1. An edge-admissibility oracle — the one that fits this framework

`fas_induction` wired GZ06's p53 to Kallenberger's CD95, cited to Owen-Schaub
1995. The citation supports the edge *existing*; it does not support CD95 being
how an irradiated fibroblast dies, and the composite carried that error for a
week (see `docs/design-fate-architecture.md`). A signed, typed, cited map corpus
is literature-curated adjacency ground truth, and the check is mechanical:
is this declared edge present in any map, with what sign, under which PubMed id
— and which declared edges appear in no map at all.

`hallsim.coupling_wiring` and `CouplingAuditor` already run at composition time
and already own this question. The maps are the missing evidence base, not a new
subsystem.

**The design constraint comes from what a map asserts.** An edge in a map means
*this interaction has been reported in some cell type under some condition*. It
does not mean the edge is operative in WI-38 fibroblasts at day 14. So a map
**refutes a wiring well and supports one weakly** — an absent edge is a strong
signal, a present edge is a weak one. Any consumer that inverts this reproduces
the p53 → CD95 mistake with more machinery behind it.

### 2. Parameter-free structural verdicts

Stoichiometry alone is exactly what a map supplies, and it settles questions
that hold for **every** parameter set:

- **Conserved moieties** from the left null space of `N`.
  `Process.stoichiometry()` and `steady_state.conservation_laws` already exist;
  a map is a source of exact `N` where a fitted model gives only a numerical
  one.
- **Feedback circuits** on the signed graph. Thomas's rules: a positive circuit
  is necessary for multistationarity, a negative circuit for sustained
  oscillation. Their *absence* is a proof of impossibility.
- **Chemical reaction network theory.** Feinberg's deficiency-zero and
  deficiency-one theorems can establish that a network admits no multiple
  positive steady states under any kinetics whatsoever.

These are theorems, not heuristics, and they need no rate constants. This is
the strongest scientific claim available from an unparameterised map.

### 3. Scoping a subnetwork

"Which species lie on a path from DNA damage to IL6 transcription in at most
three steps" turns 324 reactions into roughly fifteen nodes that can be
hand-built or fitted. The map is a mechanism index — a search structure over
what is connected to what, used to decide model *scope* rather than to supply
model *content*.

## What not to do

Assign mass action to the map and fit it. 324 reactions at ~2 parameters each
is ~650 free parameters against the benchmark's 8 timepoints. The composite
already hit an identifiability wall at **3** parameters (`docs/known-problems.md`
P0.50, and the residual-aware selection that scored worse than out-of-the-box).
This is that wall at two hundred times the scale, and no regularisation makes
650 invented numbers into evidence.

The same objection applies to Boolean→ODE conversion (Odefy, BoolODE) of the
SBML-qual siblings these maps are converted into: every threshold and every
timescale is invented. It is the `fas_induction` τ = 3 h objection at network
scale.

## Framework edits

1. **A map is not a `Process`.** No derivative, no ports, nothing for the
   Scheduler to integrate. It needs its own type, consumed by the validation
   layer. Consistent with the existing rule that topology lives outside
   processes.
2. **`intake` routes rather than rejects.** `screen_produced_species` now
   returns `no-rate-laws` and `triage_sbml` would reject the import; that
   verdict should instead admit the deposit to the knowledge lane. The
   distinction *is not a model* / *is not useful* is one the intake currently
   collapses.
3. **A CellDesigner annotation reader** — roughly fifty lines of namespace
   parsing to recover modifier sign and reaction type, which libsbml drops.
   Without it the corpus is unsigned and use (1) is unavailable.
4. **Identity resolution on annotations.** `discovery.uniprot_accessions` and
   `hallsim.gene_reporters` are the join key between a map's UniProt
   accessions and a kinetic model's species. Wu2010's 0% coverage means a
   name-based fallback is needed too, with its own failure modes.

Order: (3) then (1), because an unsigned map supports none of the three uses,
and (2) and (4) are small once there is something to route to.
