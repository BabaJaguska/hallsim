# Design — admissibility screening for discovered models

Planned, not built. Companion to
[design-spontaneous-endpoint.md](design-spontaneous-endpoint.md): that screen
asks whether a model's *dynamics* support the claim being made with it; this
one asks whether the model is *about the right thing* before anyone imports it.

## The gap

`ModelCandidate` carries `source, id, name, format, url, curated, submitter`.
Nothing in it describes the organism, the cell type, the process modelled, or
the perturbation the model was built around. So `search_for_model("IL-6
signalling")` returns a ranked list in which a hepatoma signalling model, a
whole-body Crohn's PK/PD model and a chondrocyte ageing model are
indistinguishable, and the only way to tell is to read the paper.

An agent that does not read the paper picks on name similarity. That is the
failure mode CLAUDE.md's discussion of confidently-wrong agents names, and it
is not hypothetical:

**Worked failure — Gérard 2014 (PLoS Comput Biol 10:e1003455).** Proposed as
the composite's SASP module because it contains NF-κB, IL-6 and STAT3 wired in
a bistable loop — every keyword the slot needed. It is a model of **oncogenic
cell transformation**. Oncogene-induced senescence is the tumour-*suppressive*
alternative to transformation, so the two are opposite fates from the same
stimulus, and the model's ON state is the one senescence is defined against.
The model's own numbers say it plainly: at the ON attractor let-7 = 9×10⁻⁴ and
Lin28 is high, the anti-senescent configuration, against let-7 = 73.05 at OFF.
Reading ON as "SASP-positive senescent cell" inverts the model's semantics.
Nothing in the search result, the model name, the species list or the numerical
triage would have caught this. Only the paper does.

## Layer 1 — context extraction (deterministic, no LLM)

Most of what is needed is already in the record and is simply not read. Extend
`ModelCandidate`:

| field | source |
|---|---|
| `taxon` | `bqbiol:hasTaxon` → NCBI taxonomy id + name |
| `cell_types` | BTO / CL / EFO terms in the model and species annotations |
| `disease` | DOID / MeSH terms |
| `publication` | title, abstract, journal, year, PubMed id |
| `native_time_declared` | whether the SBML declares `timeUnits` at all |
| `n_species`, `n_params` | already available from triage |

**`format: SBML` does not mean importable, and the current filter trusts it.**
`search_biomodels(sbml_only=True)` passes anything the registry labels SBML.
SBML-qual is SBML: BioModels returns `format: SBML` for Sizek 2023
(MODEL2312140001, "a Boolean model of MiDAS") and Verlingue 2016
(MODEL1611180000, "a Boolean model of geroconversion"), neither of which
`sbmltoodejax` can turn into a right-hand side. Add a `kind` field —
ODE / qualitative / constraint-based / hybrid — determined from the SBML
package declarations (`qual:`, `fbc:`) rather than the registry's format
string, and surface it in the candidate. This is deterministic and needs no
LLM; it is the cheapest rejection available and it fires before any download.

Two more are worth a hard report on their own:

- **`taxon` is often generic or absent.** Yao 2008 annotates taxonomy/40674
  (Mammalia) while the model was calibrated in rat REF52 fibroblasts. A generic
  taxon is not "compatible", it is "unstated", and must display differently.
- **`native_time_declared` is false almost everywhere.** Every candidate
  triaged for the inflammatory slot — BIOMD 151, 544, 560 — flags "no declared
  time unit", so their clocks are guesses. Composing a guessed clock against a
  days-native model is silently 60×/3600×/86400× wrong.

## Layer 2 — the admissibility screen (LLM, structured output)

Given (a) the extracted context above, (b) the publication abstract, and (c) a
**declared target context** for the slot being filled, return a structured
verdict per criterion. The target context is the caller's responsibility and
must be explicit — e.g. for the current composite: *human diploid fibroblasts,
DNA-damage- and oncogene-induced senescence, 0–21 day horizon, needs an
emitted secretory effector.*

### The checklist

1. **Organism** — human / mouse / rat / bacterial / generic. If it differs from
   the target, is the merge defensible for *this* mechanism, or is the pathway
   known to differ between them? "Mammalian" is unstated, not compatible.
2. **Cell type and tissue** — fibroblast, hepatocyte, neuron, epithelial,
   chondrocyte, immune. A signalling topology calibrated in HepG2 is not a
   fibroblast model; a neural line is not a fibroblast.
3. **Process identity** — the one that catches Gérard. Senescence,
   quiescence, transformation, apoptosis, differentiation and proliferative
   arrest are distinct fates, and models of them are not interchangeable
   because they share transcription factors. Name the process the model is
   *of*, not the molecules it contains.
4. **State semantics** — if the model is bistable or switch-like, what do its
   attractors *mean*, and do they map onto the states the slot needs? An ON
   state meaning "transformed" cannot be read as "senescent".
5. **Perturbation identity** — irradiation, oncogene induction, oxidative
   stress, replicative exhaustion, nutrient withdrawal and cytokine stimulation
   are different insults with different kinetics. A model built around one does
   not transfer to another without saying why.
6. **Timescale** — does the model's own horizon overlap the target's? A
   seconds-to-minutes signalling module and a 14-day phenotype model can
   compose, but only deliberately, and only with the clock declared.
7. **Emission** — what does the model *output* that the composite can read? A
   module whose only inducible species is its own inhibitor contributes no
   observable. (Ihekwaba 2004 was removed for exactly this.)
8. **Calibration context** — what data was it fitted to, and under what
   conditions? Does it contain a control arm?

### Output contract

One verdict per criterion — `compatible` / `caveat` / `mismatch` /
`unstated` — each with a one-line reason and, where possible, the quote from
the abstract or annotation it rests on. Plus an overall recommendation and the
single most load-bearing question a reviewer should be asked.

**`unstated` is a distinct verdict from `compatible`,** and this is the whole
point: a generic taxon, a missing cell type, an absent time unit are all
absences of evidence, and collapsing them into "fine" reintroduces the failure.

## Calibrating strictness — the target context is a gradient, not a gate

The first target context written for the SASP slot demanded primary human
diploid fibroblasts, a specific published circuit (p38 -> NF-kB/C-EBPB with
IL-1a/IL1R1 upstream and CXCR2 autocrine), and emission of six named
chemokines. Nothing in any repository matched, and a specification that
matches nothing is not selective, it is empty. It also encoded a demo's
requirements as though they were the framework's.

So a target context declares each criterion at one of three strengths, and the
screen reports against the strength given rather than treating everything as
mandatory:

| strength | meaning | example for a SASP slot |
|---|---|---|
| **required** | a mismatch disqualifies | emits at least one measurable secreted effector; output is graded, not latched |
| **preferred** | a mismatch is a caveat carried forward | human; non-transformed; fibroblast or generic rather than a named cancer line |
| **informative** | recorded, never scored | the specific upstream circuit; the exact chemokine set |

Two rules keep this from collapsing back into "anything will do":

- **A `required` criterion must be a property of the *model*, not of its
  provenance.** "Emits a secreted effector" and "responds gradedly to its
  input" are testable against the file. "Was calibrated in WI-38" is
  provenance, and provenance belongs in `preferred` — it is a reason to
  caveat, not to reject.
- **Cell type is `preferred`, process identity is `required`.** A signalling
  topology calibrated in one cell type transfers with a caveat; a model of a
  different *fate* does not transfer at all. Neurons versus fibroblasts is a
  caveat. Transformation versus senescence is a rejection.

The failure that motivated this whole document was a process-identity
mismatch, not a cell-type one — which is why loosening the cell-type demand
costs nothing and loosening process identity would cost everything.

## What it must not do

- **Not auto-reject, and not auto-accept.** It is advisory. It ranks and it
  explains; a person or the review panel decides. A screen that silently
  filters the corpus makes the corpus invisible.
- **Not replace the review panel.** `.claude/agents/{bench-scientist,
  mathematician,physicist}.md` pull the cited papers and check the numbers.
  This screen decides which candidates are worth that cost.
- **Not judge the model's quality.** Admissibility for a slot is orthogonal to
  whether the model is any good; that is triage plus the panel.

## Where it hooks in

```
search_for_model(query, target_context=...)   # extraction + screen, ranked
candidate.admissibility(target_context)       # screen one candidate
intake.triage_batch(...)                      # numerical gate, unchanged
```

Order matters: **admissibility before triage.** Triage downloads and solves;
there is no reason to spend that on a model that is about the wrong process.

## Tests

1. **Gérard 2014 against a senescence target** must return `mismatch` on
   criterion 3 and 4. This is the regression that motivates the feature.
2. **Yao 2008 against a human fibroblast target** must return `unstated` on
   organism (taxonomy/40674 is Mammalia) — not `compatible`.
3. **Ihekwaba 2004 against a SASP target** must return `mismatch` on
   criterion 7 (emission) and `mismatch` on organism.
4. **DallePezze 2014 against the same target** must return `compatible` on
   organism, cell type and process — the positive control, or the screen is
   just a rejector.
5. **Every candidate with no declared `timeUnits`** must surface `unstated` on
   criterion 6, regardless of what the LLM says.

Criteria 2, 3 and 5 of the test list are deterministic and must be asserted
without an LLM in the loop, so the test suite does not depend on a model call.

## Corpus findings that motivate this

Measured while searching for the inflammatory slot, and worth reporting as a
statement about the field rather than only as a local difficulty:

- **BioModels returns zero hits** for "senescence SASP",
  "senescence-associated secretory phenotype", "IL-6 IL-8 senescence",
  "NF-kB senescence fibroblast" and "p38 MAPK senescence". The only hit for
  "cellular senescence inflammation" is DallePezze 2014 itself.
- Of the four plausible inflammatory candidates triaged (BIOMD 151, 544, 549,
  560), **three flag "no declared time unit"** and the fourth has two species.

The module the composite needs may not exist in the public corpus in a form
that can be imported. That is a finding about the integration tax, and it is
the argument the framework exists to make.
