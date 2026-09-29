# Census snapshots

Two dated tables, shipped so a deposit can be chosen without first spending
hours regenerating them. **They are snapshots, not a live view.** The date is
in each filename; the repositories move and these do not.

## `models_2026-09-28.csv.gz` — 2,531 deposits, 42 columns

Every BioModels deposit, curated and uncurated, run through the intake gate:
import, solve, clock, annotation, rest state, gradient. Produced by
`hallsim.census` (`supply census`).

One thing this snapshot predates: the importer used to hold a species that
nothing reads at its initial value, and now integrates it. That changes a
model's integrated width and can move the rest-state and growth verdicts, so
those columns were computed under the old behaviour. The stage a deposit
reaches is otherwise unaffected, and a rerun will settle it.

`stage` is **the first gate a deposit fails**, not a quality score, so read it
with the `how` column, which says in words what would fix it:

**`pass` is the best outcome, not `clean`.** The names invite the opposite
reading and cost a contestant real time: `pass` means nothing is outstanding,
while `clean` cleared the ordered gates and still carries a note in `how`.

**A deposit reporting zero species is rule-based, not empty.** Its state is in
`rateRule`-driven parameters: 205 of these carry no species and between one and
nine rate rules. `n_states` is the imported model's own width and `n_rate_rules`
the count, so read those beside `n_species` before writing a deposit off.

| stage | n | what it means |
|---|---|---|
| `pass` | 308 | runs and composes as-is; `how` is empty |
| `clean` | 105 | cleared every gate but still has a note in `how` |
| `at_rest` | 207 | solves; published start is not a rest state |
| `annotated` | 204 | solves; under half its species carry an ontology id |
| `clock` | 812 | solves; declares no time unit |
| `kinetic` | 687 | not an ODE deposit |
| `solves` | 92 | imports; no usable trajectory |
| `imports` | 116 | does not import |

The 1,636 at `clean` through `clock` all run. The last three stages do not.

## `datasets_loadable_2026-09-28.csv.gz` — 58,898 deposits, 20 columns

One row per accession. The run holds 2,033 more: more than one route
enumerates the same GEO series — its curated DataSet and the series itself —
and the mirror logic does not collapse those, so the same accession arrived
twice with two designs that disagreed about the control in 1,401 cases and the
arm count in 725. The export keeps the row a reader would want, preferring a
named control, then arms, then timepoints. The duplication itself is still in
the census and wants fixing there.

`pubmed` is a PubMed id or empty. Two deposits cite a DOI or "in preparation"
instead; a DOI is not a PubMed id, so those are empty here and both are still
in the run's raw rows. Pass `dtype={"pubmed": "Int64"}` to read them as
integers — with empties present a plain read infers a float and shows
`21772264.0`.

The loadable subset of the data census: deposits with a contrast, a measured
quantity, a model match, and a reader that can open them. Produced by
`hallsim.dataset_census`. The full table is 367,373 rows and 847 MB and is not
shipped; regenerate it if you need the rows that did not make this cut.

Readers: series-matrix 29,487, counts-file 25,566, maf 2,183, mwtab 1,629,
petab 31, mztab 2. Of these, 28,439 are human and 7,770 carry three or more
timepoints.

**31 rows are PEtab problems, and they are the only ones needing no bridge.** A
PEtab problem deposits a model, the data it was fitted to, and a formula stating
each observable over that model's own species, which
:class:`hallsim.petab_data.PetabDataset` reads. Every other row leaves the link
between a model's states and a measured quantity to be invented.

**A trap worth stating once.** In the raw table `stage` names the gate a row
stopped at, so `stage == "loadable"` selects rows that reached that gate and
**failed** it. The rows here were selected on the `loadable` flag, and all of
them carry `stage == "pass"`.

**A contrast is not always a design you can set up today.** 286 rows carry a
contrast their source asserts — in a study factor, or an abstract naming its
timepoints — without the groups ever being recovered from the sample titles, so
`n_arms` and `control` are empty on a row that does hold a real course. Reading
those means opening the deposit's own metadata. `design_recovered` is the column
that separates them. Of the 58,898 rows, 32,088 name a control and 10,931 carry
more than one timepoint, but only **5,297 have both**, 2,494 of them human.

**`n_models` is not a per-dataset match score.** It counts the deposits the
matching route reached, and only the `direct` route selects on the dataset's
own identifiers. The other routes select on what a deposit carries, so every
transcriptome reaches the same 119 models-with-transcription-factors — which is
why one value covers 55,059 rows. `n_same_species` is coarse for the same
reason. `n_shared_ids` carries the per-dataset strength and is above zero on
894 rows, being zero off the direct route.

## `metabolights_designs_2026-09-28.json.gz` — 3,395 declared designs

The design each MetaboLights deposit states in its ISA-Tab sample file, keyed
by accession: arms, the control arm, timepoints, per-arm times. 3,271 of the
3,395 support a comparison, 1,544 name a control and 505 carry more than one
timepoint — none of which the EBI Search listing carries, so without this the
rows arrive with no design at all.

It ships because reading it is expensive and the answer does not change: one
FTP request per deposit, seconds each, five hours for the set.
`search.datasets.shipped_designs` loads it and `declared_design` prefers it,
falling back to a fetch for an accession it does not carry. Any
`*_designs_*.json.gz` here is picked up, so a later snapshot adds to it.

Read the designs rather than the study: asking `metabolights_utils` for a whole
study model downloads its data files too, which over these 3,415 deposits is
30 GB.

## Regenerating

```
supply census                   # models
simulate dataset-census         # datasets
```

Both are multi-hour runs against live repositories. Nothing in the package
requires these files; they are a starting shelf, and code that reads them
should degrade to a search rather than fail when they are absent or stale.
