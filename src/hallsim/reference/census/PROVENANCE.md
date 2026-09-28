# Census snapshots

Two dated tables, shipped so a deposit can be chosen without first spending
hours regenerating them. **They are snapshots, not a live view.** The date is
in each filename; the repositories move and these do not.

## `models_2026-09-20.csv.gz` — 2,531 deposits, 40 columns

Every BioModels deposit, curated and uncurated, run through the intake gate:
import, solve, clock, annotation, rest state, gradient. Produced by
`hallsim.census` (`simulate census`).

`stage` is **the first gate a deposit fails**, not a quality score, so read it
with the `how` column, which says in words what would fix it:

**`pass` is the best outcome, not `clean`.** The names invite the opposite
reading and cost a contestant real time: `pass` means nothing is outstanding,
while `clean` cleared the ordered gates and still carries a defect note — 20 of
those 25 are `needs-review`, one of them for every state decaying to zero.

| stage | n | what it means |
|---|---|---|
| `pass` | 305 | runs and composes as-is; `how` is empty |
| `clean` | 25 | cleared every gate but still has a note in `how` |
| `at_rest` | 187 | solves; published start is not a rest state |
| `annotated` | 269 | solves; under half its species carry an ontology id |
| `clock` | 781 | solves; declares no time unit |
| `kinetic` | 683 | not an ODE deposit |
| `solves` | 86 | imports; no usable trajectory |
| `imports` | 195 | does not import |

The 1,567 at `clean` through `clock` all run. The last three stages do not.

## `datasets_loadable_2026-09-25.csv.gz` — 55,186 deposits, 18 columns

The loadable subset of the data census: deposits with a contrast, a measured
quantity, a model match, and a reader that can open them. Produced by
`hallsim.dataset_census`. The full table is 320,491 rows and 590 MB and is not
shipped; regenerate it if you need the rows that did not make this cut.

Readers: series-matrix 28,536, counts-file 25,566, maf 665, mwtab 417,
mztab 2. Of these, 28,478 are human and 7,236 carry three or more timepoints.

**A trap worth stating once.** In the raw table `stage` names the gate a row
stopped at, so `stage == "loadable"` selects rows that reached that gate and
**failed** it. The rows here were selected on the `loadable` flag, and all of
them carry `stage == "pass"`.

**A contrast is not always a design you can set up today.** 1,221 rows in this
snapshot — every metabolomics and proteomics row, and 137 expression rows —
carry `n_arms=0`, `n_timepoints=0` and no control. They are here legitimately:
their source asserts a course in a study factor or an abstract, which is real
information, but the groups were never recovered from the sample titles and
reading them means opening the deposit's own metadata. From the next run
onward `design_recovered` separates those from the rows whose arms are known;
in this snapshot, filter on `n_arms >= 2 or n_timepoints >= 2` instead. Of the
55,186 rows, 27,194 name a control and 10,061 carry more than one timepoint,
but only **4,305 have both**.

**`n_models` is not a per-dataset match score.** It counts the deposits the
matching route reached, and only the `direct` route selects on the dataset's
own identifiers. The other routes select on what a deposit carries, so every
transcriptome reaches the same 119 models-with-transcription-factors — which is
why one value covers 54,106 rows. `n_same_species` is coarse for the same
reason. From the next run onward `n_shared_ids` carries the per-dataset
strength, and it is zero off the direct route.

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
simulate census                 # models
simulate dataset-census         # datasets
```

Both are multi-hour runs against live repositories. Nothing in the package
requires these files; they are a starting shelf, and code that reads them
should degrade to a search rather than fail when they are absent or stale.
