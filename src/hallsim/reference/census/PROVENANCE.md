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

| stage | n | what it means |
|---|---|---|
| `clean` | 25 | clears every check |
| `pass` | 305 | runs and composes as-is |
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

## Regenerating

```
simulate census                 # models
simulate dataset-census         # datasets
```

Both are multi-hour runs against live repositories. Nothing in the package
requires these files; they are a starting shelf, and code that reads them
should degrade to a search rather than fail when they are absent or stale.
