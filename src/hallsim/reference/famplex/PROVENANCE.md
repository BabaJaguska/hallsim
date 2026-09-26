# FamPlex relations — vendored snapshot

- Source: https://raw.githubusercontent.com/sorgerlab/famplex/master/relations.csv
- Fetched: 2026-09-26
- Rows: 5,284 — `HGNC,<gene>,isa|partof,FPLX,<family>` (4,915) and
  `FPLX,<family>,isa|partof,FPLX,<parent family>` (360); the 9 `UP` rows
  are ignored
- Licence: CC0-1.0 (the repository's declared licence)
- Used by: `hallsim.mechanisms.families_of`, to fold a family-level
  literature statement (INDRA grounds "ERK" to the family, not to MAPK1)
  onto the member a model carries

Citation: Bachman, Gyori & Sorger, BMC Bioinformatics 2018 (FamPlex).
