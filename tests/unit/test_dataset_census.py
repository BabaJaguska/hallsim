"""The data census screens a candidate through nested gates and reports.

Offline: candidates and models are built here; the repositories are
tested in test_datasets.
"""

import json

from hallsim import dataset_census as dc
from hallsim.datasets import DatasetCandidate, Measured

TITLES = [
    "WI38_ETOPOSIDE_D00_REP1",
    "WI38_ETOPOSIDE_D07_REP1",
    "WI38_ETOPOSIDE_D14_REP1",
    "WI38_ETOPOSIDE_RAPAMYCIN_D07_REP1",
    "WI38_ETOPOSIDE_RAPAMYCIN_D14_REP1",
]


def _models():
    return [
        dc.ModelIds(
            "BIOMD1",
            "p53 and ATP",
            frozenset({"P04637"}),
            frozenset({"chebi:15422"}),
            frozenset({"P04637"}),
        ),
        dc.ModelIds(
            "BIOMD2",
            "a metabolite",
            frozenset(),
            frozenset({"chebi:16761"}),
            frozenset(),
        ),
    ]


def _cand(source, measured, **kw):
    base = dict(
        accession="X1",
        title="",
        kind="",
        organism="Homo sapiens",
        n_samples=0,
        platform="",
        url="",
    )
    base.update(kw)
    return DatasetCandidate(source=source, measured=measured, **base)


def test_an_array_time_course_clears_every_gate():
    cand = _cand(
        "geo",
        Measured("expression", True),
        accession="GSE1",
        kind="Expression profiling by array",
        samples=tuple(TITLES),
        n_samples=5,
    )
    row = dc.screen_dataset(cand, _models(), route="geo")
    assert row["timed_evidence"] == "titles" and row["n_timepoints"] == 3
    # every sample is etoposide-treated, so that is the study, not an arm
    assert row["n_arms"] == 2 and row["perturbations"] == ["RAPAMYCIN"]
    assert (row["via"], row["n_models"]) == ("regulon", 1)
    assert row["top_models"][0]["model"] == "BIOMD1"
    assert row["loader"] == "series-matrix" and row["stage"] == "pass"
    assert dc.reason_of(row) == ""


def test_listed_metabolites_pair_directly_and_are_readable():
    cand = _cand(
        "metabolights",
        Measured("metabolomics", False, ("chebi:15422", "chebi:99")),
        accession="MTBLS1",
        factors=("Timepoint", "Treatment:Rapamycin"),
    )
    row = dc.screen_dataset(cand, _models(), route="ebi")
    assert row["timed_evidence"] == "factors"
    assert row["perturbations"] == ["Rapamycin"]
    assert row["via"] == "direct" and row["n_models"] == 1
    assert row["direct_pairs"] == [
        {"model": "BIOMD1", "n_shared": 1, "shared": ["chebi:15422"]}
    ]
    assert row["loader"] == "maf" and row["stage"] == "pass"
    assert dc.reason_of(row) == ""


def test_a_proteome_reads_its_time_course_from_the_text():
    cand = _cand(
        "pride",
        Measured("proteomics", True),
        accession="PXD1",
        summary="Lysates at 0, 6 and 24 h after rapamycin",
    )
    row = dc.screen_dataset(cand, _models(), route="ebi")
    assert row["timed_evidence"] == "text"
    assert (row["via"], row["n_models"]) == ("complete", 1)
    assert row["stage"] == "loadable" and row["loader"] == "result-files"


def test_the_earlier_gates_name_their_reason():
    genotype = dc.screen_dataset(
        _cand("geo", Measured("genotype", True), summary="0, 5, 10 min"),
        _models(),
        route="geo",
    )
    assert genotype["stage"] == "measured"
    assert "genotype" in dc.reason_of(genotype)
    short = dc.screen_dataset(
        _cand(
            "geo",
            Measured("expression", True),
            samples=("ctrl 0h", "ctrl 24h", "TNF 0h", "TNF 24h"),
            n_samples=4,
        ),
        _models(),
        route="geo",
    )
    assert short["stage"] == "timed" and not short["timed"]
    assert dc.reason_of(short) == "2 timepoints, fewer than three"
    assert short["perturbed"] and short["control"] == "ctrl"
    unmatched = dc.screen_dataset(
        _cand(
            "metabolights",
            Measured("metabolomics", False, ("chebi:1",)),
            factors=("time",),
        ),
        _models(),
        route="ebi",
    )
    assert unmatched["stage"] == "matched"


def test_a_counts_file_is_a_loader_route():
    cand = _cand(
        "geo",
        Measured("expression", True),
        kind="Expression profiling by high throughput sequencing",
        files=("GSE1_raw_counts.txt.gz", "GSE1_RAW.tar"),
    )
    assert dc.loader_of(cand) == "counts-file"
    assert dc.loader_of(_cand("geo", Measured("expression", True))) == "none"


def test_the_report_reproduces_from_the_rows(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        dc.screen_dataset(
            _cand(
                "geo",
                Measured("expression", True),
                accession="GSE1",
                kind="Expression profiling by array",
                samples=tuple(TITLES),
                n_samples=5,
            ),
            _models(),
            route="geo",
        ),
        dc.screen_dataset(
            _cand(
                "metabolights",
                Measured("metabolomics", False, ("chebi:15422",)),
                accession="MTBLS1",
                factors=("Timepoint",),
            ),
            _models(),
            route="ebi",
        ),
        dc.screen_dataset(
            _cand("bioimages", Measured("imaging"), accession="S-BIAD1"),
            _models(),
            route="bioimages",
        ),
    ]
    (run / "rows.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    (run / "config.json").write_text(
        json.dumps({"routes": ["geo", "ebi"], "n_models": 2})
    )
    (run / "papers.jsonl").write_text(
        json.dumps(
            {
                "pubmed": "1",
                "models": ["BIOMD1"],
                "pmcid": "PMC1",
                "has_data": True,
                "has_supplement": True,
                "datasets": [
                    {
                        "source": "supplement",
                        "accession": "S-EPMC1",
                        "files": ["a.pdf", "b.xlsx"],
                    }
                ],
                "chemicals": [["chebi:15422", "ATP"]],
                "biomodels": ["BIOMD1"],
            }
        )
        + "\n"
    )
    report = dc.write_report(run)
    text = report.read_text()
    assert (
        "## Funnel" in text
        and "route: geo" in text
        and "modality: imaging" in text
    )
    assert "The models' own papers" in text and ".xlsx 1" in text
    summary = json.loads((run / "summary.json").read_text())
    assert summary["funnel"] == [
        ["listed", 3],
        ["timed", 2],
        ["measured", 2],
        ["matched", 2],
        ["loadable", 2],
    ]
    assert summary["arms"]["two"] == 1 and summary["timed_unperturbed"] == 1
    pairs = (run / "pairs.csv").read_text().splitlines()
    assert len(pairs) == 2 and "MTBLS1" in pairs[1] and "BIOMD1" in pairs[1]
    assert (run / "funnel.png").exists()
    failures = (run / "failures.csv").read_text()
    assert "S-BIAD1" in failures and "GSE1" not in failures


def test_rows_screen_again_from_their_raw_part(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    cand = _cand(
        "metabolights",
        Measured("metabolomics", False, ("chebi:15422",)),
        accession="MTBLS1",
        factors=("Timepoint", "Treatment:Rapamycin"),
        summary="a metabolite time course",
    )
    row = dc.screen_dataset(cand, _models(), route="ebi")
    assert dc.candidate_of(row) == cand
    (run / "rows.jsonl").write_text(json.dumps(row) + "\n")
    # a model set without the metabolite: the same row no longer matches
    other = [
        dc.ModelIds(
            "BIOMD9", "", frozenset(), frozenset({"chebi:1"}), frozenset()
        )
    ]
    assert dc.rescreen(run, other) == 1
    again = json.loads((run / "rows.jsonl").read_text())
    assert again["stage"] == "matched" and again["perturbations"] == [
        "Rapamycin"
    ]
    assert dc.rescreen(run, _models()) == 1
    assert json.loads((run / "rows.jsonl").read_text())["stage"] == "pass"


def test_a_panel_is_nameable_even_though_it_is_not_a_proteome():
    """A reporter reads one species, so a chosen panel is as nameable as a
    whole proteome; only the route it matches by differs."""
    row = dc.screen_dataset(
        _cand(
            "geo",
            Measured("proteomics", False),
            samples=("ctrl 0h", "ctrl 24h", "ctrl 48h"),
            n_samples=3,
        ),
        _models(),
        route="geo",
    )
    assert row["measured"] and row["matched"]
    assert row["via"] == "panel"


def test_a_binding_assay_reads_a_factor_nearer_than_a_transcript_does():
    row = dc.screen_dataset(
        _cand(
            "geo",
            Measured("binding", True),
            samples=("ctrl 0h", "ctrl 24h", "ctrl 48h"),
            n_samples=3,
        ),
        _models(),
        route="geo",
    )
    assert row["via"] == "occupancy"


def test_an_arrayexpress_mirror_collapses_onto_its_geo_original(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    mirror = dc.screen_dataset(
        _cand(
            "biostudies-arrayexpress",
            Measured("expression", True),
            accession="E-GEOD-777",
            factors=("time",),
        ),
        _models(),
        route="ebi",
    )
    original = dc.screen_dataset(
        _cand(
            "geo",
            Measured("expression", True),
            accession="GSE777",
            factors=("time",),
        ),
        _models(),
        route="geo",
    )
    (run / "rows.jsonl").write_text(
        json.dumps(mirror) + "\n" + json.dumps(original) + "\n"
    )
    df = dc.load_rows(run)
    assert len(df) == 1
    # The GEO row survives, because that is the one a reader can open.
    assert df.iloc[0]["source"] == "geo"
    assert dc.original_of("biostudies-arrayexpress", "E-GEOD-777") == (
        "geo",
        "GSE777",
    )
    assert dc.original_of("geo", "GSE777") == ("geo", "GSE777")
