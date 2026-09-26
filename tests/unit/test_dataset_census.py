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
    # Whether PRIDE deposited a quantification file is not in the
    # enumeration metadata, so the loader is not asserted from the modality.
    assert row["stage"] == "loadable" and row["loader"] == "unchecked"


def test_the_earlier_gates_name_their_reason():
    genotype = dc.screen_dataset(
        _cand("geo", Measured("genotype", True), summary="0, 5, 10 min"),
        _models(),
        route="geo",
    )
    assert genotype["stage"] == "measured"
    assert "genotype" in dc.reason_of(genotype)
    # Two timepoints and two arms is not a time course, but it is a
    # contrast, so it is kept and labelled rather than discarded.
    short = dc.screen_dataset(
        _cand(
            "geo",
            Measured("expression", True),
            samples=(
                "ctrl 0h r1",
                "ctrl 0h r2",
                "ctrl 24h r1",
                "ctrl 24h r2",
                "TNF 0h r1",
                "TNF 0h r2",
                "TNF 24h r1",
                "TNF 24h r2",
            ),
            n_samples=8,
        ),
        _models(),
        route="geo",
    )
    assert not short["dynamics"]
    assert short["contrast"] and short["contrast_kind"] == "course"
    # Kept through matching; it stops at the loader, not at the gate.
    assert short["stage"] == "loadable"
    assert short["perturbed"] and short["control"] == "ctrl"
    lone = dc.screen_dataset(
        _cand(
            "geo",
            Measured("expression", True),
            samples=("one sample",),
            n_samples=1,
        ),
        _models(),
        route="geo",
    )
    assert lone["stage"] == "contrast"
    assert "nothing to divide by" in dc.reason_of(lone)
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
        ["contrast", 2],
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


def test_the_file_list_decides_the_loader_where_a_modality_cannot():
    """ArrayExpress labels deposits processed while mostly shipping raw
    per-sample arrays, and a third ship no data at all, so the modality
    cannot name a reader."""
    raw = _cand(
        "biostudies-arrayexpress",
        Measured("expression", True),
        files=("E-MEXP-1-raw-data-1.txt", "sample1.cel"),
    )
    processed = _cand(
        "biostudies-arrayexpress",
        Measured("expression", True),
        files=("arrayexpress_counts.txt",),
    )
    unknown = _cand("biostudies-arrayexpress", Measured("expression", True))
    assert dc.loader_of(raw) == "none"
    assert dc.loader_of(processed) == "counts-file"
    assert dc.loader_of(unknown) == "unchecked"

    with_mztab = _cand(
        "pride", Measured("proteomics", True), files=("study.mzTab",)
    )
    ids_only = _cand(
        "pride", Measured("proteomics", True), files=("run.mzid", "run.mgf")
    )
    assert dc.loader_of(with_mztab) == "mztab"
    assert dc.loader_of(ids_only) == "none"
    assert "mztab" in dc.READABLE
    assert "unchecked" not in dc.READABLE


def test_a_perturbation_at_one_timepoint_is_a_contrast():
    """Every reader returns a fold change between two groups, so a
    perturbed arm beside a control is usable even with no time axis. Gating
    on three timepoints discarded nine such deposits for every one kept."""
    row = dc.screen_dataset(
        _cand(
            "geo",
            Measured("expression", True),
            samples=("ctrl rep1", "ctrl rep2", "TNF rep1", "TNF rep2"),
            n_samples=4,
        ),
        _models(),
        route="geo",
    )
    assert row["contrast_kind"] == "arms"
    assert row["contrast"] and not row["dynamics"]
    assert row["measured"] and row["matched"]


def test_dynamics_is_recorded_beside_the_contrast():
    row = dc.screen_dataset(
        _cand(
            "geo",
            Measured("expression", True),
            samples=tuple(TITLES),
            n_samples=5,
        ),
        _models(),
        route="geo",
    )
    assert row["contrast_kind"] == "dynamics" and row["dynamics"]


def test_an_arm_per_sample_is_not_a_contrast():
    """Labels that give nearly one arm per sample are parsed tokens, not
    conditions, and an arm holding one sample cannot be contrasted."""
    row = dc.screen_dataset(
        _cand(
            "geo",
            Measured("expression", True),
            samples=("P01 A1", "P02 B2", "P03 C3", "P04 D4"),
            n_samples=4,
        ),
        _models(),
        route="geo",
    )
    assert row["n_arms"] == 4 and row["n_samples"] == 4
    assert row["contrast_kind"] == "" and not row["contrast"]


def test_replication_is_counted_in_subjects_not_samples():
    """Three brain regions from ten mice is ten independent units, not
    thirty: scoring it as thirty is how an effect appears that is not
    there."""
    from hallsim.datasets import Design

    thirty_samples_ten_mice = Design(
        arms=("flight", "ground"),
        per_arm=(("flight", ()), ("ground", ())),
        n_titles=30,
        n_subjects=10,
    )
    assert dc._replicated(thirty_samples_ten_mice)

    six_arms_ten_mice = Design(
        arms=tuple(f"arm{i}" for i in range(6)),
        per_arm=tuple((f"arm{i}", ()) for i in range(6)),
        n_titles=30,
        n_subjects=10,
    )
    # Six arms over ten animals cannot hold two subjects each.
    assert not dc._replicated(six_arms_ten_mice)
    # Counting the thirty samples instead would have let it through.
    assert dc._replicated(
        Design(
            arms=six_arms_ten_mice.arms,
            per_arm=six_arms_ten_mice.per_arm,
            n_titles=30,
        )
    )


def test_deposits_sharing_subjects_are_one_experiment():
    """One mission's animals, assayed per tissue, arrive as one accession
    each; counting them separately multiplies the apparent evidence."""
    groups = dc.shared_subjects(
        {
            "OSD-563": ("RR-10_FL-01", "RR-10_FL-03"),
            "OSD-564": ("RR-10_FL-01", "RR-10_FL-03"),
            "OSD-612": ("RR-10_FL-01", "RR-10_FL-03"),
            "OSD-613": ("RRRM2_A", "RRRM2_B"),
            "GSE1": (),
        }
    )
    assert groups == [("OSD-563", "OSD-564", "OSD-612")]


def test_a_cross_species_match_is_named_not_hidden():
    """A mouse series matched to a human-annotated model goes through an
    ortholog step that the identifiers do not show, so the route says so."""
    human = dc.ModelIds(
        "BIOMD1",
        "",
        frozenset({"P04637"}),
        frozenset(),
        frozenset({"P04637"}),
        "Homo sapiens",
    )
    mouse_data = _cand(
        "geo",
        Measured("expression", True),
        samples=("ctrl 0h", "ctrl 24h", "ctrl 48h"),
        n_samples=3,
        organism="Mus musculus",
    )
    row = dc.screen_dataset(mouse_data, [human], route="geo")
    assert row["via"] == "regulon" and row["species"] == "ortholog"

    human_data = _cand(
        "geo",
        Measured("expression", True),
        samples=("ctrl 0h", "ctrl 24h", "ctrl 48h"),
        n_samples=3,
        organism="Homo sapiens",
    )
    assert (
        dc.screen_dataset(human_data, [human], route="geo")["species"]
        == "same"
    )


def test_a_metabolite_identity_carries_no_species_step():
    """ATP is ATP in every organism, so a ChEBI match needs no ortholog."""
    model = dc.ModelIds(
        "BIOMD1",
        "",
        frozenset(),
        frozenset({"chebi:15422"}),
        frozenset(),
        "Homo sapiens",
    )
    row = dc.screen_dataset(
        _cand(
            "metabolights",
            Measured("metabolomics", False, ("chebi:15422",)),
            factors=("Timepoint",),
            organism="Mus musculus",
        ),
        [model],
        route="ebi",
    )
    assert row["via"] == "direct" and row["species"] == "n/a"


def test_the_same_species_count_is_what_informs_not_the_best_case():
    """A row matching many deposits of which few share its organism has not
    been matched within species; reporting the best case would say it had."""
    models = [
        dc.ModelIds(
            "H1",
            "",
            frozenset({"P04637"}),
            frozenset(),
            frozenset({"P04637"}),
            "Homo sapiens",
        ),
        dc.ModelIds(
            "H2",
            "",
            frozenset({"P04637"}),
            frozenset(),
            frozenset({"P04637"}),
            "Homo sapiens",
        ),
        dc.ModelIds(
            "M1",
            "",
            frozenset({"P04637"}),
            frozenset(),
            frozenset({"P04637"}),
            "Mus musculus",
        ),
    ]
    row = dc.screen_dataset(
        _cand(
            "geo",
            Measured("expression", True),
            samples=("ctrl 0h", "ctrl 24h", "ctrl 48h"),
            n_samples=3,
            organism="Mus musculus",
        ),
        models,
        route="geo",
    )
    assert row["n_models"] == 3
    assert row["n_same_species"] == 1


def test_a_course_with_one_sample_per_cell_is_not_replicated():
    """Two timepoints and two arms is four cells; four samples fill them
    with one each, and one sample per cell is no replicate."""
    row = dc.screen_dataset(
        _cand(
            "geo",
            Measured("expression", True),
            samples=("ctrl 0h", "ctrl 24h", "TNF 0h", "TNF 24h"),
            n_samples=4,
        ),
        _models(),
        route="geo",
    )
    assert row["n_arms"] == 2 and row["n_timepoints"] == 2
    assert row["contrast_kind"] == "" and not row["contrast"]


def test_a_listed_protein_matches_by_accession():
    """A deposit naming UniProt accessions pairs with a model carrying
    them, in one spelling on both sides."""
    row = dc.screen_dataset(
        _cand(
            "pride",
            Measured("proteomics", False, ("uniprot:P04637",)),
            factors=("time",),
        ),
        _models(),
        route="ebi",
    )
    assert row["via"] == "direct"
    assert row["direct_pairs"][0]["shared"] == ["uniprot:p04637"]


def test_imaging_is_measured_but_matches_nothing():
    row = dc.screen_dataset(
        _cand("bioimages", Measured("imaging"), summary="0, 5, 10 min"),
        _models(),
        route="bioimages",
    )
    assert row["measured"] and not row["matched"]
    assert row["via"] == "none"


def test_a_mirror_survives_a_rescreen(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    row = dc.screen_dataset(
        _cand(
            "osdr",
            Measured("expression", True),
            accession="OSD-115",
            mirrors="E-GEOD-12647",
            factors=("time",),
        ),
        _models(),
        route="osdr",
    )
    assert row["mirrors"] == "E-GEOD-12647"
    assert dc.candidate_of(row).mirrors == "E-GEOD-12647"
    (run / "rows.jsonl").write_text(json.dumps(row) + "\n")
    assert dc.rescreen(run, _models()) == 1
    again = json.loads((run / "rows.jsonl").read_text())
    assert again["mirrors"] == "E-GEOD-12647"
    assert dc.original_of("osdr", "OSD-115", "E-GEOD-12647") == (
        "geo",
        "GSE12647",
    )
    assert dc.original_of("osdr", "OSD-9", "E-MTAB-77") == (
        "biostudies-arrayexpress",
        "E-MTAB-77",
    )


def test_osdr_reads_only_genelab_counts():
    """GeneLab's differential-expression table matches the generic
    counts-file pattern and would be read as counts; only the pipeline's
    counts files are a known layout."""
    assert (
        dc._osdr_loader(
            ("GLDS-1_rna_seq_STAR_Unnormalized_Counts_GLbulkRNAseq.csv",)
        )
        == "glbulkrnaseq"
    )
    assert (
        dc._osdr_loader(
            ("GLDS-1_rna_seq_differential_expression_GLbulkRNAseq.csv",)
        )
        == "none"
    )
    assert dc._osdr_loader(()) == "none"


def test_rows_from_earlier_gates_are_refused_not_misread():
    import pandas as pd
    import pytest

    old = pd.DataFrame(
        [{"timed": True, "measured": True, "matched": True, "loadable": True}]
    )
    with pytest.raises(ValueError, match="rescreen"):
        dc.funnel(old)
