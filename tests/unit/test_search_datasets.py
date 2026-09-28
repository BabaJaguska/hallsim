"""Dataset search: GEO hits become candidates the loader can be checked
against."""

import gzip
import io

from Bio.Entrez.Parser import IntegerElement

from hallsim.search import datasets, fetch

GSE248823 = {
    "Accession": "GSE248823",
    "title": "Senescence in WI-38",
    "gdsType": "Expression profiling by array",
    "taxon": "Homo sapiens",
    "n_samples": 20,
    "GPL": "17586",
    "summary": "etoposide and rapamycin",
    "suppFile": "CEL",
    "PubMedIds": [IntegerElement(38409325, "Item", {}, None)],
    "Samples": [
        {"Accession": "GSM1", "Title": "WI38_ETOPOSIDE_D00"},
        {"Accession": "GSM2", "Title": "WI38_ETOPOSIDE_RAPAMYCIN_D07"},
    ],
}


def _fake_esearch(term, *, db="gds", history=False, retmax=25):
    assert "gse[EntryType]" in term
    return {"Count": "1", "IdList": ["200248823"]}


def _fake_esummary(*, db="gds", ids=None, **kw):
    assert list(ids) == ["200248823"]
    return [GSE248823]


def test_geo_hits_become_candidates(monkeypatch):
    monkeypatch.setattr(datasets, "esearch", _fake_esearch)
    monkeypatch.setattr(datasets, "esummary", _fake_esummary)
    (hit,) = datasets.search_for_dataset(
        "etoposide", organism="Homo sapiens", sources=["geo"]
    )
    assert hit.accession == "GSE248823"
    assert hit.platform == "GPL17586"
    assert hit.n_samples == 20 and hit.pubmed == "38409325"
    assert hit.series_matrix_has_values and hit.files == ("CEL",)
    assert hit.samples == (
        "WI38_ETOPOSIDE_D00",
        "WI38_ETOPOSIDE_RAPAMYCIN_D07",
    )


def test_the_default_sources_are_the_index_and_what_it_lacks(monkeypatch):
    asked = []

    def source(name):
        def search(query, limit=25, **kw):
            asked.append(name)
            return []

        return search

    monkeypatch.setattr(
        datasets, "SOURCES", {n: source(n) for n in datasets.SOURCES}
    )
    datasets.search_for_dataset("x")
    assert asked == list(datasets.DEFAULT_SOURCES)
    assert asked[0] == "omicsdi"
    asked.clear()
    datasets.search_for_dataset("x", sources=list(datasets.SOURCES))
    assert set(asked) == set(datasets.SOURCES)


def test_platform_head_reads_the_table_head(monkeypatch):
    soft = (
        "^SERIES = GSE1\n^PLATFORM = GPL1\n!platform_table_begin\n"
        "ID\tprobeset_id\tgene_assignment\n"
        + "".join(
            f"TC0100000{i}.hg.1\tTC0100000{i}.hg.1\tNM_{i} // GENE{i} // x\n"
            for i in range(8)
        )
        + "!platform_table_end\n^SAMPLE = GSM1\n"
    )
    monkeypatch.setattr(
        datasets.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(gzip.compress(soft.encode())),
    )
    head = datasets.platform_head("GSE1")
    assert list(head.columns) == ["ID", "probeset_id", "gene_assignment"]
    assert len(head) == 8


def test_zenodo_records_become_candidates_with_their_files(monkeypatch):

    def fake_get_json(url, params, timeout):
        if "zenodo" in url:
            assert params["type"] == "dataset"
            assert params["q"] == "senescence Homo sapiens"
            return {
                "hits": {
                    "total": 1,
                    "hits": [
                        {
                            "id": 18008608,
                            "doi_url": "https://doi.org/10.5281/zenodo.18008608",
                            "metadata": {
                                "doi": "10.5281/zenodo.18008608",
                                "title": "Monocyte activation program",
                                "description": "<p>Bulk RNA-seq <b>counts</b></p>",
                                "resource_type": {"title": "Dataset"},
                            },
                            "files": [
                                {"key": "counts.csv", "size": 6263127},
                                {"key": "meta.csv", "size": 5760},
                            ],
                        }
                    ],
                }
            }
        return {"esearchresult": {"count": "0", "idlist": []}}

    monkeypatch.setattr(datasets, "get_json", fake_get_json)
    (hit,) = datasets.search_for_dataset(
        "senescence", organism="Homo sapiens", sources=["zenodo"]
    )
    assert hit.source == "zenodo"
    assert hit.accession == "10.5281/zenodo.18008608"
    assert hit.url == "https://doi.org/10.5281/zenodo.18008608"
    assert hit.files == ("counts.csv", "meta.csv")
    assert hit.summary == "Bulk RNA-seq counts"
    assert not hit.series_matrix_has_values


GSE248823_TITLES = [
    "WI38_RNA_PHARM_INHIB_RAS_D00_REP2",
    "WI38_RNA_PHARM_INHIB_RAS_D07_REP1",
    "WI38_RNA_PHARM_INHIB_RAS_DMOG_D04_REP2",
    "WI38_RNA_PHARM_INHIB_ETOPOSIDE_D14_REP2",
    "WI38_RNA_PHARM_INHIB_ETOPOSIDE_RAPAMYCIN_D14_REP1",
    "WI38_RNA_PHARM_INHIB_RAS_D04_REP1",
    "WI38_RNA_PHARM_INHIB_ETOPOSIDE_RAPAMYCIN_D07_REP1",
    "WI38_RNA_PHARM_INHIB_ETOPOSIDE_RAPAMYCIN_D14_REP2",
    "WI38_RNA_PHARM_INHIB_ETOPOSIDE_D07_REP2",
    "WI38_RNA_PHARM_INHIB_ETOPOSIDE_D00_REP1",
    "WI38_RNA_PHARM_INHIB_ETOPOSIDE_D14_REP1",
    "WI38_RNA_PHARM_INHIB_ETOPOSIDE_D00_REP2",
    "WI38_RNA_PHARM_INHIB_ETOPOSIDE_RAPAMYCIN_D07_REP2",
    "WI38_RNA_PHARM_INHIB_RAS_D07_REP2",
    "WI38_RNA_PHARM_INHIB_RAS_DMOG_D07_REP1",
    "WI38_RNA_PHARM_INHIB_ETOPOSIDE_D07_REP1",
    "WI38_RNA_PHARM_INHIB_RAS_DMOG_D04_REP1",
    "WI38_RNA_PHARM_INHIB_RAS_DMOG_D07_REP2",
    "WI38_RNA_PHARM_INHIB_RAS_D04_REP2",
    "WI38_RNA_PHARM_INHIB_RAS_D00_REP1",
]


def test_design_of_the_calibration_series():
    # GSE248823 as deposited: four arms, days in the D00 form, replicates,
    # a shared prefix, and no arm called control (the baseline is day 0).
    d = datasets.parse_design(GSE248823_TITLES)
    assert d.arms == ("ETOPOSIDE", "ETOPOSIDE RAPAMYCIN", "RAS", "RAS DMOG")
    assert dict(d.per_arm)["ETOPOSIDE"] == (0.0, 7.0, 14.0)
    assert d.time_unit == "d" and d.n_timepoints == 3
    assert d.perturbed and d.time_course and d.control is None
    assert d.summary() == "4 arms, 3 timepoints d"


def test_design_reads_controls_units_and_single_arms():
    d = datasets.parse_design(
        ["ctrl 30 min", "ctrl 2 h", "ctrl 24 h", "drug 30 min", "drug 24 h"]
    )
    assert d.control == "ctrl" and d.time_unit == "h"
    assert d.timepoints == (0.5, 2.0, 24.0)
    d = datasets.parse_design(["WT_rep1", "WT_rep2", "Tp53_KO_rep1"])
    assert d.arms == ("Tp53 KO", "WT") and d.control == "WT"
    assert not d.time_course
    d = datasets.parse_design(["diff day0", "diff day2", "diff day4"])
    assert d.arms == ("",) and d.time_course and not d.perturbed
    assert datasets.parse_design([]).summary() == "no sample titles"


def test_geo_types_map_to_a_modality():
    m = datasets.geo_measured("Expression profiling by high throughput seq")
    assert (m.modality, m.complete) == ("expression", True)
    assert datasets.geo_measured(
        "Methylation profiling by array"
    ).modality == ("methylation")


#: An abstract that names its timepoints only after its 400th character.
LONG_DESCRIPTION = (
    "Rapamycin acts through mTORC1 to reshape the metabolome of dividing "
    "cells, and the response unfolds over a scale of minutes rather than "
    "the hours most profiling studies resolve. We profiled polar "
    "metabolites in HeLa cells after a single dose, with matched vehicle "
    "controls, using untargeted LC-MS on a QTOF instrument with internal "
    "standards for the central carbon intermediates and nucleotides, and "
    "asked which pools moved first. Cells were sampled at 0, 5, 15, 30 and "
    "60 min after treatment, in triplicate."
)
assert len(LONG_DESCRIPTION) > 400
assert datasets.time_values(LONG_DESCRIPTION[:400]) == set()


def test_ebi_search_entries_become_candidates(monkeypatch):
    def fake(url, params, timeout):
        assert url.endswith("/metabolights") and params["query"] == "*:*"
        return {
            "hitCount": 1,
            "entries": [
                {
                    "id": "MTBLS1",
                    "fields": {
                        "name": ["A metabolomics time course"],
                        "description": [LONG_DESCRIPTION],
                        "organism": ["Homo sapiens"],
                        "study_factor": ["time", "treatment"],
                        "study_design": [],
                        "technology_type": ["mass spectrometry"],
                        "CHEBI": ["CHEBI:15422", "CHEBI:16761"],
                        "PUBMED": ["12345"],
                    },
                }
            ],
        }

    monkeypatch.setattr(datasets, "get_json", fake)
    total, (hit,) = datasets.ebi_page(datasets.EBI_DOMAINS["metabolights"])
    assert total == 1 and hit.source == "metabolights"
    assert hit.measured.modality == "metabolomics"
    assert hit.measured.ids == ("chebi:15422", "chebi:16761")
    assert hit.factors == ("time", "treatment") and hit.pubmed == "12345"
    assert hit.short_kind == "metabolomics"
    # kept whole: the time-course check reads the tail of an abstract
    assert hit.summary == LONG_DESCRIPTION
    assert datasets.time_values(hit.summary) == {0.0, 5.0, 15.0, 30.0, 60.0}


def test_a_paper_lists_its_own_data(monkeypatch):
    def fake(url, params, timeout):
        if url.endswith("/search"):
            return {
                "resultList": {
                    "result": [
                        {"pmcid": "PMC1", "hasData": "Y", "hasSuppl": "Y"}
                    ]
                }
            }
        assert url.endswith("/MED/16773083/datalinks")
        link = lambda cat, ident, title="": {  # noqa: E731
            "Name": cat,
            "Section": [
                {
                    "Linklist": {
                        "Link": [
                            {
                                "Target": {
                                    "Identifier": {"ID": ident},
                                    "Title": title,
                                }
                            }
                        ]
                    }
                }
            ],
        }
        return {
            "dataLinkList": {
                "Category": [
                    link(
                        "BioModels",
                        "http://identifiers.org/biomodels.db/BIOMD0000000157",
                    ),
                    link(
                        "BioStudies: supplemental material",
                        "http://www.ebi.ac.uk/biostudies/studies/S-EPMC1681500?xr=true",
                    ),
                    link("Data Citations", "GSE248823", "a series"),
                    link("Chemicals", "CHEBI:15422", "ATP"),
                ]
            }
        }

    monkeypatch.setattr(datasets, "get_json", fake)
    paper = datasets.paper_data("16773083")
    assert paper.has_data and paper.has_supplement and paper.pmcid == "PMC1"
    assert paper.biomodels == ("BIOMD0000000157",)
    assert paper.chemicals == (("chebi:15422", "ATP"),)
    kinds = {d.accession: d.measured.modality for d in paper.datasets}
    assert kinds == {"S-EPMC1681500": "supplement", "GSE248823": "expression"}
    assert all(d.pubmed == "16773083" for d in paper.datasets)


def test_perturbations_resolve_structurally(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch, "CACHE_ROOT", tmp_path)

    def fake(url, params, timeout):
        if url == datasets.OLS_SEARCH:
            return {
                "response": {
                    "docs": [{"obo_id": "CHEBI:9168", "label": "sirolimus"}]
                }
            }
        if url.endswith("/molecule.json"):
            return {"molecules": [{"molecule_chembl_id": "CHEMBL413"}]}
        if url.endswith("/mechanism.json"):
            return {
                "mechanisms": [
                    {
                        "mechanism_of_action": "FKBP1A inhibitor",
                        "target_chembl_id": "CHEMBL1902",
                    }
                ]
            }
        if url.endswith("/target/CHEMBL1902.json"):
            return {"target_components": [{"accession": "P62942"}]}
        if url == datasets.MYGENE_QUERY:
            return {"hits": [{"uniprot": {"Swiss-Prot": "P04637"}}]}
        raise AssertionError(url)

    monkeypatch.setattr(datasets, "get_json", fake)
    p = datasets.resolve_perturbation("rapamycin")
    assert (p.kind, p.chebi, p.targets) == (
        "compound",
        "chebi:9168",
        ("uniprot:P62942",),
    )
    assert p.mechanism == "FKBP1A inhibitor" and p.name == "sirolimus"
    g = datasets.resolve_perturbation("TP53 KO")
    assert (g.kind, g.name, g.targets) == ("gene", "TP53", ("uniprot:P04637",))
    # cached: a second call makes no request
    monkeypatch.setattr(
        datasets,
        "get_json",
        lambda *a: (_ for _ in ()).throw(AssertionError("network")),
    )
    assert datasets.resolve_perturbation("rapamycin").chebi == "chebi:9168"


def test_geo_enumeration_pages_through_the_history_server(monkeypatch):
    calls = []

    def esearch(term, *, db="gds", history=False, retmax=25):
        assert history and "gse[EntryType]" in term
        return {"Count": "7", "WebEnv": "W", "QueryKey": "1"}

    def esummary(
        *,
        db="gds",
        ids=None,
        webenv=None,
        query_key=None,
        retstart=0,
        retmax=300,
    ):
        assert (webenv, query_key) == ("W", "1")
        calls.append((retstart, retmax))
        return [
            {
                "Accession": f"GSE{u}",
                "gdsType": "Expression profiling by array",
                "Samples": [],
                "suppFile": "a_counts.txt, b.tar",
            }
            for u in range(retstart, min(retstart + retmax, 7))
        ]

    monkeypatch.setattr(datasets, "esearch", esearch)
    monkeypatch.setattr(datasets, "esummary", esummary)
    offsets = []
    hits = list(
        datasets.iter_geo(("Homo sapiens",), page=4, on_page=offsets.append)
    )
    assert [h.accession for h in hits] == [f"GSE{i}" for i in range(7)]
    assert hits[0].files == ("a_counts.txt", "b.tar")
    assert offsets == [4, 8] and calls == [(0, 4), (4, 4)]


def test_omicsdi_hits_are_the_deposits_the_index_lists(monkeypatch):
    def fake(url, params, timeout):
        assert (
            url.endswith("/dataset/search") and params["query"] == "rapamycin"
        )
        return {
            "count": 3,
            "datasets": [
                {
                    "id": "GSE21755",
                    "source": "geo",
                    "title": "Rapamycin: time course",
                    "description": "mTOR",
                    "organisms": [{"name": "Mus musculus"}],
                    "omicsType": ["Transcriptomics"],
                },
                {
                    "id": "MTBLS1",
                    "source": "metabolights_dataset",
                    "title": "a metabolome",
                    "description": "",
                    "organisms": [{"name": "Homo sapiens"}],
                    "omicsType": ["Metabolomics"],
                },
                {
                    "id": "BIOMD0000000001",
                    "source": "biomodels",
                    "title": "a model",
                    "description": "",
                    "organisms": [],
                    "omicsType": ["Models"],
                },
            ],
        }

    monkeypatch.setattr(datasets, "get_json", fake)
    monkeypatch.setattr(
        datasets,
        "geo_by_accession",
        lambda acc, timeout=30.0: datasets.DatasetCandidate(
            source="geo",
            accession=acc,
            title="from GEO",
            kind="Expression profiling by array",
            organism="Mus musculus",
            n_samples=25,
            platform="GPL1261",
            url="",
            samples=(
                "wt 0h",
                "wt 6h",
                "wt 24h",
                "tsc 0h",
                "tsc 6h",
                "tsc 24h",
            ),
        ),
    )
    hits = datasets.search_omicsdi("rapamycin")
    assert [(h.source, h.accession) for h in hits] == [
        ("geo", "GSE21755"),
        ("metabolights", "MTBLS1"),
    ]  # the model is left to the model search
    assert hits[0].title == "from GEO" and hits[0].design.time_course
    assert hits[1].measured.modality == "metabolomics"
    assert "omicsdi.org/dataset/metabolights_dataset/MTBLS1" in hits[1].url
    only = datasets.search_omicsdi("rapamycin", organism="Homo sapiens")
    assert [h.accession for h in only] == ["MTBLS1"]


def test_pride_files_come_through_ppx(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch, "CACHE_ROOT", tmp_path)

    class _Project:
        def remote_files(self):
            return ["a.raw", "a.mzTab", "a.raw"]

    seen = {}

    def find_project(accession, local=None):
        seen["accession"], seen["local"] = accession, local
        return _Project()

    monkeypatch.setattr(datasets.ppx, "find_project", find_project)
    assert datasets.pride_files("PXD1") == ("a.raw", "a.mzTab")
    assert seen["accession"] == "PXD1" and "PXD1" in str(seen["local"])


def test_a_per_sample_identifier_is_not_an_arm():
    # Animal and accession tokens are unique per sample, so keeping them
    # made every sample its own arm and no arm held a time course.
    titles = [
        "MUC26513_young_control_d0",
        "MUC26536_young_control_d0",
        "MUC26533_young_bleo_d10",
        "MUC26541_young_bleo_d21",
        "MUC26550_old_control_d0",
        "MUC26562_old_bleo_d10",
        "MUC26571_old_bleo_d21",
    ]
    d = datasets.parse_design(titles)
    assert d.arms == ("old bleo", "old control", "young bleo", "young control")
    assert dict(d.per_arm)["young bleo"] == (10.0, 21.0)
    assert d.time_unit == "d"


def test_identifier_pruning_leaves_a_real_design_alone():
    # Every token here partitions the samples, so nothing may be dropped.
    d = datasets.parse_design(
        ["ctrl 0h", "ctrl 6h", "ctrl 24h", "drug 0h", "drug 6h", "drug 24h"]
    )
    assert d.arms == ("ctrl", "drug") and d.control == "ctrl"
    assert d.timepoints == (0.0, 6.0, 24.0)
    # Too few samples to judge frequency: left untouched rather than merged.
    pair = datasets.parse_design(["a_x_1", "b_y_2"])
    assert len(pair.arms) == 2


def test_a_series_record_names_its_supplementary_files():
    """E-utilities gives only file types; the series' brief SOFT record
    names each file. An unknown accession answers with an HTML page and
    status 200, which must not be read as "no files"."""
    soft = (
        "^SERIES = GSE67270\n"
        "!Series_type = Expression profiling by high throughput sequencing\n"
        "!Series_supplementary_file = ftp://ftp.ncbi.nlm.nih.gov/geo/series/"
        "GSE67nnn/GSE67270/suppl/GSE67270_HP32R_FPKM.txt.gz\n"
        "!Series_supplementary_file = ftp://ftp.ncbi.nlm.nih.gov/geo/series/"
        "GSE67nnn/GSE67270/suppl/GSE67270_RAW.tar\n"
    )
    assert datasets.supplementary_names(soft) == (
        "GSE67270_HP32R_FPKM.txt.gz",
        "GSE67270_RAW.tar",
    )
    assert (
        datasets.supplementary_names("^SERIES = GSE1\n!Series_type = x\n")
        == ()
    )
    try:
        datasets.supplementary_names("<html><title>Not found</title></html>")
    except LookupError:
        pass
    else:
        raise AssertionError("an HTML page was read as a series record")
    assert datasets._FILE_LISTERS["geo"] is datasets.geo_files


# --- designs a source states outright -------------------------------------


def test_a_design_survives_a_round_trip_through_a_dict():
    d = datasets.Design(
        arms=("ctrl", "drug"),
        control="ctrl",
        per_arm=(("ctrl", (0.0, 6.0)), ("drug", (0.0, 6.0, 24.0))),
        time_unit="h",
        n_titles=6,
        n_subjects=3,
    )
    assert datasets.Design.from_dict(d.to_dict()) == d


def test_a_stated_design_outranks_the_titles():
    stated = datasets.Design(arms=("x",), per_arm=(("x", (0.0, 1.0, 2.0)),))
    cand = datasets.DatasetCandidate(
        source="petab",
        accession="A",
        title="",
        kind="",
        organism="",
        n_samples=0,
        platform="",
        url="",
        samples=("a_d0", "a_d7"),
        stated=stated,
    )
    assert cand.design is stated and cand.design.time_course


def test_a_long_table_with_a_time_column_states_its_design():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "Time (h)": [0, 0, 6, 6, 24, 24],
            "Treatment": ["ctrl", "drug"] * 3,
            "value": [1.0, 1.1, 2.0, 3.5, 2.5, 6.0],
        }
    )
    d = datasets.design_of_table(frame)
    assert d.arms == ("ctrl", "drug") and d.control == "ctrl"
    assert dict(d.per_arm)["drug"] == (0.0, 6.0, 24.0)
    assert d.time_unit == "h" and d.n_titles == 6 and d.time_course


def test_a_wide_table_is_one_unperturbed_arm():
    import pandas as pd

    frame = pd.DataFrame(
        {"t": [0, 1, 2, 3], "A": [1, 2, 3, 4], "B": [4, 3, 2, 1]}
    )
    d = datasets.design_of_table(frame)
    assert d.arms == ("",) and d.timepoints == (0.0, 1.0, 2.0, 3.0)
    assert d.time_course and not d.perturbed
    assert datasets.design_of_table(frame[["A", "B"]]) is None


def test_the_richest_table_states_the_design(tmp_path):
    (tmp_path / "meta.csv").write_text("sample,group\ns1,a\ns2,b\n")
    (tmp_path / "course.tsv").write_text(
        "time_min\tarm\tv\n0\tctrl\t1\n10\tctrl\t2\n30\tctrl\t3\n"
        "0\tdrug\t1\n10\tdrug\t4\n30\tdrug\t9\n"
    )
    d = datasets.table_design(sorted(tmp_path.iterdir()))
    assert d.arms == ("ctrl", "drug") and d.time_unit == "min"
    assert d.n_timepoints == 3


# --- GEO DataSets: a curator typed the subsets ----------------------------


def test_gds_subsets_state_arms_times_and_membership():
    subsets = [
        ("time", "hour 6", ("s1", "s2", "s4", "s5")),
        ("time", "hour 24", ("s3", "s6")),
        ("infection", "H5N1 (MOI 1)", ("s4", "s5", "s6")),
        ("infection", "control", ("s1", "s2", "s3")),
    ]
    d = datasets.gds_design(subsets)
    assert d.arms == ("H5N1 (MOI 1)", "control") and d.control == "control"
    assert dict(d.per_arm)["H5N1 (MOI 1)"] == (6.0, 24.0)
    assert d.time_unit == "h" and d.n_titles == 6


def test_a_categorical_age_subset_is_an_arm_not_a_timepoint():
    subsets = [
        ("age", "young", ("s1", "s2", "s3", "s4")),
        ("age", "middle age", ("s5", "s6", "s7", "s8")),
    ]
    d = datasets.gds_design(subsets)
    assert d.arms == ("middle age", "young") and d.n_titles == 8
    assert not d.time_course and d.perturbed


def test_a_dataset_candidate_is_the_series_it_curates(monkeypatch):
    monkeypatch.setattr(datasets, "_geo_throttle", lambda: None)
    monkeypatch.setattr(
        datasets,
        "gds_subsets",
        lambda acc, timeout=60.0: [
            ("time", "hour 6", ("s1", "s3")),
            ("time", "hour 12", ("s2", "s4")),
            ("time", "hour 24", ("s5", "s6")),
            ("infection", "H5N1", ("s3", "s4", "s6")),
            ("infection", "control", ("s1", "s2", "s5")),
        ],
    )
    records = [
        {
            "Accession": "GDS6010",
            "GSE": "66597",
            "title": "H5N1 infection of astrocytes: time course",
            "gdsType": "Expression profiling by array",
            "taxon": "Homo sapiens",
            "n_samples": 18,
            "GPL": "6480",
            "SSInfo": "time;time;time;infection;infection",
            "PubMedIds": [26008703],
        }
    ]
    (hit,) = datasets._gds_candidates(records)
    assert (hit.source, hit.accession, hit.curated) == (
        "geo",
        "GSE66597",
        "GDS6010",
    )
    assert hit.factors == ("time", "infection") and hit.pubmed == "26008703"
    assert hit.series_matrix_has_values
    assert hit.design.time_course and hit.design.control == "control"
    assert dict(hit.design.per_arm)["H5N1"] == (6.0, 12.0, 24.0)


# --- Expression Atlas: factor values per assay group or contrast ----------


def test_atlas_baseline_groups_state_arms_and_times():
    record = {
        "columnHeaders": [
            {
                "assayGroupSummary": {
                    "replicates": 3,
                    "properties": [
                        {
                            "propertyName": "age",
                            "testValue": f"{m} month",
                            "contrastPropertyType": "FACTOR",
                        },
                        {
                            "propertyName": "compound",
                            "testValue": arm,
                            "contrastPropertyType": "FACTOR",
                        },
                        {
                            "propertyName": "genotype",
                            "testValue": "wild type",
                            "contrastPropertyType": "SAMPLE",
                        },
                    ],
                }
            }
            for arm in ("control", "rapamycin")
            for m in (6, 12, 24)
        ]
    }
    d = datasets.atlas_design(record)
    assert d.arms == ("control", "rapamycin") and d.control == "control"
    assert dict(d.per_arm)["rapamycin"] == (6.0, 12.0, 24.0)
    assert d.time_unit == "mo" and d.n_titles == 18


def test_atlas_contrasts_state_both_sides_and_the_time():
    record = {
        "columnHeaders": [
            {
                "displayName": f"'H5N1' vs 'mock' in 'A549' at '{h} hour'",
                "testAssayGroup": {"replicates": 3},
                "referenceAssayGroup": {"replicates": 3},
            }
            for h in (6, 12, 24)
        ]
    }
    d = datasets.atlas_design(record)
    assert d.arms == ("H5N1 A549", "mock A549") and d.control == "mock A549"
    assert dict(d.per_arm)["H5N1 A549"] == (6.0, 12.0, 24.0)
    assert d.time_unit == "h"
    assert datasets.atlas_design({"columnHeaders": []}) is None


def test_atlas_candidates_point_at_the_deposit_they_curate():
    def rec(acc, tech):
        return {
            "experimentAccession": acc,
            "experimentDescription": "a study",
            "species": "Homo sapiens",
            "technologyType": [tech],
            "numberOfAssays": 12,
            "experimentalFactors": ["time", "compound"],
        }

    geo = datasets._atlas_candidate(
        rec("E-GEOD-147507", "RNA-Seq mRNA"), designs=False
    )
    assert (geo.source, geo.accession, geo.curated) == (
        "geo",
        "GSE147507",
        "E-GEOD-147507",
    )
    assert geo.short_kind == "rna-seq" and geo.factors == ("time", "compound")
    ae = datasets._atlas_candidate(
        rec("E-MTAB-12011", "microarray"), designs=False
    )
    assert (
        ae.source == "biostudies-arrayexpress" and ae.series_matrix_has_values
    )
    own = datasets._atlas_candidate(
        rec("E-PROT-82", "Proteomics"), designs=False
    )
    assert own.source == "expression-atlas"
    assert own.measured.modality == "proteomics"


# --- BioStudies: the factors the EBI Search index leaves out --------------


def test_biostudies_factors_read_the_study_attributes(monkeypatch):
    monkeypatch.setattr(
        datasets,
        "biostudies_study",
        lambda acc, timeout=60.0: {
            "section": {
                "attributes": [{"name": "Title", "value": "x"}],
                "subsections": [
                    {
                        "attributes": [
                            {
                                "name": "Experimental Designs",
                                "value": "time series design",
                            },
                            {"name": "Experimental Factors", "value": "time"},
                            {
                                "name": "Experimental Factors",
                                "value": "compound",
                            },
                        ]
                    }
                ],
            }
        },
    )
    assert datasets.biostudies_factors("E-MTAB-12011") == (
        "time series design",
        "time",
        "compound",
    )


def test_every_data_source_is_registered():
    assert set(datasets.SOURCES) == {
        "omicsdi",
        "geo",
        "geo-datasets",
        "expression-atlas",
        "petab",
        "zenodo",
        "pride",
        "metabolights",
        "metabolomics-workbench",
        "arrayexpress",
        "bioimages",
    }


def test_the_reference_arm_is_found_without_a_control_word():
    """A dose of none and a label the others extend both name the baseline;
    an old-against-young contrast names neither, and says so."""
    from hallsim.search.datasets import reference_arm

    assert reference_arm(("control", "treated")) == "control"
    assert reference_arm(("a", "b"), {"a": {0.0}, "b": {10.0}}) == "a"
    assert reference_arm(("WT", "WT plus drug")) == "WT"
    assert reference_arm(("Young", "Old")) is None
    assert reference_arm(("only one",)) is None
