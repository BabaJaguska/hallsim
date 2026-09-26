"""Dataset search: GEO hits become candidates the loader can be checked
against."""

import gzip
import io

from hallsim import datasets


def _fake_get_json(url, params, timeout):
    if url.endswith("esearch.fcgi"):
        assert "gse[EntryType]" in params["term"]
        return {"esearchresult": {"count": "1", "idlist": ["200248823"]}}
    assert url.endswith("esummary.fcgi") and params["id"] == "200248823"
    return {
        "result": {
            "uids": ["200248823"],
            "200248823": {
                "accession": "GSE248823",
                "title": "Senescence in WI-38",
                "gdstype": "Expression profiling by array",
                "taxon": "Homo sapiens",
                "n_samples": "20",
                "gpl": "17586",
                "summary": "etoposide and rapamycin",
                "samples": [
                    {"title": "WI38_ETOPOSIDE_D00"},
                    {"title": "WI38_ETOPOSIDE_RAPAMYCIN_D07"},
                ],
            },
        }
    }


def test_geo_hits_become_candidates(monkeypatch):
    monkeypatch.setattr(datasets, "_get_json", _fake_get_json)
    (hit,) = datasets.search_for_dataset("etoposide", organism="Homo sapiens")
    assert hit.accession == "GSE248823"
    assert hit.platform == "GPL17586"
    assert hit.n_samples == 20
    assert hit.series_matrix_has_values
    assert hit.samples == (
        "WI38_ETOPOSIDE_D00",
        "WI38_ETOPOSIDE_RAPAMYCIN_D07",
    )


def test_platform_head_and_loader_route(monkeypatch):
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
    assert datasets.loader_route(head) == (
        "the loader reads symbols from column 'gene_assignment'"
    )
    assert "cannot map" in datasets.loader_route(head[["ID", "probeset_id"]])


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

    monkeypatch.setattr(datasets, "_get_json", fake_get_json)
    (hit,) = datasets.search_for_dataset("senescence", organism="Homo sapiens")
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
                        "description": ["Rapamycin over 24 h"],
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

    monkeypatch.setattr(datasets, "_get_json", fake)
    total, (hit,) = datasets.ebi_page(datasets.EBI_DOMAINS["metabolights"])
    assert total == 1 and hit.source == "metabolights"
    assert hit.measured.modality == "metabolomics"
    assert hit.measured.ids == ("chebi:15422", "chebi:16761")
    assert hit.factors == ("time", "treatment") and hit.pubmed == "12345"
    assert hit.short_kind == "metabolomics"


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

    monkeypatch.setattr(datasets, "_get_json", fake)
    paper = datasets.paper_data("16773083")
    assert paper.has_data and paper.has_supplement and paper.pmcid == "PMC1"
    assert paper.biomodels == ("BIOMD0000000157",)
    assert paper.chemicals == (("chebi:15422", "ATP"),)
    kinds = {d.accession: d.measured.modality for d in paper.datasets}
    assert kinds == {"S-EPMC1681500": "supplement", "GSE248823": "expression"}
    assert all(d.pubmed == "16773083" for d in paper.datasets)


def _annotated_composite():
    import equinox as eqx

    from hallsim.composite import Composite
    from hallsim.process import Port, PortRole, Process, ProcessKind

    class P53(Process):
        kind: ProcessKind = ProcessKind.CONTINUOUS
        timescale: float = eqx.field(static=True, default=1.0)

        def ports_schema(self):
            return {
                "p53": Port(
                    role=PortRole.EVOLVED,
                    default=1.0,
                    ontology={"uniprot": "P04637"},
                ),
                "atp": Port(
                    role=PortRole.EVOLVED,
                    default=1.0,
                    ontology={"chebi": "CHEBI:15422"},
                ),
            }

        def derivative(self, t, state):
            return {"p53": -state["p53"], "atp": -state["atp"]}

    return Composite(
        processes={"m": P53()},
        topology={"m": {"p53": "cell/p53", "atp": "cell/atp"}},
        semantic_validation=False,
    )


def _candidate(measured, **kw):
    base = dict(
        source="x",
        accession="A",
        title="",
        kind="",
        organism="",
        n_samples=0,
        platform="",
        url="",
    )
    base.update(kw)
    return datasets.DatasetCandidate(measured=measured, **base)


def test_coverage_joins_measured_quantities_to_store_paths():
    comp = _annotated_composite()
    M = datasets.Measured
    cov = datasets.coverage(
        _candidate(M("metabolomics", False, ("chebi:15422",))), comp
    )
    assert (cov.paths, cov.via) == (("cell/atp",), "direct")
    cov = datasets.coverage(_candidate(M("proteomics", True)), comp)
    assert (cov.paths, cov.via) == (("cell/p53",), "complete")
    cov = datasets.coverage(_candidate(M("expression", True)), comp)
    assert (cov.paths, cov.via) == (("cell/p53",), "regulon")  # p53 is a TF
    assert not datasets.coverage(_candidate(M("imaging")), comp)
    assert not datasets.coverage(
        _candidate(M("metabolomics", False, ("chebi:1",))), comp
    )


def test_perturbations_resolve_structurally(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "_cache_dir", lambda: tmp_path)

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

    monkeypatch.setattr(datasets, "_get_json", fake)
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
        "_get_json",
        lambda *a: (_ for _ in ()).throw(AssertionError("network")),
    )
    assert datasets.resolve_perturbation("rapamycin").chebi == "chebi:9168"


def test_geo_enumeration_splits_a_page_the_converter_refuses(monkeypatch):
    monkeypatch.setattr(datasets.time, "sleep", lambda s: None)
    calls = []

    def fake(url, params, timeout):
        if url.endswith("esearch.fcgi"):
            return {
                "esearchresult": {"count": "7", "webenv": "W", "querykey": "1"}
            }
        start, size = int(params["retstart"]), int(params["retmax"])
        calls.append((start, size))
        if size > 2:
            return {
                "eutilsresult": {"ERROR": "Input XML ... the max size is 10MB"}
            }
        uids = [str(u) for u in range(start, min(start + size, 7))]
        return {
            "result": {
                "uids": uids,
                **{
                    u: {
                        "accession": f"GSE{u}",
                        "gdstype": "Expression profiling by array",
                        "samples": [],
                        "suppfile": "a_counts.txt, b.tar",
                    }
                    for u in uids
                },
            }
        }

    monkeypatch.setattr(datasets, "_get_json", fake)
    offsets = []
    hits = list(
        datasets.iter_geo(("Homo sapiens",), page=4, on_page=offsets.append)
    )
    assert [h.accession for h in hits] == [f"GSE{i}" for i in range(7)]
    assert hits[0].files == ("a_counts.txt", "b.tar")
    assert offsets == [4, 8]
    assert (0, 4) in calls and (
        0,
        1,
    ) in calls  # the refused page was quartered


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
