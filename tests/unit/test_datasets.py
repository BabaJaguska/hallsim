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
