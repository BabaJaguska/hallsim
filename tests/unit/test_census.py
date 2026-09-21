"""The census classifies what it sees and writes what it counted.

Offline: the gate itself is `hallsim.intake`'s and is tested there. This
covers the parts the census adds — the stage a row stops at, the salvage
class, the record flattening, the listing's paging, and the report files.
"""

import json

import pytest

from hallsim import census


def _row(**kw):
    row = census._blank_row("BIOMD0000000001")
    row["kind"] = "ode"
    row["time_unit_declared"] = True
    row["annotation_coverage"] = 0.9
    row["rest_residual"] = 0.1
    row.update(kw)
    return row


def test_a_clean_row_passes():
    row = _row()
    assert census.stage_of(row) == "pass"
    assert census.salvage_verdict(row) == ("as-is", "")


@pytest.mark.parametrize(
    "kw, stage, salvage",
    [
        (
            {"kind": "qualitative", "kind_note": "SBML-qual"},
            "kinetic",
            "wrong-formalism",
        ),
        ({"kind": "unreadable"}, "kinetic", "deposit-defect"),
        (
            {
                "import_error": "import failed: UnsupportedSBMLFeatureError: algebraic rules are not supported"
            },
            "imports",
            "importer-work",
        ),
        (
            {
                "import_error": "import failed: UnsupportedSBMLFeatureError: reaction 'r1' has no kinetic law"
            },
            "imports",
            "deposit-defect",
        ),
        (
            {"import_error": "import failed: KeyError: 'x'"},
            "imports",
            "framework-defect",
        ),
        (
            {"exploding": True, "framework_suspect": True},
            "solves",
            "framework-defect",
        ),
        ({"exploding": True}, "solves", "needs-review"),
        (
            {"screen_raised": "screen raised: boom"},
            "solves",
            "framework-defect",
        ),
        ({"time_unit_declared": False}, "clock", "cheap-fix"),
        ({"annotation_coverage": 0.2}, "annotated", "cheap-fix"),
        ({"rest_residual": 40.0}, "at_rest", "cheap-fix"),
        (
            {"flags": ["tolerance-sensitive"], "tolerance_sensitive": True},
            "clean",
            "cheap-fix",
        ),
        ({"flags": ["neg"], "negative": True}, "clean", "needs-review"),
        ({"flags": ["grad"], "tunes": False}, "clean", "needs-review"),
        ({"error": "timeout: not screened within 300 s"}, "solves", "timeout"),
        (
            {"error": "RuntimeError: worker died"},
            "imports",
            "framework-defect",
        ),
    ],
)
def test_stage_and_salvage(kw, stage, salvage):
    row = _row(**kw)
    assert census.stage_of(row) == stage
    assert census.salvage_verdict(row)[0] == salvage


def test_every_salvage_class_is_documented():
    assert set(census.SALVAGE) == set(census.SALVAGE_MEANING)


def test_describe_flattens_a_record():
    record = {
        "name": "Zatorsky2006_p53_Model4",
        "description": "<notes><p>The model reproduces Fig 6B.</p><p>To the extent possible under law, ...</p></notes>",
        "publication": {
            "type": "PubMed ID",
            "accession": "16773083",
            "title": "Oscillations and variability in the p53 system.",
            "journal": "Molecular systems biology",
            "year": 2006,
            "synopsis": "Understanding the dynamics. More.",
            "authors": [{"name": "Naama Geva-Zatorsky"}, {"name": "Uri Alon"}],
        },
        "modellingApproach": {"name": "ordinary differential equation model"},
        "modelLevelAnnotations": [
            {"qualifier": "bqbiol:hasTaxon", "name": "Homo sapiens"},
            {"resource": "Gene Ontology", "name": "DNA damage response"},
        ],
        "files": {
            "main": [{"name": "BIOMD0000000157_url.xml"}],
            "additional": [
                {"name": "BIOMD0000000157_url.sedml"},
                {"name": "data.csv"},
            ],
        },
    }
    d = census.describe(record, "BIOMD0000000157")
    assert d["cite"] == "Geva-Zatorsky & Alon 2006"
    assert d["about"] == "Understanding the dynamics."
    assert d["curator_note"] == "The model reproduces Fig 6B."
    assert census._surname("Novak B") == "Novak"
    assert census._surname("Le Novère N") == "Le Novère"
    assert d["taxon"] == "Homo sapiens"
    assert d["has_sedml"] and d["data_files"] == "data.csv"
    assert d["pubmed"] == "16773083"


def test_list_accessions_pages_the_index_and_splits_branches(
    monkeypatch, tmp_path
):
    def page(ids, fmt="SBML"):
        return [{"id": i, "format": fmt, "name": i} for i in ids]

    pages = {
        0: {
            "matches": 250,
            "models": page([f"BIOMD{i:010d}" for i in range(100)]),
        },
        100: {
            "matches": 250,
            "models": page([f"MODEL{i:010d}" for i in range(100)]),
        },
        200: {
            "matches": 250,
            "models": page([f"MODEL{i:010d}" for i in range(100, 140)])
            + page(["MODEL9"], fmt="Python")
            + page([f"BIOMD{i:010d}" for i in range(100, 109)]),
        },
    }
    calls = []

    def fake_get_json(url, params, timeout):
        calls.append(params["offset"])
        return pages[params["offset"]]

    import hallsim.discovery as disc

    monkeypatch.setattr(disc, "_get_json", fake_get_json)
    monkeypatch.setattr(disc, "_cache_dir", lambda: tmp_path)
    everything = census.list_accessions("all", refresh=True)
    assert calls == [0, 100, 200]
    assert len(everything) == 249 and "MODEL9" not in everything
    assert len(census.list_accessions("curated")) == 109
    assert len(census.list_accessions("uncurated")) == 140
    assert census.branch_of("MODEL1") == "uncurated"


def test_write_report_produces_every_file(tmp_path):
    rows = [
        census._finish(_row(), 0.0),
        census._finish(
            _row(
                accession="BIOMD0000000002",
                kind="qualitative",
                kind_note="SBML-qual",
            ),
            0.0,
        ),
        census._finish(
            _row(accession="BIOMD0000000003", time_unit_declared=False), 0.0
        ),
        census._finish(
            _row(
                accession="BIOMD0000000004",
                exploding=True,
                framework_suspect=True,
            ),
            0.0,
        ),
        census._finish(_row(accession="MODEL0000000001"), 0.0),
        census._finish(
            _row(accession="MODEL0000000002", tunes=False, flags=["g"]), 0.0
        ),
    ]
    (tmp_path / "rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    deps = [
        {
            "accession": r["accession"],
            "name": f"m{i}",
            "cite": "A et al. 2000",
            "about": "x",
            "has_sedml": i % 2 == 0,
            "data_files": "",
        }
        for i, r in enumerate(rows)
    ]
    (tmp_path / "deposits.jsonl").write_text(
        "\n".join(json.dumps(d) for d in deps) + "\n"
    )
    out = census.write_report(tmp_path)
    for name in (
        "census.csv",
        "summary.json",
        "failures.md",
        "funnel.png",
        "salvage.png",
        "writeup.md",
    ):
        assert (tmp_path / name).exists(), name
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert dict(summary["funnel"])["listed"] == 6
    assert dict(summary["funnel"])["kinetic"] == 5
    assert summary["pass"] == 2
    assert summary["branches"]["curated"]["pass"] == 1
    assert summary["branches"]["uncurated"]["n"] == 2
    text = out.read_text()
    assert "1 of 4 curated" in text and "1 of 2 uncurated" in text
    assert "framework-defect" in text
    assert "BIOMD0000000004" in (tmp_path / "failures.md").read_text()


def test_a_non_sbml_deposit_is_wrong_formalism_not_a_broken_file(tmp_path):
    row = census._finish(
        _row(accession="BIOMD0000001066", kind="unreadable"), 0.0
    )
    assert row["salvage"] == "deposit-defect"
    (tmp_path / "rows.jsonl").write_text(json.dumps(row) + "\n")
    dep = {
        "accession": "BIOMD0000001066",
        "name": "onnx",
        "format": "Open Neural Network Exchange",
    }
    (tmp_path / "deposits.jsonl").write_text(json.dumps(dep) + "\n")
    df = census.load_rows(tmp_path)
    assert df.loc[0, "stage"] == "kinetic"
    assert df.loc[0, "salvage"] == "wrong-formalism"
    assert "not SBML" in df.loc[0, "how"]


def test_cobra_style_export_is_constraint_based(tmp_path):
    import libsbml

    doc = libsbml.SBMLDocument(2, 4)
    model = doc.createModel()
    c = model.createCompartment()
    c.setId("c")
    c.setSize(1.0)
    for sid in ("A", "B"):
        sp = model.createSpecies()
        sp.setId(sid)
        sp.setCompartment("c")
        sp.setInitialConcentration(1.0)
    for rid in ("R1", "R2"):
        rx = model.createReaction()
        rx.setId(rid)
        rx.createReactant().setSpecies("A")
        rx.createProduct().setSpecies("B")
        kl = rx.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula("FLUX_VALUE"))
        for pid in ("LOWER_BOUND", "UPPER_BOUND", "FLUX_VALUE"):
            prm = kl.createParameter()
            prm.setId(pid)
            prm.setValue(0.0)
    kind, note = census._formalism(model)
    assert kind == "constraint-based" and "COBRA" in note
