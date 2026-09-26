"""Literature-mined mechanisms: INDRA statements parsed into cited
mechanisms, and a model's neighbourhood sorted into what it carries and
what it could attach. Offline: the database answers are fixtures."""

from hallsim import mechanisms as mech


def _stmt(kind, source, target, evidence, pmid, text, h):
    role_s, role_t = (
        ("enz", "sub")
        if "phosph" in kind.lower()
        else (
            "subj",
            "obj",
        )
    )
    stmt = {
        "type": kind,
        role_s: {"name": source, "db_refs": {"HGNC": "1"}},
        role_t: {"name": target, "db_refs": {"HGNC": "2"}},
        "belief": 0.9,
        "evidence": [{"text": text, "pmid": pmid}] * min(evidence, 2),
    }
    return h, stmt, evidence


def _payload(*stmts):
    statements = {h: stmt for h, stmt, _ in stmts}
    counts = {h: n for h, _, n in stmts}
    return {"statements": statements, "evidence_counts": counts}


def test_statements_parse_into_cited_mechanisms():
    h, s, _ = _stmt(
        "Dephosphorylation",
        "DUSP6",
        "ERK",
        46,
        "38550381",
        "DUSP6 dephosphorylates ERK1 and ERK2, rendering them inactive.",
        "1",
    )
    complex_h = "2"
    payload = {
        "statements": {
            h: s,
            complex_h: {
                "type": "Complex",
                "members": [{"name": "MAPK1"}, {"name": "DUSP6"}],
                "belief": 0.7,
                "evidence": [{"text": "ERK2 binds MKP-3.", "pmid": "1"}],
            },
        },
        "evidence_counts": {h: 46, complex_h: 3},
    }
    out = mech._parse(payload)
    assert [m.hash for m in out] == [h, complex_h]  # best-attested first
    first = out[0]
    assert (first.kind, first.source, first.target) == (
        "Dephosphorylation",
        "DUSP6",
        "ERK",
    )
    assert first.evidence == 46 and first.pmid == "38550381"
    assert out[1].members == ("MAPK1", "DUSP6") and out[1].agents == (
        "MAPK1",
        "DUSP6",
    )
    assert "PMID 38550381" in str(first)


def test_a_neighbourhood_splits_among_from_outside(monkeypatch):
    """A statement the database returns for two of the model's symbols
    involves both, whatever it named the agents; one returned for a single
    symbol reaches outside the model and is a candidate piece."""
    inside = _stmt(
        "Phosphorylation", "MAP2K1", "ERK", 120, "1", "MEK on ERK", "a"
    )
    outside = _stmt("Dephosphorylation", "DUSP6", "ERK", 46, "2", "DUSP6", "b")
    weak = _stmt("Activation", "X", "MAPK1", 1, "3", "weak", "c")
    # A GO process is not a molecule a process could carry.
    process = _stmt("Activation", "MAPK1", "proliferation", 900, "4", "", "d")
    process[1]["obj"]["db_refs"] = {"GO": "GO:0008283"}
    # "MAPK1 is phosphorylated", by nobody named, cannot be composed.
    headless = _stmt("Phosphorylation", "", "MAPK1", 500, "5", "", "e")
    del headless[1]["enz"]
    answers = {
        "MAPK1": _payload(inside, outside, weak, process, headless),
        "MAP2K1": _payload(inside),
    }
    asked = []

    def fake(params, timeout):
        asked.append(params["agent"])
        return answers[params["agent"]]

    monkeypatch.setattr(mech, "_fetch", fake)
    hood = mech.around(["MAPK1", "MAP2K1"], min_evidence=2)
    assert asked == ["MAPK1", "MAP2K1"]
    assert [m.hash for m in hood.among] == ["a"]
    assert [m.hash for m in hood.outside] == ["b"]  # the weak one is dropped


def test_model_symbols_come_from_uniprot_annotations(monkeypatch):
    from hallsim.process import Port, PortRole, Process

    class Annotated(Process):
        def ports_schema(self):
            return {
                "erk": Port(
                    role=PortRole.EVOLVED,
                    default=0.0,
                    ontology={"uniprot": "P28482"},
                ),
                "mek": Port(
                    role=PortRole.EVOLVED,
                    default=0.0,
                    ontology={"UniProt": "Q02750"},
                ),
                "bare": Port(role=PortRole.EVOLVED, default=0.0),
            }

        def derivative(self, t, state):
            return {"erk": 0.0, "mek": 0.0, "bare": 0.0}

    from hallsim import reporter_wiring

    names = {"P28482": ("MAPK1", None), "Q02750": ("MAP2K1", None)}
    monkeypatch.setattr(reporter_wiring, "_human_symbol", lambda a: names[a])
    assert mech.model_symbols(Annotated()) == ["MAPK1", "MAP2K1"]


def test_a_family_statement_counts_as_among_when_a_member_is_inside(
    monkeypatch,
):
    """INDRA grounds "ERK" to the FamPlex family, not to MAPK1, so a
    statement about ERK is returned only for the other agent's query; the
    vendored FamPlex table folds it back in."""
    family = _stmt("Activation", "EGF", "ERK", 2963, "9", "EGF on ERK", "f")
    family[1]["obj"]["db_refs"] = {"FPLX": "ERK"}
    stranger = _stmt("Activation", "EGF", "STAT3", 800, "8", "", "g")
    answers = {"EGF": _payload(family, stranger), "MAPK1": _payload()}
    monkeypatch.setattr(mech, "_fetch", lambda p, t: answers[p["agent"]])
    hood = mech.around(["MAPK1", "EGF"])
    assert [m.hash for m in hood.among] == ["f"]
    assert [m.hash for m in hood.outside] == ["g"]
    assert "ERK" in mech.families_of()["MAPK1"]
