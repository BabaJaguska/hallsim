"""The intake gate on a deposit from any repository (hallsim.intake)."""

from hallsim import intake
from hallsim.search import models as search_models

# Two states relaxing to 1; nothing to initialise, nothing that grows.
ODE = "par k=1\nA'=k*(1-A)\nB'=A-B\ndone\n"


def test_a_modeldb_ode_is_triaged_through_the_xpp_importer(
    tmp_path, monkeypatch
):
    ode = tmp_path / "toy.ode"
    ode.write_text(ODE)
    asked = []
    monkeypatch.setattr(
        search_models,
        "model_files",
        lambda mid, src, timeout: asked.append((mid, src)) or [ode],
    )
    v = intake.triage_model("266", "modeldb")
    assert asked == [("266", "modeldb")]
    assert v.name == "modeldb:266"
    assert v.status in ("pass", "flag") and v.escalate
    assert v.n_species == 2 and v.n_parameters == 1
    # XPP carries no unit metadata, so the clock is the caller's guess.
    assert any("time unit" in f for f in v.flags)


def test_a_local_ode_file_takes_the_same_route(tmp_path):
    ode = tmp_path / "toy.ode"
    ode.write_text(ODE)
    v = intake.triage_model(str(ode))
    assert v.name == str(ode) and v.n_species == 2


def test_a_rehosted_biomodels_deposit_is_triaged_as_the_original(
    monkeypatch,
):
    seen = []
    monkeypatch.setattr(
        intake,
        "triage_sbml",
        lambda mid, **kw: seen.append(mid)
        or intake.TriageVerdict(name=str(mid), status="pass"),
    )
    v = intake.triage_model("BIOMD0000000582_tellurium_x", "biosimulations")
    assert seen == ["BIOMD0000000582"] and v.escalate
    assert search_models.canonical_deposit(157, "biomodels") == (
        157,
        "biomodels",
    )


def test_a_repository_without_a_fetcher_is_unscreenable_not_an_error():
    v = intake.triage_model("a-project", "biosimulations")
    assert v.status == "unscreenable" and not v.escalate
    assert "biosimulations" in v.blockers[0]


def test_a_fetched_sbml_goes_through_the_sbml_gate_under_its_hit_name(
    tmp_path, monkeypatch
):
    sbml = tmp_path / "m.xml"
    sbml.write_text("<sbml/>")
    monkeypatch.setattr(search_models, "model_files", lambda *a: [sbml])
    given = {}
    monkeypatch.setattr(
        intake,
        "triage_sbml",
        lambda mid, **kw: given.update(mid=mid, **kw)
        or intake.TriageVerdict(name=str(mid), status="flag"),
    )
    v = intake.triage_model("glycolysis1", "jws", t_end=5.0, produces="ATP")
    assert v.name == "jws:glycolysis1" and v.status == "flag"
    assert given["mid"] == str(sbml) and given["t_end"] == 5.0
    assert given["produces"] == "ATP"


def test_a_deposit_with_no_model_file_is_unscreenable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        search_models,
        "model_files",
        lambda *a: [tmp_path / "readme.txt", tmp_path / "run.m"],
    )
    v = intake.triage_model("1", "modeldb")
    assert v.status == "unscreenable"
    assert "readme.txt, run.m" in v.blockers[0]


def test_a_fetch_that_fails_is_unscreenable(monkeypatch):
    def refuse(*a):
        raise OSError("connection refused")

    monkeypatch.setattr(search_models, "model_files", refuse)
    v = intake.triage_model("e/1", "physiome")
    assert v.status == "unscreenable"
    assert v.blockers == ("fetch failed: OSError: connection refused",)
