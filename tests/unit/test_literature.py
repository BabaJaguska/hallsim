"""Where a paper says its model lives."""

from hallsim.literature import pointers_in


def test_pointers_keep_an_organisation_and_read_every_forge():
    text = (
        "Code: https://github.com/pclus/neuromorphology. Data at "
        "https://bitbucket.org/3k1m/dimer_model_ad/src/master/ ; the lab "
        "publishes under https://github.com/neuralcodinglab and mirrors to "
        "https://gitlab.com/lab/model.git (10.5281/zenodo.1234567)."
    )
    found = pointers_in(text)
    assert found["github"] == ["pclus/neuromorphology", "neuralcodinglab"]
    assert found["bitbucket"] == ["3k1m/dimer_model_ad"]
    assert found["gitlab"] == ["lab/model"]
    assert found["zenodo"] == ["1234567"]


def test_no_pointers_is_an_empty_dict():
    assert pointers_in("no links here") == {}


class TestRepositoryClassification:
    """A cited repository is a labelled candidate, not a dead end."""

    TREES = {
        "cosbi/focm": ["README.md", "FOCM_model.m", "MM1.m", "run.m"],
        "lab/deposit": ["model.cps", "fit.py", "README.md"],
        "lab/empty": ["LICENSE"],
    }

    def _fake_get_json(self, url, params, timeout):
        if url.endswith("/users/neuralcodinglab/repos"):
            return [{"full_name": "neuralcodinglab/HYPER"}]
        for pointer, files in self.TREES.items():
            if url == f"https://api.github.com/repos/{pointer}":
                return {"default_branch": "main"}
            if url.endswith(f"/repos/{pointer}/git/trees/main"):
                return {"tree": [{"path": f, "type": "blob"} for f in files]}
        raise AssertionError(url)

    def test_kinds_by_what_the_tree_holds(self, monkeypatch):
        from hallsim import literature

        monkeypatch.setattr(literature, "_get_json", self._fake_get_json)
        matlab = literature.classify_repository("cosbi/focm")
        assert (matlab.kind, matlab.format) == ("source:matlab", "matlab")
        assert "FOCM_model.m" in matlab.description
        copasi = literature.classify_repository("lab/deposit")
        assert copasi.kind == "importable:copasi"
        assert literature.classify_repository("lab/empty").kind == "unknown"
        org = literature.classify_repository("neuralcodinglab")
        assert org.kind == "organisation"
        assert "neuralcodinglab/HYPER" in org.description

    def test_cited_repositories_are_grouped_by_paper(self, monkeypatch):
        from hallsim import literature
        from hallsim.discovery import ModelCandidate

        monkeypatch.setattr(literature, "_get_json", self._fake_get_json)
        texts = {
            "PMC1": "code at https://github.com/cosbi/focm",
            "PMC2": "see https://github.com/cosbi/focm and "
            "https://github.com/lab/deposit",
        }
        monkeypatch.setattr(
            literature, "full_text", lambda pmcid, timeout: texts[pmcid]
        )
        papers = [
            ModelCandidate("europepmc", p, p, "paper", "", False)
            for p in texts
        ]
        cited = literature.repositories_cited(papers, limit=5)
        assert [(c.id, by) for c, by in cited] == [
            ("cosbi/focm", ["PMC1", "PMC2"]),
            ("lab/deposit", ["PMC2"]),
        ]
