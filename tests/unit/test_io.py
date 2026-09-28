"""Output conventions: where runs and tracked results go, and the stamp
that says what produced them."""

import re

from hallsim import io


def test_versions_names_the_libraries_and_the_commit():
    v = io.versions()
    for key in ("jax", "diffrax", "equinox", "optax", "hallsim"):
        assert v[key], key
    # A git checkout: `git describe`, which is the tag alone on a release
    # commit ("v0.1.0"), the tag plus distance and short hash after it, or
    # the short hash before any tag; "-dirty" when the tree has edits.
    assert re.fullmatch(
        r"(v\d[\w.]*(-\d+-g[0-9a-f]{7,})?|[0-9a-f]{7,})(-dirty)?",
        v["hallsim_commit"],
    ), v["hallsim_commit"]
    assert v["python"].count(".") == 2
    assert v["platform"]


def test_results_dir_is_tracked_and_outputs_are_not(monkeypatch, tmp_path):
    monkeypatch.setattr(io, "_ROOT", tmp_path)
    assert io.results_dir("census") == tmp_path / "results" / "census"
    assert io.outdir("census") == tmp_path / "outputs" / "census"
    assert (tmp_path / "results" / "census").is_dir()


class TestIdentityStamp:
    """A result carries what produced it, derived from the object rather
    than from the caller's description of it."""

    @staticmethod
    def _comp(extra=False, rate=0.5):
        import equinox as eqx

        from hallsim.composite import Composite
        from hallsim.process import Port, PortRole, Process

        class Decay(Process):
            timescale: float = eqx.field(static=True, default=1.0)
            rate: float = 0.5

            def ports_schema(self):
                return {"x": Port(role=PortRole.EVOLVED, default=4.0)}

            def derivative(self, t, s):
                return {"x": -self.rate * s["x"]}

        procs = {"d": Decay(rate=rate)}
        topo = {"d": {"x": "c/x"}}
        if extra:
            procs["e"] = Decay()
            topo["e"] = {"x": "c/y"}
        return Composite(
            processes=procs, topology=topo, semantic_validation=False
        )

    def test_structure_separates_a_real_change_from_a_claimed_one(self):
        from hallsim.io import identity

        a = identity(self._comp())
        same = identity(self._comp())
        real = identity(self._comp(extra=True))
        assert a["structure"] == same["structure"]
        assert a["structure"] != real["structure"]

    def test_params_and_start_move_independently_of_structure(self):
        from hallsim.io import identity

        base = identity(self._comp())
        dosed = identity(self._comp(rate=0.9))
        started = identity(self._comp().with_initial({"c/x": 10.0}))
        assert dosed["structure"] == base["structure"]
        assert dosed["params"] != base["params"]
        assert started["structure"] == base["structure"]
        assert started["params"] == base["params"]
        assert started["start"] != base["start"]

    def test_two_arms_that_claim_to_differ_and_do_not_are_visible(
        self, tmp_path
    ):
        """The failure this exists for: a table captioned as a sweep whose
        arms were all built the same way."""
        import json

        from hallsim.io import write_results

        out = write_results(
            tmp_path / "arms.json",
            [
                {"arm": "delay off", "composite": self._comp(), "d": 4.04},
                {"arm": "delay on", "composite": self._comp(), "d": 4.00},
            ],
        )
        rows = json.loads(out.read_text())["rows"]
        assert (
            rows[0]["composite"]["structure"]
            == rows[1]["composite"]["structure"]
        )

    def test_the_written_table_records_its_versions(self, tmp_path):
        import json

        from hallsim.io import write_results

        out = write_results(tmp_path / "r.json", [{"x": 1}], note="hello")
        payload = json.loads(out.read_text())
        assert payload["note"] == "hello"
        assert "hallsim_commit" in payload["versions"]
        assert payload["rows"] == [{"x": 1}]
