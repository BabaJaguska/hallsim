"""XPPAUT files as they arrive in the wild, not as the format documents them.

Each case here cost a real ModelDB import: a macOS resource fork with a .ode
suffix, `%` used as a comment, and keyword abbreviations the parser did not
know. None is exotic — they are what published .ode files actually contain.
"""

import pytest

from hallsim.discovery import _is_junk_archive_entry
from hallsim.xpp_import import process_from_xpp

MINIMAL = """% Spike adaptation by an erg-like K+ current.
% Written by someone, J Physiol 1997;501:313-318
params gk=0.5, tau=10
ini x=0.1
y(0)=0.2
x' = gk*(1-x) - x/tau
y' = x - 0.05*y
@ total=100
done
"""


class TestJunkArchiveEntries:
    """A zip written on macOS carries an AppleDouble beside every file. It has
    the right extension and none of the content, so an extractor that trusts
    the suffix hands the importer a binary blob — ModelDB 35358's
    `UnicodeDecodeError: 0xa2`."""

    @pytest.mark.parametrize(
        "name",
        [
            "__MACOSX/model.ode",
            "__MACOSX/sub/._model.ode",
            "._model.ode",
            "dir/._model.cps",
            ".DS_Store",
        ],
    )
    def test_debris_is_rejected(self, name):
        assert _is_junk_archive_entry(name)

    @pytest.mark.parametrize(
        "name", ["model.ode", "sub/model.ode", "a_._b/model.ode"]
    )
    def test_real_files_are_kept(self, name):
        assert not _is_junk_archive_entry(name)


class TestPercentLines:
    def test_a_percent_comment_does_not_abort_the_parse(self, tmp_path):
        """`%` opens XPPAUT's array block only when followed by `[`. Three
        ModelDB entries were unreadable because a title line began with it."""
        f = tmp_path / "m.ode"
        f.write_text(MINIMAL)
        proc = process_from_xpp(str(f), name="m")
        assert set(proc.ports_schema()) == {"x", "y"}

    def test_an_array_block_is_still_refused(self, tmp_path):
        """`%[1..10]` is the real construct and is not supported; it must not
        be silently swallowed as a comment."""
        f = tmp_path / "m.ode"
        f.write_text(MINIMAL.replace("% Spike", "%[1..10]\n% Spike"))
        with pytest.raises(Exception):
            process_from_xpp(str(f), name="m")


class TestKeywordSynonyms:
    @pytest.mark.parametrize(
        "param_kw", ["par", "param", "params", "parameters", "p"]
    )
    def test_parameter_keyword_forms(self, tmp_path, param_kw):
        f = tmp_path / "m.ode"
        f.write_text(MINIMAL.replace("params ", f"{param_kw} "))
        assert process_from_xpp(str(f), name="m").ports_schema()

    @pytest.mark.parametrize("init_kw", ["i", "ini", "init", "initial"])
    def test_init_keyword_forms(self, tmp_path, init_kw):
        f = tmp_path / "m.ode"
        f.write_text(MINIMAL.replace("ini ", f"{init_kw} "))
        assert process_from_xpp(str(f), name="m").ports_schema()


def test_a_non_utf8_byte_does_not_abort_the_read(tmp_path):
    """A degree sign or Greek letter in a comment, in the author's encoding."""
    f = tmp_path / "m.ode"
    f.write_bytes(
        MINIMAL.replace("% Written", "% 37\xb0C, \xb5M").encode("latin-1")
    )
    assert set(process_from_xpp(str(f), name="m").ports_schema()) == {"x", "y"}
