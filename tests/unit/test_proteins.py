"""PRIDE mzTab protein sections as UniProt-keyed contrasts.

Parsing is ``pyteomics.mztab``; what is tested here is the layer over it —
the UniProt keying, the ambiguity rule, and the groups a study variable
declares.
"""

import numpy as np
import pytest

from hallsim.proteins import ProteinDataset, read_mztab

_HEAD = """\
MTD\tmzTab-version\t1.0.0
MTD\tmzTab-mode\tSummary
MTD\tmzTab-type\tQuantification
MTD\tdescription\ta synthetic quantification report
MTD\tms_run[1]-location\tfile://run.raw
MTD\tprotein-quantification-unit\tnull
MTD\tprotein_search_engine_score[1]\t[MS,MS:1001155,SEQUEST:xcorr,]
"""

BY_ASSAY = _HEAD + """\
MTD\tassay[1]-ms_run_ref\tms_run[1]
MTD\tassay[2]-ms_run_ref\tms_run[1]
MTD\tassay[3]-ms_run_ref\tms_run[1]
MTD\tassay[4]-ms_run_ref\tms_run[1]
MTD\tstudy_variable[1]-assay_refs\tassay[1],assay[2]
MTD\tstudy_variable[1]-description\tcontrol 24 h
MTD\tstudy_variable[2]-assay_refs\tassay[3],assay[4]
MTD\tstudy_variable[2]-description\trapamycin 24 h

PRH\taccession\tdescription\ttaxid\tspecies\tdatabase\tdatabase_version\tsearch_engine\tbest_search_engine_score[1]\tambiguity_members\tmodifications\tprotein_abundance_assay[1]\tprotein_abundance_assay[2]\tprotein_abundance_assay[3]\tprotein_abundance_assay[4]
PRT\tP04637\tp53\t9606\tHomo sapiens\tUniProt\t1\t[MS,MS:1001155,x,]\t1\tnull\tnull\t100\t110\t200\t220
PRT\tP42345\tmTOR\t9606\tHomo sapiens\tUniProt\t1\t[MS,MS:1001155,x,]\t1\tnull\tnull\t50\t55\t25\t27.5
PRT\tQ00987\tMDM2\t9606\tHomo sapiens\tUniProt\t1\t[MS,MS:1001155,x,]\t1\tnull\tnull\t0\t-1\t10\t10
PRT\tP31749\tAKT1\t9606\tHomo sapiens\tUniProt\t1\t[MS,MS:1001155,x,]\t1\tP31749,P31751\tnull\t9\t9\t9\t9
"""

BY_STUDY_VARIABLE = _HEAD + """\
MTD\tstudy_variable[1]-description\tcontrol 24 h
MTD\tstudy_variable[2]-description\trapamycin 24 h

PRH\taccession\tdescription\ttaxid\tspecies\tdatabase\tdatabase_version\tsearch_engine\tbest_search_engine_score[1]\tambiguity_members\tmodifications\tprotein_abundance_study_variable[1]\tprotein_abundance_stdev_study_variable[1]\tprotein_abundance_study_variable[2]\tprotein_abundance_stdev_study_variable[2]
PRT\tP04637\tp53\t9606\tHomo sapiens\tUniProt\t1\t[MS,MS:1001155,x,]\t1\tnull\tnull\t105\t7\t210\t14
PRT\tP42345\tmTOR\t9606\tHomo sapiens\tUniProt\t1\t[MS,MS:1001155,x,]\t1\tnull\tnull\t52.5\t3\t26.25\t2
"""

IDS_ONLY = _HEAD + """\

PRH\taccession\tdescription\ttaxid\tspecies\tdatabase\tdatabase_version\tsearch_engine\tbest_search_engine_score[1]\tambiguity_members\tmodifications
PRT\tP04637\tp53\t9606\tHomo sapiens\tUniProt\t1\t[MS,MS:1001155,x,]\t1\tnull\tnull
"""


def _write(tmp_path, text, name="test.mzTab"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_abundances_per_assay_are_keyed_by_uniprot(tmp_path):
    table = read_mztab(_write(tmp_path, BY_ASSAY))

    assert table.level == "assay"
    assert table.proteins == 4
    assert list(table.quantities.columns) == [
        "assay[1]",
        "assay[2]",
        "assay[3]",
        "assay[4]",
    ]
    assert list(table.quantities.index) == ["P04637", "P42345", "Q00987"]


def test_an_ambiguous_protein_group_is_counted_not_attributed(tmp_path):
    """The abundance stands for whichever member is present, so crediting
    the leading accession would assert an identification the file declined
    to make."""
    table = read_mztab(_write(tmp_path, BY_ASSAY))

    assert table.ambiguous == 1
    assert "P31749" not in table.quantities.index


def test_below_detection_is_missing_not_zero(tmp_path):
    table = read_mztab(_write(tmp_path, BY_ASSAY))
    mdm2 = table.quantities.loc["Q00987"]

    assert np.isnan(mdm2["assay[1]"]) and np.isnan(mdm2["assay[2]"])
    assert mdm2["assay[3]"] == 10.0


def test_groups_come_from_the_declared_study_variables(tmp_path):
    ds = ProteinDataset.from_mztab(_write(tmp_path, BY_ASSAY))

    assert ds.sample_groups == {
        "control 24 h": ["assay[1]", "assay[2]"],
        "rapamycin 24 h": ["assay[3]", "assay[4]"],
    }
    assert ds.has_replicates
    assert ds.design.arms == ("control", "rapamycin")
    assert ds.design.timepoints == (24.0,)


def test_a_contrast_is_keyed_so_it_joins_model_species(tmp_path):
    ds = ProteinDataset.from_mztab(_write(tmp_path, BY_ASSAY))
    delta = ds.delta("rapamycin 24 h", "control 24 h")

    assert delta["P04637"] == pytest.approx(1.0)
    assert delta["P42345"] == pytest.approx(-1.0)
    # No baseline left once below-detection cells are missing.
    assert np.isnan(delta["Q00987"])
    assert {"P04637"} == set(delta.index) & {"P04637", "P99999"}


def test_replicate_variance_is_available_per_assay(tmp_path):
    ds = ProteinDataset.from_mztab(_write(tmp_path, BY_ASSAY))
    var = ds.variance("rapamycin 24 h", "control 24 h")

    assert var["P04637"] > 0 and np.isfinite(var["P04637"])


def test_a_summary_level_file_has_no_replicates_but_keeps_its_spread(
    tmp_path,
):
    ds = ProteinDataset.from_mztab(_write(tmp_path, BY_STUDY_VARIABLE))

    assert ds.level == "study_variable"
    assert not ds.has_replicates
    assert ds.sample_groups == {
        "control 24 h": ["study_variable[1]"],
        "rapamycin 24 h": ["study_variable[2]"],
    }
    assert ds.delta("rapamycin 24 h", "control 24 h")[
        "P04637"
    ] == pytest.approx(1.0)
    assert ds.stdev is not None
    assert ds.stdev.loc["P04637", "study_variable[1]"] == 7.0


def test_a_file_with_only_identifications_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no protein abundances"):
        read_mztab(_write(tmp_path, IDS_ONLY))


def test_explicit_groups_override_the_declared_ones(tmp_path):
    ds = ProteinDataset.from_mztab(
        _write(tmp_path, BY_ASSAY),
        sample_groups={"a": ["assay[3]", "assay[4]"], "b": ["assay[1]"]},
    )
    assert ds.design is None
    # A group mean is taken on the log scale, so the ratio is of geometric
    # means, as it is for expression.
    assert ds.delta("a", "b")["P42345"] == pytest.approx(
        np.log2(np.sqrt(25.0 * 27.5) / 50.0)
    )
