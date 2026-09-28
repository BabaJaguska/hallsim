"""The search served as MCP tools."""

import asyncio

import pytest

from hallsim.search import server
from hallsim.search.datasets import DatasetCandidate, Design
from hallsim.search.models import ModelCandidate

fastmcp = pytest.importorskip("fastmcp")


def _call(mcp, tool, **args):
    async def go():
        async with fastmcp.Client(mcp) as client:
            return (await client.call_tool(tool, args)).data

    return asyncio.run(go())


def test_the_tools_are_thin_calls_into_the_search(monkeypatch):
    stated = Design(
        arms=("ctrl", "drug"),
        control="ctrl",
        per_arm=(("ctrl", (0.0, 6.0)), ("drug", (0.0, 6.0, 24.0))),
        time_unit="h",
        n_titles=6,
    )
    hit = DatasetCandidate(
        source="geo",
        accession="GSE1",
        title="a time course",
        kind="Expression profiling by array",
        organism="Homo sapiens",
        n_samples=6,
        platform="GPL1",
        url="u",
        stated=stated,
        curated="GDS1",
    )
    asked = {}

    def find(query, limit=25, sources=None, organism=None):
        asked.update(
            query=query, limit=limit, sources=sources, organism=organism
        )
        return [hit]

    monkeypatch.setattr(server, "search_for_dataset", find)
    monkeypatch.setattr(
        server,
        "search_for_model",
        lambda query, limit=25, sources=None: [
            ModelCandidate("biomodels", "BIOMD1", "p53", "SBML", "u", True)
        ],
    )
    mcp = server.build_server()
    names = (
        {t.name for t in asyncio.run(mcp.get_tools()).values()}
        if hasattr(mcp, "get_tools")
        else set()
    )
    data = _call(
        mcp, "find_data", query="p53", limit=3, organism="Homo sapiens"
    )
    assert asked == {
        "query": "p53",
        "limit": 3,
        "sources": None,
        "organism": "Homo sapiens",
    }
    (row,) = data
    assert row["accession"] == "GSE1" and row["curated"] == "GDS1"
    assert row["design"]["arms"] == ["ctrl", "drug"]
    assert row["design_summary"] == "2 arms (control: ctrl), 3 timepoints h"
    _call(mcp, "find_data", query="p53", sources=["all"])
    assert set(asked["sources"]) == set(server.SOURCES)
    (model,) = _call(mcp, "find_models", query="p53")
    assert (model["source"], model["id"]) == ("biomodels", "BIOMD1")
    listed = _call(mcp, "sources")
    assert (
        listed["data_default"][0] == "omicsdi"
        and "biomodels" in listed["models"]
    )
    assert (
        not names
        or {
            "find_data",
            "find_models",
            "dataset_design",
            "paper_datasets",
            "sources",
        }
        <= names
    )


def test_a_dataset_design_is_read_by_accession_kind(monkeypatch):
    monkeypatch.setattr(
        server,
        "gds_subsets",
        lambda acc, timeout=60.0: [
            ("time", "hour 6", ("s1",)),
            ("time", "hour 24", ("s2",)),
            ("agent", "drug", ("s1", "s2")),
        ],
    )
    monkeypatch.setattr(
        server, "geo_by_accession", lambda acc, timeout=30.0: None
    )
    mcp = server.build_server()
    d = _call(mcp, "dataset_design", accession="gds6010")
    assert d["accession"] == "GDS6010" and d["arms"] == ["drug"]
    assert d["per_arm"] == [["drug", [6.0, 24.0]]]
    assert "error" in _call(mcp, "dataset_design", accession="GSE0")
