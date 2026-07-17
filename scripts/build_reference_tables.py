"""Build the vendored reference tables the reporter-wiring checker reads
(:mod:`hallsim.reporter_wiring`).

These are **complete, model-agnostic** resources — not scoped to any
model's IDs — so the checker resolves any annotated composite without
re-running. Run deliberately to refresh a pinned snapshot (per release,
not per model). Runtime never fetches: it reads these flat files, so
validation stays offline and deterministic.

    python scripts/build_reference_tables.py

Build-time network sources:
  - UniProt REST stream — reviewed human+mouse accession→gene symbol
  - GO OBO (go-basic) — every GO term's aspect
  - MGI homology report — mouse→human ortholog symbols
  - OmniPath via `decoupler` — the CollecTRI regulon (whole)
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_ONT = REPO / "data" / "ontology"
_COLLECTRI = REPO / "data" / "collectri"
_UA = {"User-Agent": "hallsim-build"}


def _get(url: str, timeout: int = 180) -> bytes:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=_UA), timeout=timeout
    ).read()


def _write_tsv(path: Path, header: str, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [header] + ["\t".join(map(str, r)) for r in rows]
    path.write_text("\n".join(body) + "\n")
    print(f"  wrote {path.relative_to(REPO)} ({len(rows)} rows)")


def build_uniprot_symbols() -> None:
    url = (
        "https://rest.uniprot.org/uniprotkb/stream?query=reviewed:true+AND+"
        "(organism_id:9606+OR+organism_id:10090)"
        "&fields=accession,gene_primary,organism_id&format=tsv"
    )
    rows = []
    for line in _get(url).decode().splitlines()[1:]:
        acc, sym, taxon = (line.split("\t") + ["", ""])[:3]
        if sym:
            rows.append((acc, sym, taxon))
    _write_tsv(
        _ONT / "uniprot_symbol.tsv", "uniprot\tsymbol\ttaxon", sorted(rows)
    )


def build_go_aspects() -> None:
    obo = _get("https://purl.obolibrary.org/obo/go/go-basic.obo").decode()
    rows, cur = [], {}
    for line in obo.splitlines():
        if line == "[Term]":
            cur = {}
        elif line.startswith("id: GO:"):
            cur["id"] = line[4:]
        elif line.startswith("namespace: "):
            cur["aspect"] = line[11:]
        elif line.startswith("name: "):
            cur["name"] = line[6:]
        elif line.startswith("is_obsolete: true"):
            cur["obsolete"] = True
        elif line == "" and cur.get("id") and not cur.get("obsolete"):
            rows.append(
                (cur["id"], cur.get("aspect", ""), cur.get("name", ""))
            )
            cur = {}
    _write_tsv(
        _ONT / "go_aspect.tsv", "go_id\taspect\tname", sorted(set(rows))
    )


def build_orthologs() -> None:
    rpt = _get(
        "https://www.informatics.jax.org/downloads/reports/"
        "HOM_MouseHumanSequence.rpt"
    ).decode()
    lines = rpt.splitlines()
    header = lines[0].split("\t")
    ci = {h: i for i, h in enumerate(header)}
    groups: dict[str, dict[str, list[str]]] = {}
    for line in lines[1:]:
        f = line.split("\t")
        key, org, sym = (
            f[ci["DB Class Key"]],
            f[ci["Common Organism Name"]],
            f[ci["Symbol"]],
        )
        species = "human" if org == "human" else "mouse"
        groups.setdefault(key, {}).setdefault(species, []).append(sym)
    pairs = {
        (m, h)
        for g in groups.values()
        for m in g.get("mouse", [])
        for h in g.get("human", [])
    }
    _write_tsv(
        _ONT / "ortholog_mouse_human.tsv",
        "mouse_symbol\thuman_symbol",
        sorted(pairs),
    )


def build_collectri() -> None:
    try:
        import decoupler as dc
    except ImportError:
        print(
            "  SKIP collectri: `pip install decoupler` to refresh the regulon."
        )
        return
    net = dc.op.collectri(organism="human")
    slim = net[["source", "target", "weight", "resources"]]
    _COLLECTRI.mkdir(parents=True, exist_ok=True)
    slim.to_csv(_COLLECTRI / "collectri_human.tsv", sep="\t", index=False)
    (_COLLECTRI / "PROVENANCE.md").write_text(
        "# CollecTRI (human) — vendored snapshot\n\n"
        "- Source: OmniPath via `decoupler.op.collectri(organism='human')`\n"
        f"- decoupler version: {dc.__version__}\n"
        f"- Edges: {len(slim)} | TFs: {slim.source.nunique()} | "
        f"targets: {slim.target.nunique()}\n"
        "- Columns: source (TF), target (gene), weight (+1 activation / "
        "-1 repression), resources\n\n"
        "Citation: Müller-Dott et al., Nucleic Acids Research 2023 "
        "(CollecTRI); Türei et al. 2016/2021 (OmniPath).\n"
    )
    print(f"  wrote data/collectri/collectri_human.tsv ({len(slim)} edges)")


def main() -> None:
    print("Building complete reference tables (offline at runtime).")
    build_uniprot_symbols()
    build_go_aspects()
    build_orthologs()
    build_collectri()
    print("Done.")


if __name__ == "__main__":
    main()
