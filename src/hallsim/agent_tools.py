"""The tools that open a search hit, for an agent: what role a species plays
in a deposit, the intake triage, and the shipped census tables.

:func:`framework_tools` registers them on a server from
:func:`hallsim.search.server.build_server`, which stays free of the
framework so the search package can be lifted out whole. Composing and
simulating need a live Python object and stay in code.
"""

from __future__ import annotations


def _number(x) -> float | None:
    """A float for JSON: NaN and the infinities become ``None``."""
    x = float(x)
    return x if x == x and abs(x) != float("inf") else None


def verdict_record(verdict) -> dict:
    """A triage verdict as a dictionary, the numerical screen as its text."""
    return {
        "name": verdict.name,
        "status": verdict.status,
        "escalate": verdict.escalate,
        "blockers": list(verdict.blockers),
        "flags": list(verdict.flags),
        "n_species": verdict.n_species,
        "n_parameters": verdict.n_parameters,
        "time_unit_declared": verdict.time_unit_declared,
        "annotation_coverage": _number(verdict.annotation_coverage),
        "rest_residual": _number(verdict.rest_residual),
        "screen": None if verdict.screen is None else str(verdict.screen),
    }


def framework_tools(mcp):
    """Register ``screen_models``, ``triage_model``, ``census_models`` and
    ``census_datasets`` on ``mcp`` and return it."""
    from hallsim.census_snapshots import (
        find_shipped_datasets,
        find_shipped_models,
    )

    @mcp.tool
    def screen_models(
        model_ids: list[str], pattern: str, roles: list[str] | None = None
    ) -> list[dict]:
        """What role each species matching ``pattern`` (a regex against
        species ids and names) plays in each deposit (BioModels accessions,
        ``jws:`` slugs or ``PMC`` ids from find_models): produces, consumes,
        reads (a modifier or rate-law symbol) or input (boundary or
        constant). ``status`` is ``match`` where one plays any of ``roles``
        (default all). A consumer or reader is the downstream half of a
        composite; the producer is an edge to attach."""
        from dataclasses import asdict

        from hallsim.screens import ROLES, screen_species_roles

        return [
            asdict(r)
            for r in screen_species_roles(
                model_ids, pattern, roles=tuple(roles or ROLES)
            )
        ]

    @mcp.tool
    def triage_model(
        model_id: str,
        source: str = "biomodels",
        t_end: float | None = None,
        produces: str | None = None,
    ) -> dict:
        """Import one deposit and run the intake gate on it, from any
        repository find_models lists: ``source`` is the hit's source
        (biomodels, jws, modeldb, physiome, biosimulations, europepmc) and
        ``model_id`` its id. ``status`` is ``pass``; ``flag`` (it runs, and
        each flag says what to do); ``reject`` (it will not import, or
        produced no usable trajectory); or ``unscreenable`` (it could not be
        fetched, or holds no readable model file, so nothing is known about
        the model). ``t_end`` is the solo solve window in the model's own
        time unit, by default the deposit's authored SED-ML horizon or 10;
        ``produces`` is a regex for a quantity the deposit must emit.
        Imports and solves the model, so a call takes seconds to minutes."""
        from hallsim import intake

        return verdict_record(
            intake.triage_model(
                model_id, source, t_end=t_end, produces=produces
            )
        )

    mcp.tool(find_shipped_models, name="census_models")
    mcp.tool(find_shipped_datasets, name="census_datasets")
    return mcp
