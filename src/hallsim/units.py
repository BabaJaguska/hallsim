"""Shared Pint registry and unit reconciliation.

Each store path is held in a single **canonical unit** — the unit declared by
the first port (in process-insertion order) that touches the path, matching
the port whose default seeds the initial state (``build_initial_store``). A
port that declares a *compatible but different* unit has its contributions
auto-converted to/from canonical, so a ``uM`` writer and an ``nM`` writer to
the same species sum correctly instead of silently adding 1 and 1000.
Dimensionally *incompatible* units are a hard error (raised here and by the
:class:`~hallsim.validation.UnitChecker`).
"""

from __future__ import annotations

import math

import pint

from hallsim.store import as_paths

# One registry, shared with the validation layer, so quantities interoperate.
UREG = pint.UnitRegistry()


def _clean(u: str | None) -> str:
    return (u or "").strip()


def conversion_factor(from_u: str | None, to_u: str | None) -> float:
    """Multiplier taking a magnitude in ``from_u`` to ``to_u``.

    ``1.0`` when either unit is unspecified, they are identical, or a unit
    string cannot be parsed (the validator warns about unparseable units;
    the RHS must not silently rescale on a guess). Raises ``ValueError`` on
    dimensionally incompatible units, and on an affine one: ``f(x) = ax + b``
    has no single multiplier, and ``f(1) = a + b`` is not it.
    """
    a, b = _clean(from_u), _clean(to_u)
    if not a or not b or a == b:
        return 1.0
    try:
        src, dst = UREG.parse_expression(a), UREG.parse_expression(b)
    except Exception:
        return 1.0
    try:
        one = float(src.to(dst).magnitude)
    except pint.DimensionalityError as e:
        raise ValueError(f"incompatible units {a!r} -> {b!r}: {e}") from e
    except Exception:
        return 1.0
    # A ratio-scale unit satisfies f(2) == 2 f(1). An affine one does not, and
    # pint refuses the doubling outright — either answer means "not a scale".
    try:
        two = float((2 * src).to(dst).magnitude)
        linear = math.isclose(two, 2.0 * one, rel_tol=1e-9, abs_tol=0.0)
    except pint.DimensionalityError as e:
        raise ValueError(f"incompatible units {a!r} -> {b!r}: {e}") from e
    except Exception:
        linear = False
    if not linear:
        raise ValueError(
            f"{a!r} -> {b!r} is an affine conversion, which has no single "
            f"multiplier: f(1) = {one:g} is scale plus offset, and applying "
            "it per port would rescale every value wrongly. Declare the port "
            "in a ratio-scale unit."
        )
    return one


def canonical_units(
    processes: dict, topology: dict[str, dict[str, str]]
) -> dict[str, str]:
    """Canonical unit per store path.

    The unit of the first port (process-insertion order, then schema order)
    mapped to that path — the same port whose default seeds the initial
    state, so the store's initial value is already in canonical units. May be
    ``""`` (unspecified) for a path whose first port declares no unit, in
    which case no reconciliation is applied there.
    """
    canon: dict[str, str] = {}
    for pname, proc in processes.items():
        topo = topology.get(pname, {})
        for port_name, port in proc.ports_schema().items():
            for path in as_paths(topo[port_name]):
                if path in canon:
                    continue
                canon[path] = _clean(getattr(port, "units", ""))
    return canon
