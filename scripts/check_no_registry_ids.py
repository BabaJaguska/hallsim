"""Fail when code cites a defect-registry id.

A `P0.17` in a comment dates the line to an incident and sends the reader out
of the file to learn nothing the sentence should not already say. The registry
is prose, and the reviews and the diary are what cite it.

    python3 scripts/check_no_registry_ids.py src demos tests
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REFERENCE = re.compile(r"\bP\d+\.\d+\b")
#: This file and the registry's own test hold the pattern, not a citation.
EXEMPT = {"check_no_registry_ids.py", "test_known_problems.py"}


def violations(roots: list[Path]) -> list[str]:
    """``["path:line: text", ...]`` for every cited id under ``roots``."""
    out = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if path.name in EXEMPT:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for n, line in enumerate(text.splitlines(), 1):
                if REFERENCE.search(line):
                    out.append(f"{path}:{n}: {line.strip()[:88]}")
    return out


def main() -> int:
    roots = [Path(a) for a in sys.argv[1:]] or [Path("src")]
    found = violations([r for r in roots if r.is_dir()])
    for line in found:
        print(line)
    if found:
        print(
            f"\n{len(found)} defect id(s) cited from code. State the reason "
            "in words; the id belongs in docs/known-problems.md."
        )
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
