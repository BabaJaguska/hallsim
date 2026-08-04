"""Fail when the package drifts back toward more narration than code.

Docstrings and comments accumulate: each edit adds a paragraph explaining what
it did, nothing removes one. Two caps, because they catch different failures —
the **aggregate** is what actually drifts (every file gaining a little), while
the **per-file** cap catches one module going pathological. The per-file bound
is deliberately loose: an abstract interface is mostly contract, and squeezing
that is not the goal.

    python scripts/check_prose_ratio.py src/hallsim
"""

from __future__ import annotations

import argparse
import io
import sys
import tokenize
from pathlib import Path

DEFAULT_MAX_FILE_RATIO = 0.50
DEFAULT_MAX_TOTAL_RATIO = 0.32


def prose_and_code(path: Path) -> tuple[int, int]:
    """``(prose_lines, code_lines)`` — docstrings and comments vs the rest."""
    src = path.read_text(encoding="utf-8")
    doc_lines: set[int] = set()
    comment_lines: set[int] = set()
    prev = None
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            comment_lines.add(tok.start[0])
        elif tok.type == tokenize.STRING and prev in (
            tokenize.INDENT,
            tokenize.NEWLINE,
            tokenize.NL,
            None,
        ):
            doc_lines.update(range(tok.start[0], tok.end[0] + 1))
        if tok.type not in (tokenize.NL, tokenize.COMMENT):
            prev = tok.type

    prose = code = 0
    for n, line in enumerate(src.splitlines(), 1):
        if not line.strip():
            continue
        if n in doc_lines or n in comment_lines:
            prose += 1
        else:
            code += 1
    return prose, code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument(
        "--max-file-ratio", type=float, default=DEFAULT_MAX_FILE_RATIO
    )
    ap.add_argument(
        "--max-total-ratio", type=float, default=DEFAULT_MAX_TOTAL_RATIO
    )
    ap.add_argument(
        "--min-lines",
        type=int,
        default=80,
        help="skip files below this size; a short module is mostly docstring "
        "by construction",
    )
    args = ap.parse_args()

    offenders = []
    total_prose = total_code = 0
    for path in sorted(args.root.rglob("*.py")):
        prose, code = prose_and_code(path)
        total_prose += prose
        total_code += code
        if prose + code < args.min_lines:
            continue
        ratio = prose / (prose + code)
        if ratio > args.max_file_ratio:
            offenders.append((ratio, path, prose, code))

    overall = total_prose / max(1, total_prose + total_code)
    print(
        f"prose {total_prose} / code {total_code} — {overall:.0%} overall "
        f"(cap {args.max_total_ratio:.0%}), "
        f"{len(offenders)} file(s) over {args.max_file_ratio:.0%}"
    )
    for ratio, path, prose, code in sorted(offenders, reverse=True):
        print(f"  {ratio:.0%}  {path}  ({prose} prose / {code} code)")
    if offenders or overall > args.max_total_ratio:
        print(
            "\nSay it once, in the place a reader will look; rationale and "
            "measurements go to docs/diary.md."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
