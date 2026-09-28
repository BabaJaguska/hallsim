"""The search layer is a package of its own inside HallSim: it imports
nothing from the rest of the framework, so it can be lifted out whole."""

import ast
from pathlib import Path

import hallsim.search


def _imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.lineno, node.module


def test_search_imports_nothing_from_the_rest_of_hallsim():
    root = Path(hallsim.search.__file__).parent
    offenders = [
        f"{path.name}:{lineno} {name}"
        for path in sorted(root.glob("*.py"))
        for lineno, name in _imports(path)
        if name == "hallsim"
        or (
            name.startswith("hallsim.")
            and not name.startswith("hallsim.search")
        )
    ]
    assert offenders == []
