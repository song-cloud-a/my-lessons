"""Fast checks that do not require downloading ML dependencies or weights."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_are_valid() -> None:
    sources = list(ROOT.glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    assert sources
    for source in sources:
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


def test_required_community_files_exist() -> None:
    for name in ("README.md", "LICENSE", "CONTRIBUTING.md", ".gitignore"):
        assert (ROOT / name).is_file()
