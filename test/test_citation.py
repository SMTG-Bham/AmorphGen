"""
tests/test_citation.py
----------------------
CITATION.cff names an archived release; the changelog has the package version.
"""

from pathlib import Path

import pytest
import yaml

import amorphgen

ROOT = Path(__file__).resolve().parents[1]


def _read(name):
    path = ROOT / name
    if not path.is_file():
        # an sdist ships the tests but not these files
        pytest.skip(f"{name} is not in this source tree")
    return path.read_text(encoding="utf-8")


def _release(version):
    return tuple(int(part) for part in version.split(".")[:3])


def test_citation_names_an_archived_release():
    """CITATION.cff names the archived release that the paper cites (its
    version DOI), which may be older than the package but never newer."""
    cff = yaml.safe_load(_read("CITATION.cff"))
    assert _release(cff["version"]) <= _release(amorphgen.__version__), (
        "CITATION.cff names a release newer than the package version")

    heading = f"## v{cff['version']} ({cff['date-released']})"
    assert heading in _read("docs/changelog.md").splitlines(), (
        f"docs/changelog.md has no '{heading}' heading: date-released in "
        "CITATION.cff should be the release date the changelog gives")


def test_changelog_has_the_package_version():
    lines = _read("docs/changelog.md").splitlines()
    assert any(line.startswith(f"## v{amorphgen.__version__} (") for line in lines), (
        f"docs/changelog.md has no section for v{amorphgen.__version__}")
