"""
tests/test_citation.py
----------------------
CITATION.cff names the current release.
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


def test_citation_names_the_current_release():
    cff = yaml.safe_load(_read("CITATION.cff"))
    assert cff["version"] == amorphgen.__version__, (
        "CITATION.cff names another release: update its version and "
        "date-released along with the package version")

    heading = f"## v{cff['version']} ({cff['date-released']})"
    assert heading in _read("docs/changelog.md").splitlines(), (
        f"docs/changelog.md has no '{heading}' heading: date-released in "
        "CITATION.cff should be the release date the changelog gives")
