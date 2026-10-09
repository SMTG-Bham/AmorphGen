"""Integrity and failure behavior of the shared persistence primitives."""

import hashlib
import json

import pytest

from amorphgen.utils import persistence


@pytest.mark.parametrize("contents", [b"", b"\x00\xff\r\n" * 300_000])
def test_file_digest_covers_binary_contents_across_chunks(tmp_path, contents):
    path = tmp_path / "weights.bin"
    path.write_bytes(contents)
    expected = hashlib.sha256(contents).hexdigest()
    assert persistence.sha256_file(path) == expected
    assert persistence.sha256_file(str(path)) == expected
    with pytest.raises(FileNotFoundError):
        persistence.sha256_file(tmp_path / "missing")


@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_failed_atomic_publication_keeps_old_document_and_cleans_up(tmp_path, monkeypatch, operation):
    path = tmp_path / "record.json"
    path.write_text('{"version": 1}\n')

    def fail(*args):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(persistence.os, operation, fail)
    with pytest.raises(OSError, match="simulated publication failure"):
        persistence.atomic_write_text(path, '{"version": 2}\n')
    assert json.loads(path.read_text()) == {"version": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_text_preserves_unicode_and_caller_formatting(tmp_path):
    path = tmp_path / "record.json"
    text = '{"composition": "α-Si", "value": 2}\n'
    persistence.atomic_write_text(path, text)
    assert path.read_bytes() == text.encode("utf-8")
    assert list(tmp_path.iterdir()) == [path]


def test_resume_comparison_preserves_sorted_paths_and_first_difference():
    from amorphgen.pipeline.manifest import _changed_settings
    from amorphgen.pipeline.random_gen import _changed_setting

    previous = {"z": [1, 2], "opt": {"fmax": .01, "old": True}}
    current = {"z": [1, 3], "opt": {"fmax": .02, "new": True}}
    assert _changed_settings(previous, current) == ["opt.fmax", "opt.new", "opt.old", "z"]
    assert _changed_setting(previous, current) == "opt.fmax"
    assert _changed_settings(current, current) == []
    assert _changed_setting(current, current) is None
