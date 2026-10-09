"""Small persistence primitives shared by run records and structure sidecars."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile


def sha256_file(path):
    """Hash file bytes without loading the whole file into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path, text, *, prefix=None, fsync=True):
    """Replace a UTF-8 document after writing a temporary file beside it.

    Callers own serialization and parent-directory creation. ``fsync=False``
    is intended for reconstructible sidecars; durable run records use the
    default. Errors propagate after cleaning up the temporary file.
    """
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=prefix if prefix is not None else destination.name + ".",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(text)
            if fsync:
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def changed_settings(previous, current, prefix=""):
    """Return sorted dotted paths that differ between JSON-compatible values."""
    if isinstance(previous, dict) and isinstance(current, dict):
        changes = []
        for key in sorted(previous.keys() | current.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in previous or key not in current:
                changes.append(path)
            else:
                changes.extend(changed_settings(previous[key], current[key], path))
        return changes
    return [] if previous == current else [prefix]
