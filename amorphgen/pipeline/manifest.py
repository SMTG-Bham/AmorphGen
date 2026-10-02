"""Durable, versioned records of pipeline invocations and stage outcomes."""

from __future__ import annotations

import json
import math
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _json_value(value):
    """Snapshot config values without serialising arbitrary calculator objects."""
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    cls = type(value)
    return {"python_type": f"{cls.__module__}.{cls.__qualname__}"}


class RunManifest:
    """Append an invocation and atomically persist each lifecycle transition.

    Earlier attempts are retained verbatim, including attempts left running
    by a process kill. A work directory, like the pipeline itself, must have
    only one writer at a time.
    """

    def __init__(self, work_dir, input_file, cfg, stages, stage_names, resume,
                 resume_settings=None):
        from .. import __version__

        self.path = Path(work_dir).resolve() / "run_manifest.json"
        self._started = time.perf_counter()
        self._active_stage = None
        self._stage_started = None
        if self.path.exists():
            try:
                with self.path.open(encoding="utf-8") as stream:
                    self.data = json.load(stream)
            except (ValueError, OSError) as exc:
                raise ValueError(f"Cannot read existing run manifest {self.path}: {exc}") from exc
            if (not isinstance(self.data, dict)
                    or self.data.get("schema_version") != 1
                    or not isinstance(self.data.get("attempts"), list)
                    or not all(isinstance(a, dict) for a in self.data["attempts"])):
                raise ValueError(f"Unsupported or invalid run manifest: {self.path}")
        else:
            self.data = {"schema_version": 1, "attempts": []}

        self.previous_stages = {}
        if resume:
            if self.data["attempts"]:
                previous = self.data["attempts"][-1].get("resume_settings")
                if previous is None or resume_settings is None:
                    raise ValueError(
                        "Cannot resume: run manifest has no verifiable resume settings. "
                        "Use a new work directory."
                    )
                current = _json_value(resume_settings)
                # Appending later stages is safe with the same full protocol;
                # removing/reordering the existing sequence changes its inputs.
                if isinstance(previous, dict) and isinstance(previous.get("stages"), list):
                    old_stages = previous["stages"]
                    if current.get("stages", [])[:len(old_stages)] == old_stages:
                        previous = {**previous, "stages": current["stages"]}
                changes = _changed_settings(previous, current)
                if changes:
                    raise ValueError(
                        "Cannot resume: settings changed: " + ", ".join(changes) +
                        ". Use a new work directory for a different run."
                    )
                # Only checkpoints from the latest fresh run belong to this
                # resume chain. A pending stage may have stale files from a
                # different protocol previously run in the same directory.
                for attempt in reversed(self.data["attempts"]):
                    for record in attempt.get("stages", []):
                        stage = record.get("stage")
                        if stage not in self.previous_stages and record.get("status") != "pending":
                            self.previous_stages[stage] = record.get("status")
                    if not attempt.get("resume"):
                        break
            elif any(p.name != ".amorphgen.lock" and p.resolve() != Path(input_file).resolve()
                     for p in self.path.parent.iterdir()):
                raise ValueError(
                    "Cannot resume: existing outputs have no verifiable run manifest. "
                    "Use a new work directory."
                )

        self.attempt = {
            "attempt": len(self.data["attempts"]) + 1,
            "package_version": __version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "started_at": _timestamp(),
            "finished_at": None,
            "elapsed_seconds": None,
            "status": "running",
            "resume": bool(resume),
            "requested_input_file": os.path.abspath(input_file),
            "input_file": os.path.abspath(input_file),
            "work_dir": str(self.path.parent),
            "requested_stages": list(stages),
            "config": _json_value(cfg),
            "resume_settings": _json_value(resume_settings),
            "seed": _json_value(cfg.get("seed")),
            "seed_index": None,
            # This orchestrator always calls ASE stages, even if the config
            # contains the engine option used by ensemble workflows.
            "engine": "ase",
            "precision": {"requested": cfg.get("default_dtype", "auto"), "resolved": None},
            "device": {"requested": cfg.get("device", "auto"), "resolved": None},
            "model": {
                "name": cfg.get("model"), "path": None, "sha256": None,
                "hash_source": None, "hash_unavailable_reason": "Calculator not loaded",
                "calculator_class": None,
            },
            "stages": [
                {"stage": s, "name": stage_names.get(s, f"Stage {s}"), "status": "pending"}
                for s in stages
            ],
        }
        self.data["attempts"].append(self.attempt)
        self.save()

    def save(self):
        """Replace the manifest only after the complete new JSON is on disk."""
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=".run_manifest.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = stream.name
                json.dump(_json_value(self.data), stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def skip_stages(self, stages, checkpoints):
        for record in self.attempt["stages"]:
            if record["stage"] in stages:
                record.update(status="skipped", reason="checkpoint_exists",
                              checkpoint=checkpoints[record["stage"]])
        self.save()

    def start_stage(self, stage):
        self._active_stage = next(
            record for record in self.attempt["stages"]
            if record["stage"] == stage and record["status"] == "pending"
        )
        self._stage_started = time.perf_counter()
        self._active_stage.update(status="running", started_at=_timestamp(),
                                  finished_at=None, elapsed_seconds=None)
        self.save()

    def finish_stage(self, status="completed", error=None):
        if self._active_stage is not None:
            self._active_stage.update(
                status=status, finished_at=_timestamp(),
                elapsed_seconds=time.perf_counter() - self._stage_started,
            )
            if error is not None:
                self._active_stage["error"] = error
            self._active_stage = None
            self._stage_started = None
        self.save()

    def finish(self, status, exc=None):
        error = None
        if exc is not None:
            error = {"type": type(exc).__name__, "message": str(exc)}
            self.attempt["error"] = error
        self.attempt.update(status=status, finished_at=_timestamp(),
                            elapsed_seconds=time.perf_counter() - self._started)
        self.finish_stage(status, error)


def _changed_settings(previous, current, prefix=""):
    """Return the specific configuration paths that differ, without values."""
    if isinstance(previous, dict) and isinstance(current, dict):
        changes = []
        for key in sorted(previous.keys() | current.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in previous or key not in current:
                changes.append(path)
            else:
                changes.extend(_changed_settings(previous[key], current[key], path))
        return changes
    return [] if previous == current else [prefix]
