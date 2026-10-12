"""Persist convergence evidence from completed MD stages without plotting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase.io import read

from .common import TRAJ_LOG_INTERVAL
from .equilibration import convergence_report, parse_md_log
from .persistence import atomic_write_text


def write_stage_diagnostics(logfile, trajfile, *, timestep_fs, stage,
                            T_target=None, temperatures=None, steps_per_T=None,
                            traj_format="extxyz", engine="ase"):
    """Write ``<log stem>_diagnostics.{json,txt}`` and return the report.

    The paired log supplies MD step numbers, including resumed runs and
    torch-sim's shorter final output block. Ramp targets describe the actual
    resolved schedule, not instantaneous noisy kinetic temperatures. A sample
    exactly at a hold boundary belongs to the hold that just finished.

    Missing or unreadable evidence is recorded explicitly. Diagnostics never
    infer a liquid or a freezing temperature from an unavailable trajectory.
    """
    logfile, trajfile = Path(logfile), Path(trajfile)
    metadata = {
        "schema_version": 1,
        "stage": int(stage),
        "engine": engine,
        "timestep_fs": float(timestep_fs),
        "frame_stride": TRAJ_LOG_INTERVAL,
        "source": {"log": str(logfile), "trajectory": str(trajfile)},
    }
    try:
        fmt = "extxyz" if traj_format.lower() == "xyz" else traj_format.lower()
        frames = read(str(trajfile), index=":", format=fmt)
        log_data = parse_md_log(logfile)
        steps = log_data["step"]
        if len(frames) != len(steps):
            raise ValueError(
                f"Trajectory/log sample counts differ ({len(frames)} vs {len(steps)}).")
        # The formatted time column has finite precision; step numbers preserve
        # exact timing even for small timesteps and a partial final block.
        time_ps = steps * float(timestep_fs) / 1000.0
        targets = None
        if temperatures is not None:
            schedule = np.asarray(temperatures, dtype=float)
            if (schedule.ndim != 1 or not len(schedule)
                    or steps_per_T is None or steps_per_T <= 0):
                raise ValueError("A ramp needs temperatures and positive steps_per_T.")
            indices = np.maximum(steps - 1, 0) // int(steps_per_T)
            if np.any(indices >= len(schedule)):
                raise ValueError("Logged steps extend beyond the resolved temperature ramp.")
            targets = schedule[indices]
        report = convergence_report(
            frames, timestep_fs=timestep_fs, T_target=T_target,
            frame_stride=TRAJ_LOG_INTERVAL, make_plots=False,
            time_ps=time_ps, temperatures=targets)
        report["sample_steps"] = steps.tolist()
        report["sample_time_ps"] = time_ps.tolist()
    except (OSError, ValueError, RuntimeError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        report = {
            "status": "unavailable",
            "block_test_passed": None,
            "block_data": {"status": "unavailable"},
            "diffusion_coefficients_cm2_s": {},
            "msd_final_A2": {},
            "liquid_test": {"status": "unavailable", "is_liquid": None},
            "stationarity": {
                kind: {"status": "unavailable", "passed": None}
                for kind in ("energy", "diffusion")
            },
            "temperature_windows": [],
            "freezing_temperature_K": None,
            "diffusion_freezing": {
                "status": "unavailable", "temperature_K": None, "bracket_K": None,
            },
            "warnings": [reason],
            "summary_text": f"MD CONVERGENCE DIAGNOSTICS\nUnavailable: {reason}\n",
        }
    report.update(metadata)
    stem = logfile.with_suffix("")
    json_path = stem.with_name(stem.name + "_diagnostics.json")
    text_path = stem.with_name(stem.name + "_diagnostics.txt")
    atomic_write_text(json_path, json.dumps(report, indent=2, allow_nan=False) + "\n",
                      fsync=False)
    atomic_write_text(text_path, report["summary_text"] + "\n", fsync=False)
    print(f"[Stage {stage}] Diagnostics -> {json_path} ({report['status']})")
    return report
