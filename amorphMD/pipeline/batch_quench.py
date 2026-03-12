"""
pipeline/batch_quench.py
------------------------
Batch runner for stages 5 → 6 → 7 on a set of independent high-T snapshots
extracted from the Stage 4 long equilibration trajectory.

Typical workflow
----------------
  Stage 1 → Stage 2 → Stage 3 → Stage 4 (long, e.g. 1 ns, sample every 10 ps)
                                    ↓  100 decorrelated snapshots
              batch_quench runs Stage 5 → 6 → 7 on each snapshot
                                    ↓
              N independent amorphous final structures in batch_run/run_XXXX/

Usage (Python API)
------------------
    from amorphMD.pipeline import batch_quench

    # From snapshot files produced by Stage 4 with sample_interval_ps set:
    results = batch_quench.run(
        snapshot_files=snapshot_paths,  # list[str] of .extxyz paths
        n_runs=20,
        select="uniform",               # best for decorrelation
        cfg_override={...},
        work_dir="batch_run",
    )

    # Or directly from any trajectory file:
    results = batch_quench.run_from_traj(
        traj_file="stage4_eq_high.extxyz",
        sample_interval_ps=10,
        n_runs=20,
    )

Usage (CLI)
-----------
    run_melt_quench.py --batch-quench \\
        --snapshot-dir snapshots/ \\
        --n-runs 20 --select uniform \\
        --work-dir batch_run/

    run_melt_quench.py --batch-quench \\
        --traj-file stage4_eq_high.extxyz \\
        --sample-interval 10 --n-runs 20 \\
        --work-dir batch_run/
"""

from __future__ import annotations

import os
import time
import traceback
from copy import deepcopy

from ase.io import read, write

from . import quench, equilibrate, opt_cell
from ..utils import get_mace_calculator, merge_config
from ..configs import DEFAULT_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(snapshot_files: list[str],
        n_runs: int | str = "all",
        select: str = "uniform",
        cfg_override: dict | None = None,
        work_dir: str = "batch_quench",
        stages: list[int] | None = None,
        calc=None,
        resume: bool = False) -> list[dict]:
    """
    Run stages 5 → 6 → 7 on a set of snapshot files.

    Parameters
    ----------
    snapshot_files : list[str]   Paths to snapshot .extxyz files from Stage 4.
    n_runs         : int or "all"   Number of independent runs to perform.
    select         : str   Frame selection strategy:
                       "uniform"  – evenly spaced (best for decorrelation, default)
                       "random"   – random sample without replacement
                       "first"    – first n_runs snapshots
                       "last"     – last n_runs snapshots
                       "all"      – all snapshots (ignores n_runs)
    cfg_override   : dict   Overrides for DEFAULT_CONFIG (shared across all runs).
    work_dir       : str   Parent directory; each run gets run_0000/, run_0001/, …
    stages         : list[int]   Stages to run per snapshot. Default [5, 6, 7].
    calc           : ASE calculator   Shared MACE calculator. Built if None.
    resume         : bool   If True, skip runs whose final output already exists.
                            Use this to continue a timed-out or interrupted batch.

    Returns
    -------
    list[dict]  One dict per run:
        {
          "run_id":       int,
          "snapshot":     str,    path to source snapshot file
          "work_dir":     str,    run-specific subdirectory
          "final_cif":    str,    path to stage7 output CIF
          "final_xyz":    str,    path to stage7 output XYZ
          "success":      bool,
          "error":        str or None,
          "elapsed_min":  float,
        }
    """
    if stages is None:
        stages = [5, 6, 7]

    global_cfg = merge_config(DEFAULT_CONFIG, cfg_override)
    selected   = _select_frames(snapshot_files, n_runs, select)
    n_total    = len(selected)

    # ── shared calculator ─────────────────────────────────────────────────────
    if calc is None:
        print("[BatchQuench] Initialising MACE calculator (shared across all runs)…")
        calc = get_mace_calculator(
            model=global_cfg["mace_model"],
            device=global_cfg["device"],
            model_path=global_cfg.get("model_path"),
        )

    os.makedirs(work_dir, exist_ok=True)
    orig_dir = os.getcwd()

    print(f"\n{'='*62}")
    print(f"  BATCH QUENCH  –  {n_total} independent runs")
    print(f"  Stages: {stages}  |  Work dir: {work_dir}/")
    print(f"{'='*62}\n")

    results = []

    for run_id, snap_path in enumerate(selected):
        run_subdir = os.path.join(work_dir, f"run_{run_id:04d}")
        os.makedirs(run_subdir, exist_ok=True)

        # ── Resume: skip completed runs ───────────────────────────────────────
        if resume and _is_complete(run_subdir, global_cfg):
            print(f"  [run_{run_id:04d}] Already complete — skipping.")
            results.append({
                "run_id":      run_id,
                "snapshot":    snap_path,
                "work_dir":    run_subdir,
                "final_cif":   _find_output(run_subdir, "output_cif", global_cfg),
                "final_xyz":   _find_output(run_subdir, "output_xyz", global_cfg),
                "success":     True,
                "error":       None,
                "elapsed_min": 0.0,
            })
            continue

        print(f"{'─'*62}")
        print(f"  Run {run_id:04d} / {n_total-1}  ←  {os.path.basename(snap_path)}")
        print(f"{'─'*62}")

        t0  = time.time()
        res = {
            "run_id":      run_id,
            "snapshot":    snap_path,
            "work_dir":    run_subdir,
            "final_cif":   None,
            "final_xyz":   None,
            "success":     False,
            "error":       None,
            "elapsed_min": 0.0,
        }

        try:
            atoms    = read(os.path.abspath(snap_path))
            run_cfg  = _make_run_cfg(global_cfg, run_id)
            os.chdir(run_subdir)

            for stage in stages:
                if stage == 5:
                    atoms = quench.run(atoms, cfg_override=run_cfg, calc=calc)
                elif stage == 6:
                    result = equilibrate.run(
                        atoms, stage_key="eq_low",
                        cfg_override=run_cfg, calc=calc)
                    atoms = result[0] if isinstance(result, tuple) else result
                elif stage == 7:
                    atoms = opt_cell.run(
                        atoms, stage_key="final_opt",
                        cfg_override=run_cfg, calc=calc)
                else:
                    raise ValueError(
                        f"Batch quench supports stages 5/6/7, got {stage}")

            res["final_cif"] = os.path.abspath(run_cfg["final_opt"]["output_cif"])
            res["final_xyz"] = os.path.abspath(run_cfg["final_opt"]["output_xyz"])
            res["success"]   = True

        except Exception as exc:
            res["error"] = traceback.format_exc()
            print(f"  [Run {run_id:04d}] ERROR: {exc}")
        finally:
            os.chdir(orig_dir)

        res["elapsed_min"] = (time.time() - t0) / 60
        results.append(res)
        _print_run_summary(res)

    _print_batch_summary(results, work_dir)
    return results


def run_from_traj(traj_file: str,
                  sample_interval_ps: float,
                  timestep_fs: float = 1.0,
                  n_runs: int | str = "all",
                  select: str = "uniform",
                  cfg_override: dict | None = None,
                  work_dir: str = "batch_quench",
                  stages: list[int] | None = None,
                  calc=None,
                  snapshot_dir: str = "snapshots_from_traj") -> list[dict]:
    """
    Extract frames from a trajectory file, then run batch quenching.

    Parameters
    ----------
    traj_file          : str    Path to .extxyz, .traj, or multi-frame .xyz
    sample_interval_ps : float  Picoseconds between snapshots
    timestep_fs        : float  Timestep used when the trajectory was recorded
    n_runs, select, cfg_override, work_dir, stages, calc  –  same as run()
    snapshot_dir       : str    Where to write extracted snapshot files

    Returns
    -------
    list[dict]  (same as run())
    """
    snap_paths = extract_snapshots(
        traj_file=traj_file,
        sample_interval_ps=sample_interval_ps,
        timestep_fs=timestep_fs,
        output_dir=snapshot_dir,
    )
    return run(snap_paths, n_runs=n_runs, select=select,
               cfg_override=cfg_override, work_dir=work_dir,
               stages=stages, calc=calc)


def extract_snapshots(traj_file: str,
                      sample_interval_ps: float,
                      timestep_fs: float = 1.0,
                      output_dir: str = "snapshots") -> list[str]:
    """
    Extract evenly-spaced frames from a trajectory and save each as .extxyz.

    Parameters
    ----------
    traj_file          : str    Input trajectory file
    sample_interval_ps : float  Picoseconds between saved frames
    timestep_fs        : float  MD timestep in femtoseconds
    output_dir         : str    Directory to write output files

    Returns
    -------
    list[str]  Paths to written snapshot files in time order
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[extract_snapshots] Reading: {traj_file}")
    frames   = read(traj_file, index=":")
    n_frames = len(frames)
    print(f"[extract_snapshots] {n_frames} frames found")

    steps_per_snap   = max(1, int(sample_interval_ps * 1000 / timestep_fs))
    selected_indices = list(range(0, n_frames, steps_per_snap))

    print(f"[extract_snapshots] Sampling every {sample_interval_ps} ps"
          f" (every {steps_per_snap} frames)  →  {len(selected_indices)} snapshots")

    paths = []
    for snap_idx, frame_idx in enumerate(selected_indices):
        atoms = frames[frame_idx]
        t_ps  = frame_idx * timestep_fs / 1000.0
        fname = os.path.join(
            output_dir, f"snapshot_{snap_idx:04d}_t{t_ps:.1f}ps.extxyz")
        write(fname, atoms, format="extxyz")
        paths.append(fname)
        print(f"  frame {frame_idx:6d}  t = {t_ps:8.1f} ps  →  {fname}")

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_complete(run_subdir: str, global_cfg: dict) -> bool:
    """Return True if the final output file for this run already exists."""
    p        = f"run{int(run_subdir.split('run_')[-1]):04d}_"
    cfg_fopt = global_cfg.get("final_opt", {})
    cif_name = p + cfg_fopt.get("output_cif", "stage7_amorphous_final.cif")
    xyz_name = p + cfg_fopt.get("output_xyz", "stage7_amorphous_final.xyz")
    cif_path = os.path.join(run_subdir, cif_name)
    xyz_path = os.path.join(run_subdir, xyz_name)
    return os.path.exists(cif_path) or os.path.exists(xyz_path)


def _find_output(run_subdir: str, key: str, global_cfg: dict) -> str | None:
    """Return path to an existing output file for a completed run."""
    p        = f"run{int(run_subdir.split('run_')[-1]):04d}_"
    cfg_fopt = global_cfg.get("final_opt", {})
    fname    = p + cfg_fopt.get(
        key, "stage7_amorphous_final.cif" if key == "output_cif"
             else "stage7_amorphous_final.xyz")
    full = os.path.join(run_subdir, fname)
    return full if os.path.exists(full) else None


def _select_frames(snapshot_files: list[str],
                   n_runs: int | str,
                   select: str) -> list[str]:
    """Select n_runs frames from snapshot_files using the given strategy."""
    import random as _random

    files   = list(snapshot_files)
    n_avail = len(files)
    if n_avail == 0:
        raise ValueError("snapshot_files is empty.")

    if n_runs == "all" or select == "all":
        print(f"[BatchQuench] Using all {n_avail} snapshots")
        return files

    n = min(int(n_runs), n_avail)
    if n < int(n_runs):
        print(f"[BatchQuench] WARNING: requested {n_runs} but only "
              f"{n_avail} available → using {n}")

    if select == "uniform":
        import numpy as _np
        indices = [int(i) for i in _np.linspace(0, n_avail - 1, n)]
        chosen  = [files[i] for i in indices]
    elif select == "random":
        chosen = _random.sample(files, n)
    elif select == "first":
        chosen = files[:n]
    elif select == "last":
        chosen = files[-n:]
    else:
        raise ValueError(
            f"Unknown select strategy '{select}'. "
            "Use 'uniform', 'random', 'first', 'last', or 'all'.")

    print(f"[BatchQuench] Selected {len(chosen)} snapshots "
          f"(strategy='{select}') from {n_avail} available")
    return chosen


def _make_run_cfg(global_cfg: dict, run_id: int) -> dict:
    """Build per-run config with unique output file name prefixes."""
    import copy
    cfg = copy.deepcopy(global_cfg)
    p   = f"run{run_id:04d}_"

    for stage_key, file_keys in [
        ("quench",    ["traj_file", "log_file", "output_cif", "output_xyz"]),
        ("eq_low",    ["traj_file", "log_file", "output_cif", "output_xyz"]),
        ("final_opt", ["traj_file", "logfile",  "output_cif", "output_xyz"]),
    ]:
        for k in file_keys:
            if k in cfg.get(stage_key, {}):
                cfg[stage_key][k] = p + cfg[stage_key][k]
    return cfg


def _print_run_summary(res: dict):
    status = "✓ OK" if res["success"] else "✗ FAILED"
    print(f"\n  {status}  run {res['run_id']:04d}  "
          f"({res['elapsed_min']:.1f} min)  "
          f"{os.path.basename(res['snapshot'])}")
    if res["final_cif"]:
        print(f"         → {res['final_cif']}")
    if res["error"]:
        print(f"         ERROR: {res['error'][:200]}")


def _print_batch_summary(results: list[dict], work_dir: str):
    n_ok       = sum(r["success"] for r in results)
    n_fail     = len(results) - n_ok
    total_min  = sum(r["elapsed_min"] for r in results)

    print(f"\n{'='*62}")
    print(f"  BATCH QUENCH COMPLETE")
    print(f"  Runs:   {len(results)}  |  OK: {n_ok}  |  Failed: {n_fail}")
    print(f"  Total wall time: {total_min:.1f} min")
    print(f"  Outputs in: {work_dir}/run_XXXX/")
    print(f"{'='*62}")

    if n_fail:
        print("\n  Failed runs:")
        for r in results:
            if not r["success"]:
                print(f"    run_{r['run_id']:04d}  ←  {r['snapshot']}")
                if r["error"]:
                    print(f"      {r['error'][:300]}")
