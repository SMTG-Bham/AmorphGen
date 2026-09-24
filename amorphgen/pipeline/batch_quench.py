"""
amorphgen.pipeline.batch_quench
--------------------------------
Run a subset of pipeline stages independently on each of N input
structures, producing a library of amorphous candidates.

Typical use cases:

* **MQ snapshot quench** (default ``stages=[5, 6, 7]``): take N snapshots
  extracted from a Stage 4 high-T equilibration trajectory and quench each
  through stages 5 (cooling) -> 6 (low-T eq) -> 7 (final opt).
* **Hybrid workflow** (``stages=[4, 5, 6, 7]``): take N already-disordered
  inputs (e.g. ``--random-gen`` outputs), anneal at high T, then quench.

Stage numbers follow the canonical 7-stage pipeline:
4 = eq_high, 5 = quench, 6 = eq_low, 7 = final_opt.
"""

from __future__ import annotations

import os
import json
import numpy as np
import re
from copy import deepcopy

from ase.io import read, write, iread


def _run_dir_name(snap_file: str, fallback_idx: int) -> str:
    """Pick a self-documenting run-dir name from a snapshot filename.

    ``snapshot_0007_frame00184.extxyz`` -> ``run_0007``.
    If no leading ``snapshot_NNNN`` index is parseable, fall back to
    ``run_{fallback_idx:04d}`` so the loop's enumerate index is preserved.
    """
    base = os.path.splitext(os.path.basename(snap_file))[0]
    m = re.match(r"snapshot[_-]?(\d+)", base)
    if m:
        return f"run_{int(m.group(1)):04d}"
    return f"run_{fallback_idx:04d}"

from ..utils import get_calculator, merge_config
from ..configs import DEFAULT_CONFIG
from . import quench, equilibrate, final_opt


def run(snapshot_files: list[str],
        n_runs: int | None = None,
        select: str = "uniform",
        cfg_override: dict | None = None,
        work_dir: str = "batch_quench",
        stages: list[int] | None = None,
        calc=None,
        resume: bool = False):
    """
    Batch quench multiple snapshots.

    Parameters
    ----------
    snapshot_files : list of str
        Paths to snapshot structure files.
    n_runs : int, optional
        Number of runs (defaults to len(snapshot_files)).
    select : str
        How to select snapshots: "uniform" or "last".
    cfg_override : dict, optional
    work_dir : str
        Base output directory.
    stages : list of int
        Which stages to run per snapshot. Stage numbers follow the canonical
        7-stage pipeline: 4=eq_high, 5=quench, 6=eq_low, 7=final_opt.
        Default ``[5, 6, 7]`` (quench + eq_low + final_opt — the standard
        post-Stage-4 batch workflow). Include 4 when starting from random /
        already-disordered structures that need to be annealed first
        (the 'hybrid' workflow).
    calc : ASE calculator, optional
    resume : bool
        If True, skip runs whose final output already exists.
    """
    if stages is None:
        stages = [5, 6, 7]

    global_cfg = merge_config(DEFAULT_CONFIG, cfg_override)
    os.makedirs(work_dir, exist_ok=True)

    if n_runs is None:
        n_runs = len(snapshot_files)

    # Select subset
    import numpy as np
    n_available = len(snapshot_files)
    if select == "uniform":
        indices = np.linspace(0, n_available - 1, min(n_runs, n_available), dtype=int)
    else:
        indices = list(range(max(0, n_available - n_runs), n_available))
    selected = [snapshot_files[i] for i in indices]

    # Build calculator once
    if calc is None:
        from ..utils.common import resolve_device
        device = resolve_device(global_cfg.get("device", "cuda"))
        calc = get_calculator(
            model=global_cfg.get("model", "mace-mpa-0"),
            device=device,
            model_path=global_cfg.get("model_path"),
        )

    bar = "=" * 65
    print(f"\n{bar}")
    print(f"  Batch quench: {len(selected)} runs, stages {stages}")
    print(f"  Output: {work_dir}/")
    print(f"{bar}\n")

    # v1.0.0rc2: when the caller passes exactly ONE snapshot (typical of
    # SLURM array workflows where each task processes a single input),
    # skip the per-run ``run_NNNN/`` subdir and write outputs directly
    # to ``work_dir``. Multi-snapshot runs (typical local use) still get
    # the ``run_NNNN/`` separation between runs.
    single_run = len(selected) == 1

    results = []
    for i, snap_file in enumerate(selected):
        run_name = _run_dir_name(snap_file, fallback_idx=i)
        run_dir = work_dir if single_run else os.path.join(work_dir, run_name)
        final_output = os.path.join(run_dir, "final_amorphous.xyz")
        legacy_final = os.path.join(run_dir, "final_amorphous.extxyz")

        if resume and (os.path.isfile(final_output) or os.path.isfile(legacy_final)):
            existing = final_output if os.path.isfile(final_output) else legacy_final
            label = work_dir if single_run else run_name
            print(f"  [{label}] Already complete -- skipping.")
            results.append(read(existing))
            continue

        os.makedirs(run_dir, exist_ok=True)
        target = work_dir if single_run else f"{run_name}/"
        print(f"\n  {'-' * 60}")
        print(f"  Run {i+1:04d} / {len(selected)}  <-  {os.path.basename(snap_file)}  ->  {target}")
        print(f"  {'-' * 60}")

        atoms = read(snap_file)
        atoms.calc = calc
        orig_dir = os.getcwd()
        os.chdir(run_dir)

        try:
            # MD stages get the resume flag for FRAME-level resume within
            # this run: safe because we chdir into the per-run directory
            # above, so a run can only ever see its own stage trajectories.
            # A stage whose trajectory is already complete resumes with 0
            # remaining steps (cheap skip); optimisation (7) restarts whole.
            for s in stages:
                if s == 4:
                    atoms = equilibrate.run(atoms, cfg_override=cfg_override,
                                            calc=calc, stage="high",
                                            resume=resume)
                elif s == 5:
                    atoms = quench.run(atoms, cfg_override=cfg_override,
                                       calc=calc, resume=resume)
                elif s == 6:
                    atoms = equilibrate.run(atoms, cfg_override=cfg_override,
                                            calc=calc, stage="low",
                                            resume=resume)
                elif s == 7:
                    atoms = final_opt.run(atoms, cfg_override=cfg_override, calc=calc)
                else:
                    raise ValueError(
                        f"batch_quench: unknown stage {s}. "
                        f"Allowed: 4 (eq_high), 5 (quench), 6 (eq_low), 7 (final_opt)."
                    )
        finally:
            os.chdir(orig_dir)

        write(final_output, atoms, format="extxyz")
        results.append(atoms)
        from ..utils.common import compute_density_gcm3
        d = compute_density_gcm3(atoms)
        print(f"  [{run_name}] Done -> {final_output}  density={d:.2f} g/cm3")

    print(f"\n{bar}")
    print(f"  Batch complete: {len(results)} structures generated")
    print(f"{bar}\n")
    return results


def _batched_stage_checkpoint(dirs, logname, trajname, endname, n_steps, interval, resume):
    """Frame-level resume for one batched MD stage across a chunk of runs.

    Returns ``(start_atoms, done_steps, complete)``:

    * ``complete`` is True when every run already has the stage's end file,
      so the stage is skipped and the next one starts from those files;
    * otherwise ``done_steps`` is the largest multiple of ``interval`` that
      EVERY run's ``stageN_*_traj.xyz`` has reached.  Runs that got further
      (a kill mid-write hits the chunk unevenly) are truncated to it so the
      chunk stays synchronised, and ``start_atoms`` are the frames at that
      step with momenta.  ``(None, 0, False)`` means start from scratch.
    """
    if not resume:
        return None, 0, False
    if all(os.path.isfile(os.path.join(d, endname)) for d in dirs):
        return [read(os.path.join(d, endname)) for d in dirs], n_steps, True
    frames = []
    for d in dirs:
        path = os.path.join(d, trajname)
        if not os.path.isfile(path):
            return None, 0, False
        fr = []
        try:
            for atoms in iread(path, index=":"):      # tolerate a torn last frame
                fr.append(atoms)
        except Exception:
            pass
        frames.append(fr)
    n_common = min(len(f) for f in frames)
    if n_common == 0:
        return None, 0, False
    # the last block of a stage may be shorter than ``interval`` (frame k sits
    # at step min(k * interval, n_steps)), so clamp; a trajectory that reached
    # the end but lost its end file (killed while writing it) counts as complete
    done = min(n_common * interval, n_steps)
    if done >= n_steps:
        for d, fr in zip(dirs, frames):
            if not os.path.isfile(os.path.join(d, endname)):
                write(os.path.join(d, endname), fr[n_common - 1], format="extxyz")
        return [fr[n_common - 1] for fr in frames], n_steps, True
    for d, fr in zip(dirs, frames):
        if len(fr) > n_common:                   # ran ahead of the chunk, or torn: cut back
            write(os.path.join(d, trajname), fr[:n_common], format="extxyz")
        logpath = os.path.join(d, logname)
        if os.path.isfile(logpath):
            lines = open(logpath).read().splitlines()
            is_row = [bool(l.strip()) and l.split()[0].isdigit() for l in lines]
            head = [l for l, r in zip(lines, is_row) if not r]
            body = [l for l, r in zip(lines, is_row) if r][:n_common]
            open(logpath, "w").write("\n".join(head + body) + "\n")
    return [fr[n_common - 1] for fr in frames], done, False


def run_torchsim(snapshot_files: list[str], cfg_override: dict | None = None,
                 work_dir: str = "batch_quench", stages: list[int] | None = None,
                 resume: bool = False, batch_size: int | None = None) -> list[str]:
    """Batched torch-sim version of :func:`run` for the hybrid workflow.

    All runs advance together through the NVT stages (4 = eq_high,
    5 = quench, 6 = eq_low) in one batched integration, then stage 7 is a
    batched relaxation. Per-run directories, log and trajectory names match
    the ASE path. Resume is at run level (runs with ``final_amorphous.xyz``
    are skipped). NPT stages are not supported by this engine.

    Returns the list of ``final_amorphous.xyz`` paths.
    """
    from ..utils.common import resolve_ramp
    from ..utils.torchsim_engine import build_model, batch_relax
    from ..utils.torchsim_md import batch_nvt, _RunWriter
    from ..utils.common import TRAJ_LOG_INTERVAL

    stages = stages or [4, 5, 6, 7]
    cfg = merge_config(DEFAULT_CONFIG, cfg_override)
    for key, st in (("eq_high", 4), ("quench", 5), ("eq_low", 6)):
        if st in stages and str(cfg[key].get("ensemble", "NVT")).upper() != "NVT":
            raise ValueError(f"torch-sim engine: stage {st} ({key}) must be NVT "
                             f"(got {cfg[key].get('ensemble')}); NPT stages need the ASE engine.")
    os.makedirs(work_dir, exist_ok=True)
    seed = cfg.get("seed")
    batch_size = batch_size or (cfg.get("opt", {}) or {}).get("batch_size") or "auto"

    runs = []                      # (run_dir, snapshot_file)
    finals = []
    for i, f in enumerate(snapshot_files):
        run_dir = os.path.join(work_dir, _run_dir_name(f, fallback_idx=i))
        final = os.path.join(run_dir, "final_amorphous.xyz")
        if resume and os.path.isfile(final):
            finals.append(final)
            continue
        runs.append((run_dir, f))
    if resume and finals:
        print(f"  [Resume] {len(finals)} run(s) already complete, {len(runs)} to do")
    if not runs:
        return finals

    model = build_model(cfg.get("model", "mace-mpa-0"), device=cfg.get("device", "auto"),
                        model_path=cfg.get("model_path"), classical_params=cfg.get("classical_params"),
                        dtype="float64" if cfg.get("default_dtype") in (None, "auto") else cfg["default_dtype"])
    size_file = os.path.join(work_dir, "batch_size.json")
    if str(batch_size).lower() == "auto" and resume and os.path.isfile(size_file):
        batch_size = json.load(open(size_file))["batch_size"]     # same chunking as the killed job
        print(f"  [Resume] chunk size {batch_size} taken from {size_file}")
    if str(batch_size).lower() == "auto":
        from ..utils.torchsim_engine import estimate_batch_size
        batch_size = estimate_batch_size(model, [read(f) for _, f in runs[:4]], fraction=0.5,
                                         md=True, fallback=16)
    batch_size = int(batch_size)
    json.dump({"batch_size": batch_size}, open(size_file, "w"))
    bar = "=" * 65
    print(f"\n{bar}\n  Batch quench (torch-sim engine): {len(runs)} runs, stages {stages}, "
          f"chunks of {batch_size}\n  Output: {work_dir}/\n{bar}")

    for c0 in range(0, len(runs), batch_size):
        chunk = runs[c0:c0 + batch_size]
        dirs = [d for d, _ in chunk]
        atoms = [read(f) for _, f in chunk]
        if len(runs) > batch_size:
            print(f"  [torch-sim] chunk {c0 // batch_size + 1}/{(len(runs) + batch_size - 1) // batch_size}")

        if 4 in stages:
            c = cfg["eq_high"]; n = int(c["steps"])
            start, done, complete = _batched_stage_checkpoint(dirs, "stage4_eq.log", "stage4_eq_traj.xyz",
                                                              "stage4_eq.xyz", n, TRAJ_LOG_INTERVAL, resume)
            if complete:
                print("  [Stage 4] already complete for this chunk -- skipping"); atoms = start
            else:
                if start is not None:
                    print(f"  [Stage 4] resuming from step {done}/{n}"); atoms = start
                ws = [_RunWriter(d, "stage4_eq.log", "stage4_eq_traj.xyz", append=done > 0, step_offset=done) for d in dirs]
                print(f"  [Stage 4] NVT {c['T']} K, {n - done} steps")
                atoms = batch_nvt(atoms, model, float(c["T"]), n - done, timestep_fs=float(c.get("timestep", 0.5)),
                                  friction=float(c.get("friction", 0.01)), seed=seed, stage=4, writers=ws)
                for d, a in zip(dirs, atoms):
                    write(os.path.join(d, "stage4_eq.xyz"), a, format="extxyz")
        if 5 in stages:
            c = cfg["quench"]; dt = float(c.get("timestep", 0.5))
            temps = resolve_ramp(c["T_start"], c["T_end"], c.get("T_step", -100))
            rate = c.get("rate")
            if rate is not None:
                spt = max(1, int(round(abs(float(c.get("T_step", -100))) / (abs(float(rate)) * dt / 1000))))
            else:
                spt = int(c.get("steps_per_T", 1000))
            sched = np.repeat(temps, spt)
            start, done, complete = _batched_stage_checkpoint(dirs, "stage5_quench.log", "stage5_quench_traj.xyz",
                                                              "stage5_quenched.xyz", len(sched), TRAJ_LOG_INTERVAL, resume)
            if complete:
                print("  [Stage 5] already complete for this chunk -- skipping"); atoms = start
            else:
                if start is not None:
                    print(f"  [Stage 5] resuming from step {done}/{len(sched)}"); atoms = start
                ws = [_RunWriter(d, "stage5_quench.log", "stage5_quench_traj.xyz", append=done > 0, step_offset=done) for d in dirs]
                print(f"  [Stage 5] quench {c['T_start']} -> {c['T_end']} K, {len(temps)} segments x {spt} steps"
                      + (f" (from step {done})" if done else ""))
                atoms = batch_nvt(atoms, model, sched[done:], len(sched) - done, timestep_fs=dt,
                                  friction=float(c.get("friction", 0.01)), seed=seed, stage=5, writers=ws)
                for d, a in zip(dirs, atoms):
                    write(os.path.join(d, "stage5_quenched.xyz"), a, format="extxyz")
        if 6 in stages:
            c = cfg["eq_low"]; n = int(c["steps"])
            start, done, complete = _batched_stage_checkpoint(dirs, "stage6_eq.log", "stage6_eq_traj.xyz",
                                                              "stage6_eq.xyz", n, TRAJ_LOG_INTERVAL, resume)
            if complete:
                print("  [Stage 6] already complete for this chunk -- skipping"); atoms = start
            else:
                if start is not None:
                    print(f"  [Stage 6] resuming from step {done}/{n}"); atoms = start
                ws = [_RunWriter(d, "stage6_eq.log", "stage6_eq_traj.xyz", append=done > 0, step_offset=done) for d in dirs]
                print(f"  [Stage 6] NVT {c['T']} K, {n - done} steps")
                atoms = batch_nvt(atoms, model, float(c["T"]), n - done, timestep_fs=float(c.get("timestep", 0.5)),
                                  friction=float(c.get("friction", 0.01)), seed=seed, stage=6, writers=ws)
                for d, a in zip(dirs, atoms):
                    write(os.path.join(d, "stage6_eq.xyz"), a, format="extxyz")
        if 7 in stages:
            c = cfg.get("final_opt") or cfg["opt"]
            print(f"  [Stage 7] batched relaxation, fmax {c.get('fmax', 0.01)}, cell filter {c.get('cell_filter', 'cubic')}")
            for a in atoms:
                a.set_momenta(np.zeros_like(a.positions))
            atoms = batch_relax(atoms, model, fmax=float(c.get("fmax", 0.01)), max_steps=int(c.get("max_steps", 1000)),
                                cell_filter=str(c.get("cell_filter", "cubic")), optimizer=str(c.get("optimizer", "LBFGS")),
                                pressure_tol_gpa=float(c.get("pressure_tol_gpa", 0.02)))
            for d, a in zip(dirs, atoms):
                write(os.path.join(d, "stage7_opt.xyz"), a, format="extxyz")
        for d, a in zip(dirs, atoms):
            final = os.path.join(d, "final_amorphous.xyz")
            write(final, a, format="extxyz"); finals.append(final)
    return finals
