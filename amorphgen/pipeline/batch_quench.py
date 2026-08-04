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
import re
from copy import deepcopy

from ase.io import read, write


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
