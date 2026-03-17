"""
amorphgen.pipeline.batch_quench
--------------------------------
Run quench + equilibration + final optimisation on multiple snapshots
from a high-T trajectory.

This module takes snapshot files (extracted from Stage 3's trajectory)
and runs Stages 4-5-6 independently on each one, producing a library
of amorphous candidate structures.
"""

from __future__ import annotations

import os
from copy import deepcopy

from ase.io import read, write

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
        Which stages to run per snapshot (default [4, 5, 6]).
    calc : ASE calculator, optional
    resume : bool
        If True, skip runs whose final output already exists.
    """
    if stages is None:
        stages = [4, 5, 6]

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
        device = global_cfg.get("device", "cuda")
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
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

    results = []
    for i, snap_file in enumerate(selected):
        run_dir = os.path.join(work_dir, f"run_{i:04d}")
        final_output = os.path.join(run_dir, "final_amorphous.extxyz")

        if resume and os.path.isfile(final_output):
            print(f"  [run_{i:04d}] Already complete -- skipping.")
            results.append(read(final_output))
            continue

        os.makedirs(run_dir, exist_ok=True)
        print(f"\n  {'─' * 60}")
        print(f"  Run {i:04d} / {len(selected)-1}  <-  {os.path.basename(snap_file)}")
        print(f"  {'─' * 60}")

        atoms = read(snap_file)
        atoms.calc = calc
        orig_dir = os.getcwd()
        os.chdir(run_dir)

        try:
            for s in stages:
                if s == 4:
                    atoms = quench.run(atoms, cfg_override=cfg_override, calc=calc)
                elif s == 5:
                    atoms = equilibrate.run(atoms, cfg_override=cfg_override,
                                           calc=calc, stage="low")
                elif s == 6:
                    atoms = final_opt.run(atoms, cfg_override=cfg_override, calc=calc)
        finally:
            os.chdir(orig_dir)

        write(final_output, atoms, format="extxyz")
        results.append(atoms)
        print(f"  [run_{i:04d}] Done -> {final_output}")

    print(f"\n{bar}")
    print(f"  Batch complete: {len(results)} structures generated")
    print(f"{bar}\n")
    return results
