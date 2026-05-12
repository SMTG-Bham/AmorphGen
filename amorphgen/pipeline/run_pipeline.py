"""
amorphgen.pipeline.run_pipeline
--------------------------------
Orchestrates the full melt-and-quench pipeline.

Usage
-----
    from amorphgen import MeltQuenchPipeline

    # Default MACE model
    pipe = MeltQuenchPipeline(
        input_file="POSCAR",
        work_dir="my_run",
        cfg_override={"model": "mace-mpa-0", "device": "cuda"},
    )
    final_atoms = pipe.run()

    )

    # CHGNet
    pipe = MeltQuenchPipeline(
        input_file="POSCAR",
        cfg_override={"model": "chgnet"},
    )

    # Custom fine-tuned MACE model
    pipe = MeltQuenchPipeline(
        input_file="POSCAR",
        cfg_override={"model_path": "/data/models/InO_finetuned.model"},
    )

Resuming from a checkpoint
--------------------------
    pipe.run(stages=[5, 6, 7], input_file="stage4_eq_high.xyz")
"""

import os
import time
import platform
from datetime import datetime

from ase.io import read

from . import opt_cell, melt_cell, equilibrate, quench, final_opt
from ..utils import get_calculator, merge_config
from ..configs import DEFAULT_CONFIG


class MeltQuenchPipeline:
    """
    End-to-end melt-and-quench pipeline for amorphous structure generation.

    Parameters
    ----------
    input_file : str
        Path to the starting crystalline structure (any ASE-readable format).
    work_dir : str
        Directory where all output files are written.  Created if absent.
    cfg_override : dict, optional
        Any keys in DEFAULT_CONFIG to override, including:

        * ``"model"``      : foundation model short name (any backend)
        * ``"model_path"`` : path to local .model file (overrides model)
        * ``"device"``     : ``"cuda"`` or ``"cpu"``
        * ``"eq_premelt"`` : ``{"ensemble": "NVT" or "NPT", ...}``
        * ``"melt"``       : ``{"ensemble": "NVT" or "NPT", ...}``
        * ``"quench"``     : ``{"ensemble": "NVT" or "NPT", ...}``
        * ``"eq_high"``    : ``{"ensemble": "NVT" or "NPT", ...}``
        * ``"eq_low"``     : ``{"ensemble": "NVT" or "NPT", ...}``

    share_calc : bool
        If True, one calculator is shared across all stages.
    """

    STAGE_NAMES = {
        1: "Structure optimisation (crystalline)",
        2: "Pre-melt equilibration (300 K)",
        3: "Melt (heat ramp)",
        4: "High-T equilibration",
        5: "Quench (cooling ramp)",
        6: "Low-T equilibration",
        7: "Final optimisation (amorphous)",
    }

    # Checkpoint files written by each stage (checked in reverse order for resume)
    STAGE_CHECKPOINTS = {
        1: "stage1_opt.xyz",
        2: "stage2_eq.xyz",
        3: "stage3_melted.xyz",
        4: "stage4_eq.xyz",
        5: "stage5_quenched.xyz",
        6: "stage6_eq.xyz",
        7: "stage7_opt.xyz",
    }

    def __init__(self, input_file: str,
                 work_dir: str = "melt_quench_run",
                 cfg_override: dict | None = None,
                 share_calc: bool = True):

        self.input_file = input_file
        self.work_dir   = work_dir
        self.cfg        = merge_config(DEFAULT_CONFIG, cfg_override)
        self.share_calc = share_calc
        self._calc      = None
        self._orig_dir  = os.getcwd()
        os.makedirs(work_dir, exist_ok=True)

        # Handle legacy "mace_model" key → "model"
        if self.cfg.get("mace_model") and not self.cfg.get("model"):
            self.cfg["model"] = self.cfg["mace_model"]

    # ─────────────────────────────────────────────────────────────────────────

    def _get_calc(self):
        """Build or return the shared calculator."""
        if self._calc is None or not self.share_calc:
            device = self.cfg.get("device", "cuda")
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            calc_kwargs = {}
            if self.cfg.get("classical_params"):
                calc_kwargs["classical_params"] = self.cfg["classical_params"]
            self._calc = get_calculator(
                model=self.cfg.get("model", "mace-mpa-0"),
                device=device,
                model_path=self.cfg.get("model_path"),
                **calc_kwargs,
            )
        return self._calc

    # ─────────────────────────────────────────────────────────────────────────

    def _find_resume_point(self, stages: list[int]) -> tuple[list[int], str]:
        """
        Scan work_dir for completed stage checkpoints and return the
        remaining stages and the input file to resume from.

        Parameters
        ----------
        stages : list of int
            The originally requested stages.

        Returns
        -------
        remaining_stages : list of int
            Stages that still need to run.
        resume_input : str
            Path to the checkpoint file to resume from, or the original
            input_file if no checkpoints are found.
        """
        resume_input = self.input_file
        remaining_stages = list(stages)

        # Walk stages in order; if a checkpoint exists, advance past it
        for s in stages:
            checkpoint = self.STAGE_CHECKPOINTS.get(s)
            if checkpoint is None:
                break
            checkpoint_path = os.path.join(self.work_dir, checkpoint)
            if os.path.isfile(checkpoint_path):
                resume_input = checkpoint_path
                remaining_stages.remove(s)
            else:
                break  # first missing checkpoint → start here

        return remaining_stages, resume_input

    # ─────────────────────────────────────────────────────────────────────────

    def run(self,
            stages: list[int] | None = None,
            input_file: str | None = None,
            resume: bool = False) -> object:
        """
        Execute the pipeline.

        Parameters
        ----------
        stages : list of int, optional
            Which stages to run (default: all, i.e. ``[1, 2, 3, 4, 5, 6, 7]``).
        input_file : str, optional
            Override the input structure file (useful for resuming from a
            mid-pipeline checkpoint).
        resume : bool
            If True, scan work_dir for completed stage checkpoints and
            skip stages that already have output files.  Automatically
            determines which stage to resume from and which input file
            to use.

        Returns
        -------
        ase.Atoms
            The final optimised amorphous structure.
        """
        if stages is None:
            stages = [1, 2, 3, 4, 5, 6, 7]
        if input_file is None:
            input_file = self.input_file

        if resume:
            remaining, resume_input = self._find_resume_point(stages)
            if not remaining:
                print(f"  All stages already completed in {self.work_dir}/")
                final_checkpoint = os.path.join(
                    self.work_dir,
                    self.STAGE_CHECKPOINTS[stages[-1]],
                )
                return read(final_checkpoint)
            if remaining != stages:
                skipped = [s for s in stages if s not in remaining]
                print(f"  Resuming: skipping completed stages {skipped}")
                print(f"  Starting from stage {remaining[0]} "
                      f"(input: {os.path.basename(resume_input)})")
                input_file = resume_input
            stages = remaining

        atoms = read(input_file)
        calc = self._get_calc()
        atoms.calc = calc

        model_name = self.cfg.get("model", "mace-mpa-0")
        model_path = self.cfg.get("model_path")
        model_display = model_path if model_path else model_name
        device = self.cfg.get("device", "cuda")
        n_atoms = len(atoms)
        formula = atoms.get_chemical_formula(mode="hill")

        bar = "=" * 65
        print(f"\n{bar}")
        from .. import __version__
        print(f"  AmorphGen  v{__version__}  -  Melt-and-Quench Pipeline")
        from ..utils.common import compute_density_gcm3
        density = compute_density_gcm3(atoms)
        print(f"  Model:  {model_display}")
        print(f"  Device: {device}")
        print(f"  Input:  {input_file}")
        print(f"  System: {formula} ({n_atoms} atoms)")
        print(f"  Density: {density:.2f} g/cm3")
        print(f"  Stages: {stages}")
        print(f"  Output: {self.work_dir}/")
        print(f"{bar}\n")

        os.chdir(self.work_dir)
        t0 = time.time()
        stage_timings = []

        try:
            for s in stages:
                name = self.STAGE_NAMES.get(s, f"Stage {s}")
                print(f"\n{'-' * 65}")
                print(f"  Stage {s}: {name}")
                print(f"{'-' * 65}\n")

                t_stage = time.time()

                if s == 1:
                    atoms = opt_cell.run(atoms, self.cfg, calc)
                elif s == 2:
                    atoms = equilibrate.run(atoms, self.cfg, calc, stage="premelt")
                elif s == 3:
                    atoms = melt_cell.run(atoms, self.cfg, calc)
                elif s == 4:
                    atoms = equilibrate.run(atoms, self.cfg, calc, stage="high")
                elif s == 5:
                    atoms = quench.run(atoms, self.cfg, calc)
                elif s == 6:
                    atoms = equilibrate.run(atoms, self.cfg, calc, stage="low")
                elif s == 7:
                    atoms = final_opt.run(atoms, self.cfg, calc)
                else:
                    print(f"  WARNING: Unknown stage {s} - skipping.")
                    continue

                dt = time.time() - t_stage
                stage_timings.append((s, name, dt))
                d = compute_density_gcm3(atoms)
                print(f"  [Stage {s} completed in {dt:.1f} s ({dt/60:.1f} min) "
                      f"| density={d:.2f} g/cm3]")

        finally:
            os.chdir(self._orig_dir)

        elapsed = time.time() - t0

        # Print summary
        print(f"\n{bar}")
        print(f"  Pipeline complete  ({elapsed / 60:.1f} min)")
        print(f"  Output directory:  {self.work_dir}/")
        print(f"{bar}\n")

        # Write pipeline summary log
        self._write_summary_log(
            input_file, formula, n_atoms, model_display, device,
            stages, stage_timings, elapsed
        )

        return atoms

    def _write_summary_log(self, input_file, formula, n_atoms,
                           model_display, device, stages,
                           stage_timings, total_elapsed):
        """Write a pipeline_summary.log file with timing and config."""
        logfile = os.path.join(self.work_dir, "pipeline_summary.log")
        bar = "=" * 65

        with open(logfile, "w") as f:
            f.write(f"{bar}\n")
            f.write(f"  AmorphGen — Pipeline Summary\n")
            f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Platform: {platform.platform()}\n")
            f.write(f"  Python: {platform.python_version()}\n")
            f.write(f"{bar}\n\n")

            f.write(f"  System:  {formula} ({n_atoms} atoms)\n")
            f.write(f"  Model:   {model_display}\n")
            f.write(f"  Device:  {device}\n")
            f.write(f"  Input:   {input_file}\n")
            f.write(f"  Output:  {self.work_dir}/\n")
            f.write(f"  Stages:  {stages}\n\n")

            # Stage timings
            f.write(f"  {'Stage':<45} {'Time (s)':>10} {'Time (min)':>12}\n")
            f.write(f"  {'-'*69}\n")
            for s, name, dt in stage_timings:
                f.write(f"  {s}. {name:<42} {dt:>10.1f} {dt/60:>11.1f}\n")
            f.write(f"  {'-'*69}\n")
            f.write(f"  {'Total':<45} {total_elapsed:>10.1f} "
                    f"{total_elapsed/60:>11.1f}\n\n")

            # Per-atom timing
            if n_atoms > 0:
                f.write(f"  Per-atom total: {total_elapsed/n_atoms:.2f} s/atom\n")
                for s, name, dt in stage_timings:
                    f.write(f"  Per-atom stage {s}: {dt/n_atoms:.2f} s/atom\n")
                f.write(f"\n")

            # Key config parameters
            f.write(f"  Configuration:\n")
            for key in ["opt", "eq_premelt", "melt", "eq_high",
                        "quench", "eq_low"]:
                if key in self.cfg:
                    f.write(f"    {key}:\n")
                    for k, v in self.cfg[key].items():
                        f.write(f"      {k}: {v}\n")

            f.write(f"\n{bar}\n")

        print(f"  Summary log: {logfile}")
