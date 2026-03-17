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

            self._calc = get_calculator(
                model=self.cfg.get("model", "mace-mpa-0"),
                device=device,
                model_path=self.cfg.get("model_path"),
            )
        return self._calc

    # ─────────────────────────────────────────────────────────────────────────

    def run(self,
            stages: list[int] | None = None,
            input_file: str | None = None) -> object:
        """
        Execute the pipeline.

        Parameters
        ----------
        stages : list of int, optional
            Which stages to run (default: all, i.e. ``[1, 2, 3, 4, 5, 6, 7]``).
        input_file : str, optional
            Override the input structure file (useful for resuming from a
            mid-pipeline checkpoint).

        Returns
        -------
        ase.Atoms
            The final optimised amorphous structure.
        """
        if stages is None:
            stages = [1, 2, 3, 4, 5, 6, 7]
        if input_file is None:
            input_file = self.input_file

        atoms = read(input_file)
        calc = self._get_calc()
        atoms.calc = calc

        model_name = self.cfg.get("model", "mace-mpa-0")
        model_path = self.cfg.get("model_path")
        model_display = model_path if model_path else model_name

        bar = "═" * 65
        print(f"\n{bar}")
        print(f"  AmorphGen  v2.0  —  Melt-and-Quench Pipeline")
        print(f"  Model:  {model_display}")
        print(f"  Input:  {input_file}")
        print(f"  Stages: {stages}")
        print(f"  Output: {self.work_dir}/")
        print(f"{bar}\n")

        os.chdir(self.work_dir)
        t0 = time.time()

        try:
            for s in stages:
                name = self.STAGE_NAMES.get(s, f"Stage {s}")
                print(f"\n{'─' * 65}")
                print(f"  Stage {s}: {name}")
                print(f"{'─' * 65}\n")

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
                    print(f"  ⚠ Unknown stage {s} — skipping.")

        finally:
            os.chdir(self._orig_dir)

        elapsed = time.time() - t0
        print(f"\n{bar}")
        print(f"  Pipeline complete  ({elapsed / 60:.1f} min)")
        print(f"  Output directory:  {self.work_dir}/")
        print(f"{bar}\n")

        return atoms
