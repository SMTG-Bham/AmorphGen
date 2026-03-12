"""
pipeline/run_pipeline.py
------------------------
Orchestrates the 7-stage melt-and-quench pipeline.

Stage 1 : Optimise crystalline input cell
Stage 2 : Pre-melt equilibration at 300 K   (NVT, 50 ps)
Stage 3 : Melt  -  NPT heat ramp  300 -> T_melt
Stage 4 : High-T equilibration at T_melt    (NVT, 100 ps)
Stage 5 : Quench  -  NVT cooling ramp  T_melt -> 300 K
Stage 6 : Low-T equilibration at 300 K      (NVT, 50 ps)
Stage 7 : Final optimisation  ->  amorphous structure
"""

import os
import time

from ase.io import read

from . import opt_cell, melt_cell, equilibrate, quench, batch_quench
from ..utils import get_mace_calculator, merge_config
from ..configs import DEFAULT_CONFIG


class MeltQuenchPipeline:

    STAGE_NAMES = {
        1: "Structure Optimisation   (crystalline)",
        2: "Pre-melt Equilibration   300 K",
        3: "Melt  -  heat ramp to T_melt",
        4: "High-T Equilibration     at T_melt",
        5: "Quench  -  cooling ramp to 300 K",
        6: "Low-T Equilibration      300 K",
        7: "Final Optimisation       (amorphous output)",
    }

    def __init__(self,
                 input_file: str,
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

    def run(self,
            stages: list[int] | None = None,
            input_file: str | None = None):

        if stages is None:
            stages = [1, 2, 3, 4, 5, 6, 7]

        os.chdir(self.work_dir)
        snapshot_paths = None

        try:
            src   = os.path.join(self._orig_dir, input_file or self.input_file)
            atoms = read(src)
            calc  = self._get_calc() if self.share_calc else None
            t_start = time.time()

            for stage in stages:
                self._banner(stage)
                t0 = time.time()

                if stage == 1:
                    atoms = opt_cell.run(
                        atoms, stage_key="opt",
                        cfg_override=self.cfg, calc=calc)

                elif stage == 2:
                    result = equilibrate.run(
                        atoms, stage_key="eq_premelt",
                        cfg_override=self.cfg, calc=calc)
                    atoms = result[0] if isinstance(result, tuple) else result

                elif stage == 3:
                    atoms = melt_cell.run(
                        atoms, cfg_override=self.cfg, calc=calc)

                elif stage == 4:
                    result = equilibrate.run(
                        atoms, stage_key="eq_high",
                        cfg_override=self.cfg, calc=calc)
                    if isinstance(result, tuple):
                        atoms, snapshot_paths = result
                    else:
                        atoms = result

                elif stage == 5:
                    atoms = quench.run(
                        atoms, cfg_override=self.cfg, calc=calc)

                elif stage == 6:
                    result = equilibrate.run(
                        atoms, stage_key="eq_low",
                        cfg_override=self.cfg, calc=calc)
                    atoms = result[0] if isinstance(result, tuple) else result

                elif stage == 7:
                    atoms = opt_cell.run(
                        atoms, stage_key="final_opt",
                        cfg_override=self.cfg, calc=calc)

                else:
                    raise ValueError(
                        f"Unknown stage {stage}. Valid stages: 1-7.")

                elapsed = (time.time() - t0) / 60
                print(f"  Stage {stage} done in {elapsed:.1f} min\n")

            total = (time.time() - t_start) / 60
            print("=" * 62)
            print(f"  Pipeline complete  -  total: {total:.1f} min")
            print("=" * 62)

        finally:
            os.chdir(self._orig_dir)

        if snapshot_paths is not None:
            return atoms, snapshot_paths
        return atoms

    def _get_calc(self):
        if self._calc is None:
            print("[Pipeline] Initialising MACE calculator...")
            self._calc = get_mace_calculator(
                model=self.cfg["mace_model"],
                device=self.cfg["device"],
                model_path=self.cfg.get("model_path"),
            )
        return self._calc

    @staticmethod
    def _banner(stage: int):
        name = MeltQuenchPipeline.STAGE_NAMES.get(stage, f"Stage {stage}")
        print("\n" + "=" * 62)
        print(f"  STAGE {stage}:  {name}")
        print("=" * 62)
