"""
amorphMD
========
7-stage melt-and-quench pipeline for generating amorphous oxide structures
using MACE-MP foundation models and ASE.

Pipeline stages
---------------
  1. opt_cell    – Crystalline structure optimisation   (LBFGS)
  2. equilibrate – Pre-melt equilibration  300 K        (NVT/NPT, 50 ps)
  3. melt_cell   – Heat ramp  300 → T_melt              (NPT, 100 K/ps)
  4. equilibrate – High-T equilibration at T_melt       (NVT, 100 ps)
  5. quench      – Cooling ramp  T_melt → 300 K         (NVT, 100 K/ps)
  6. equilibrate – Low-T equilibration  300 K           (NVT, 50 ps)
  7. final_opt   – Final optimisation → amorphous       (LBFGS)

Quick start
-----------
    from amorphMD import MeltQuenchPipeline

    pipe = MeltQuenchPipeline(
        input_file   = "In2O3_POSCAR",
        work_dir     = "InO_amorphous",
        cfg_override = {
            "mace_model": "mace-mpa-0-medium",
            "device":     "cuda",
            "melt":   {"T_end":         2500},
            "eq_high":{"temperature_K": 2500},
            "quench": {"T_start":       2500},
        }
    )
    atoms = pipe.run()               # all 7 stages
    atoms = pipe.run(stages=[1, 2])  # selected stages only

Individual stage access
-----------------------
    from amorphMD import opt_cell, melt_cell, equilibrate, quench, final_opt
    atoms = opt_cell.run("POSCAR")
    atoms = equilibrate.run(atoms, stage_key="eq_premelt")
"""

from .pipeline.run_pipeline import MeltQuenchPipeline
from .pipeline import (
    opt_cell,
    melt_cell,
    equilibrate,
    quench,
    final_opt,
    batch_quench,
)

__version__ = "1.2.0"
__author__  = "amorphMD contributors"

__all__ = [
    # Main pipeline class
    "MeltQuenchPipeline",
    # Individual stage modules (can be called directly)
    "opt_cell",       # Stage 1 & 7
    "melt_cell",      # Stage 3
    "equilibrate",    # Stage 2, 4, 6
    "quench",         # Stage 5
    "final_opt",      # Stage 7
    "batch_quench",   # Batch quench workflow
]
