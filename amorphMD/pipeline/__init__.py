"""
amorphMD.pipeline
-----------------
Stage modules for the 7-stage melt-and-quench pipeline.

  opt_cell    – stages 1 (crystalline opt) and 7 (final opt via final_opt)
  equilibrate – stages 2 (pre-melt 300 K), 4 (high-T), 6 (low-T 300 K)
  melt_cell   – stage 3 (heat ramp)
  quench      – stage 5 (cooling ramp)
  final_opt   – stage 7 (amorphous optimisation)
  batch_quench – batch quench workflow from multiple snapshots
"""

from . import opt_cell, melt_cell, equilibrate, quench, final_opt, batch_quench
from .run_pipeline import MeltQuenchPipeline

__all__ = [
    "opt_cell",
    "melt_cell",
    "equilibrate",
    "quench",
    "final_opt",
    "batch_quench",
    "MeltQuenchPipeline",
]
