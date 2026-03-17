"""
amorphgen: Automated melt-and-quench molecular dynamics pipeline
for amorphous metal oxide structure generation using MACE-MP foundation models.
"""

from .pipeline.run_pipeline import MeltQuenchPipeline
from .configs.default_config import DEFAULT_CONFIG
from .pipeline import opt_cell, melt_cell, equilibrate, quench, final_opt, batch_quench, random_gen

__version__ = "2.0.0"
__author__ = "Chayanon Kaewmeechai"

__all__ = [
    "MeltQuenchPipeline",
    "DEFAULT_CONFIG",
    "opt_cell",
    "melt_cell",
    "equilibrate",
    "quench",
    "final_opt",
    "batch_quench",
    "random_gen",
]
