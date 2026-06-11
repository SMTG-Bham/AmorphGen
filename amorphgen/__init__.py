"""
amorphgen: Automated melt-and-quench molecular dynamics pipeline
for amorphous structure generation using universal MLIPs.

Quick start::

    # CLI: generate 10 random In2O3 structures (80 atoms each)
    amorphgen --random-gen --composition "In2O3*16" --relax

    # Python: single structure
    from amorphgen import generate_random
    atoms = generate_random({"In": 16, "O": 24})

    # Python: melt-quench pipeline
    from amorphgen import MeltQuenchPipeline
    pipe = MeltQuenchPipeline("POSCAR")
    pipe.run()
"""

from .pipeline.run_pipeline import MeltQuenchPipeline
from .configs.default_config import DEFAULT_CONFIG
from .pipeline import opt_cell, melt_cell, equilibrate, quench, final_opt, batch_quench, random_gen
from .pipeline.random_gen import generate_random, batch_random
from .utils.convert import convert
from .utils.common import extract_snapshots

__version__ = "1.0.0rc2"
__author__ = "Chaiyawat Kaewmeechai"

__all__ = [
    "MeltQuenchPipeline",
    "DEFAULT_CONFIG",
    "generate_random",
    "batch_random",
    "convert",
    "extract_snapshots",
    "opt_cell",
    "melt_cell",
    "equilibrate",
    "quench",
    "final_opt",
    "batch_quench",
    "random_gen",
]
