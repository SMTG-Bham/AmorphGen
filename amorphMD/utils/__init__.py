"""
amorphMD.utils
--------------
Shared utilities used across pipeline stages.

  get_mace_calculator  – build a MACE-MP calculator
  list_models          – print all available foundation models
  make_cubic           – reshape cell to cubic supercell
  build_md_dynamics    – create NVT or NPT ASE dynamics object
  resolve_ramp         – generate temperature ramp list
  MDLogger             – per-step MD logging to file + stdout
  TrajectoryWriter     – unified trajectory output (extxyz/xyz/traj/lammps)
  attach_outputs       – attach logger + traj writer to a dynamics object
  merge_config         – deep-merge two config dicts
"""

from .common import (
    get_mace_calculator,
    list_models,
    make_cubic,
    build_md_dynamics,
    resolve_ramp,
    MDLogger,
    TrajectoryWriter,
    TRAJ_FORMATS,
    attach_outputs,
    merge_config,
    extract_snapshots,
)

__all__ = [
    "get_mace_calculator",
    "list_models",
    "make_cubic",
    "build_md_dynamics",
    "resolve_ramp",
    "MDLogger",
    "TrajectoryWriter",
    "TRAJ_FORMATS",
    "attach_outputs",
    "merge_config",
    "extract_snapshots",
]
