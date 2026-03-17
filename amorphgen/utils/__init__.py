"""
amorphgen.utils
---------------
Shared utilities used across pipeline stages.
"""

from .calculators import (
    get_calculator,
    get_mace_calculator,
    list_models,
    MACE_FOUNDATION_MODELS,
    CHGNET_MODELS,
    M3GNET_MODELS,
    MODEL_DESCRIPTIONS,
)

from .common import (
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
    "get_calculator",
    "get_mace_calculator",
    "list_models",
    "MACE_FOUNDATION_MODELS",
    "CHGNET_MODELS",
    "M3GNET_MODELS",
    "MODEL_DESCRIPTIONS",
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
