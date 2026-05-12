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
    SEVENNET_MODELS,
    MODEL_DESCRIPTIONS,
)

from .equilibration import convergence_report

from .classical import LennardJonesCalculator, BuckinghamCalculator

from .radii import (
    # Data tables
    SHANNON_IONIC_RADII,
    METALLIC_RADII,
    PAULING_EN,
    NONMETALS,
    METALLOIDS,
    ELEMENTAL_DENSITIES,
    SCALE_FACTORS,
    # Functions
    classify_bond,
    get_ionic_radius,
    get_metallic_radius,
    get_effective_radius,
    default_minsep,
    estimate_density,
    estimate_cell_length,
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
    compute_density_gcm3,
)

from .convert import convert

__all__ = [
    # Classical calculators
    "LennardJonesCalculator",
    "BuckinghamCalculator",
    # Radii and bonding
    "SHANNON_IONIC_RADII",
    "METALLIC_RADII",
    "PAULING_EN",
    "NONMETALS",
    "METALLOIDS",
    "ELEMENTAL_DENSITIES",
    "SCALE_FACTORS",
    "classify_bond",
    "get_ionic_radius",
    "get_metallic_radius",
    "get_effective_radius",
    "default_minsep",
    "estimate_density",
    "estimate_cell_length",
    # Rest
    "convergence_report",
    "get_calculator",
    "get_mace_calculator",
    "list_models",
    "MACE_FOUNDATION_MODELS",
    "CHGNET_MODELS",
    "SEVENNET_MODELS",
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
    "compute_density_gcm3",
    "convert",
]
