"""Dependency-free defaults shared by configuration and runtime validators."""

DEFAULT_SAFETY_CONFIG = {
    "min_distance": 0.5,               # Angstrom, periodic contacts
    "max_energy_jump_per_atom": 10.0,  # eV/atom between checks
    "max_temperature": 100000.0,      # K
    "min_volume_ratio": 0.2,          # relative to stage start
    "max_volume_ratio": 5.0,
    "reference": None,               # optional second calculator
}

DEFAULT_REPULSIVE_CORE_CONFIG = {
    "enabled": False,
    "cutoff": 1.0,                    # Angstrom
    "strength": 1.0,                  # eV
}
