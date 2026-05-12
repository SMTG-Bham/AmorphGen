"""
amorphgen.configs.yaml_config
------------------------------
Load pipeline configuration from a YAML file.

Example YAML::

    model: mace-mpa-0
    device: cuda

    opt:
      fmax: 0.01
      max_steps: 1000

    melt:
      ensemble: NPT
      T_start: 300
      T_end: 3000
"""

from __future__ import annotations

import yaml


# Valid top-level keys and their expected types.
_VALID_TOP_KEYS = {
    "model": str,
    "mace_model": (str, type(None)),
    "model_path": (str, type(None)),
    "device": str,
    "default_dtype": str,
    "traj_format": str,
    "opt": dict,
    "eq_premelt": dict,
    "melt": dict,
    "eq_high": dict,
    "quench": dict,
    "eq_low": dict,
    "random_gen": dict,
    "analysis": dict,
    "classical_params": dict,
    "convert": dict,
}

# Stage sub-keys and expected types.
_STAGE_SCHEMA = {
    "opt": {
        "fmax": (int, float),
        "max_steps": int,
        "optimizer": str,
        "cell_filter": str,
    },
    "eq_premelt": {
        "ensemble": str, "T": (int, float), "steps": int,
        "timestep": (int, float), "friction": (int, float),
        "ttime": (int, float),
    },
    "melt": {
        "ensemble": str, "T_start": (int, float), "T_end": (int, float),
        "T_step": (int, float), "steps_per_T": int, "rate": (int, float, type(None)),
        "timestep": (int, float), "friction": (int, float),
        "ttime": (int, float), "make_cubic": bool,
    },
    "eq_high": {
        "ensemble": str, "T": (int, float), "steps": int,
        "timestep": (int, float), "friction": (int, float),
        "ttime": (int, float), "make_cubic": bool,
    },
    "quench": {
        "ensemble": str, "T_start": (int, float), "T_end": (int, float),
        "T_step": (int, float), "steps_per_T": int, "rate": (int, float, type(None)),
        "timestep": (int, float), "friction": (int, float),
        "ttime": (int, float),
    },
    "eq_low": {
        "ensemble": str, "T": (int, float), "steps": int,
        "timestep": (int, float), "friction": (int, float),
        "ttime": (int, float),
    },
    "convert": {
        "input": str,
        "format": str,
        "output_dir": (str, type(None)),
    },
}

_VALID_ENSEMBLES = {"NVT", "NPT", "nvt", "npt"}
_VALID_DEVICES = {"cuda", "cpu", "mps", "auto"}


def _validate_config(cfg: dict, path: str) -> tuple[list[str], list[str]]:
    """Validate config dict. Returns (warnings, errors)."""
    warnings = []
    errors = []

    for key, val in cfg.items():
        if key not in _VALID_TOP_KEYS:
            warnings.append(f"Unknown top-level key '{key}' in {path}")
            continue

        expected = _VALID_TOP_KEYS[key]
        if not isinstance(val, expected if isinstance(expected, tuple) else (expected,)):
            errors.append(
                f"Key '{key}' has type {type(val).__name__}, "
                f"expected {expected}"
            )

    # Validate device
    if "device" in cfg and cfg["device"] not in _VALID_DEVICES:
        errors.append(
            f"Invalid device '{cfg['device']}'. "
            f"Choose from: {', '.join(sorted(_VALID_DEVICES))}"
        )

    # Validate stage sub-keys
    for stage_name, schema in _STAGE_SCHEMA.items():
        if stage_name not in cfg:
            continue
        stage = cfg[stage_name]
        if not isinstance(stage, dict):
            errors.append(f"'{stage_name}' must be a dict, got {type(stage).__name__}")
            continue
        for skey, sval in stage.items():
            if skey not in schema:
                warnings.append(f"Unknown key '{skey}' in {stage_name}")
                continue
            expected_type = schema[skey]
            if not isinstance(sval, expected_type if isinstance(expected_type, tuple)
                              else (expected_type,)):
                errors.append(
                    f"{stage_name}.{skey} has type {type(sval).__name__}, "
                    f"expected {expected_type}"
                )

        # Validate ensemble values
        if "ensemble" in stage and stage["ensemble"] not in _VALID_ENSEMBLES:
            errors.append(
                f"{stage_name}.ensemble = '{stage['ensemble']}' is invalid. "
                f"Use 'NVT' or 'NPT'."
            )

        # Validate positive numeric values
        for nkey in ("T", "T_start", "T_end", "steps", "steps_per_T",
                     "timestep", "fmax", "max_steps"):
            if nkey in stage and isinstance(stage[nkey], (int, float)):
                if stage[nkey] <= 0:
                    errors.append(
                        f"{stage_name}.{nkey} must be positive, got {stage[nkey]}"
                    )

    return warnings, errors


def load_yaml_config(path: str) -> dict:
    """
    Load a YAML configuration file and return it as a dict.

    Validates the config against expected keys and types. Prints
    warnings for unknown keys or type mismatches but does not raise
    (to allow forward-compatible configs with new keys).

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    dict
        Configuration dictionary (same structure as DEFAULT_CONFIG).

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the YAML file is empty or does not contain a mapping.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"YAML config file is empty: {path}")
    if not isinstance(cfg, dict):
        raise ValueError(
            f"YAML config must be a mapping (dict), got {type(cfg).__name__}: {path}"
        )

    warnings, errors = _validate_config(cfg, path)
    for w in warnings:
        print(f"  [Config warning] {w}")
    if errors:
        for e in errors:
            print(f"  [Config ERROR] {e}")
        raise ValueError(
            f"Invalid YAML config ({len(errors)} error(s) in {path}). "
            f"Fix the errors above and retry."
        )

    return cfg
