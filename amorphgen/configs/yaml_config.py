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

import math
import re
import yaml

from ase.data import atomic_numbers


# Valid top-level keys and their expected types.
# keys AmorphGen sets internally; a user file must not smuggle one in, because
# the seed index it carries has already been banded (utils.common.scoped_run_index)
_INTERNAL_KEYS = frozenset({"seed_index"})

_VALID_TOP_KEYS = {
    "model": str,
    "mace_model": (str, type(None)),
    "seed": (int, type(None)),
    "run_index": (int, type(None)),
    "engine": str,
    "model_path": (str, type(None)),
    "device": str,
    "default_dtype": str,
    "traj_format": str,
    "opt": dict,
    "final_opt": dict,
    "eq_premelt": dict,
    "melt": dict,
    "eq_high": dict,
    "quench": dict,
    "eq_low": dict,
    "random_gen": dict,
    "analysis": dict,
    "classical_params": dict,
    "convert": dict,
    "safety": dict,
    "repulsive_core": dict,
}

# Stage sub-keys and expected types.
_STAGE_SCHEMA = {
    "opt": {
        "fmax": (int, float),
        "max_steps": int,
        "optimizer": str,
        "cell_filter": (str, type(None)),
        "logfile": str,
        "traj_file": str,
        "output_cif": str,
        "output_xyz": str,
        "output_format": str,
        "pressure_tol_gpa": (int, float),
        "batch_size": (int, str, type(None)),
    },
    "eq_premelt": {
        "ensemble": str, "npt_method": str,
        "T": (int, float), "steps": int,
        "timestep": (int, float), "friction": (int, float),
        "ttime": (int, float),
        "taup_factor": (int, float),
        "compressibility_GPa": (int, float),
    },
    "melt": {
        "ensemble": str, "npt_method": str,
        "T_start": (int, float), "T_end": (int, float),
        "T_step": (int, float), "steps_per_T": int, "rate": (int, float, type(None)),
        "timestep": (int, float), "friction": (int, float),
        "ttime": (int, float), "make_cubic": bool,
        "taup_factor": (int, float),
        "compressibility_GPa": (int, float),
    },
    "eq_high": {
        "ensemble": str, "npt_method": str,
        "T": (int, float), "steps": int,
        "timestep": (int, float), "friction": (int, float),
        "ttime": (int, float),
        "taup_factor": (int, float),
        "compressibility_GPa": (int, float),
    },
    "quench": {
        "ensemble": str, "npt_method": str,
        "T_start": (int, float), "T_end": (int, float),
        "T_step": (int, float), "steps_per_T": int, "rate": (int, float, type(None)),
        "timestep": (int, float), "friction": (int, float),
        "ttime": (int, float),
        "taup_factor": (int, float),
        "compressibility_GPa": (int, float),
    },
    "eq_low": {
        "ensemble": str, "npt_method": str,
        "T": (int, float), "steps": int,
        "timestep": (int, float), "friction": (int, float),
        "ttime": (int, float),
        "taup_factor": (int, float),
        "compressibility_GPa": (int, float),
    },
}

_STAGE_SCHEMA["final_opt"] = _STAGE_SCHEMA["opt"]
for _stage in ("eq_premelt", "melt", "eq_high", "quench", "eq_low"):
    _STAGE_SCHEMA[_stage].update({
        "log_file": str, "traj_file": str, "output_xyz": str,
    })
for _stage in ("eq_premelt", "eq_high", "eq_low"):
    _STAGE_SCHEMA[_stage]["make_cubic"] = bool

_NUMBER = (int, float)
_OPTIONAL_NUMBER = (int, float, type(None))
_OPTIONAL_DICT = (dict, type(None))
_OPTIONAL_STRING = (str, type(None))
_PAIR_PARAMS_SCHEMA = dict.fromkeys(("epsilon", "sigma", "A", "rho", "C"), _NUMBER)

# Only settings actually read from YAML belong here; CLI/API-only arguments
# must not silently pass validation and then be ignored.
_BLOCK_SCHEMA = {
    **_STAGE_SCHEMA,
    "random_gen": {
        "composition": dict,
        "n_structures": int,
        "target_density": _OPTIONAL_NUMBER,
        "density_scale": _NUMBER,
        "output_format": str,
        "minsep": _OPTIONAL_DICT,
        "target_cn": _OPTIONAL_DICT,
        "dmax": _OPTIONAL_DICT,
        "cn_tolerance": (int, type(None)),
        "dmax_factor": _NUMBER,
        "cell_filter": _OPTIONAL_STRING,
        "relax": bool,
        "seed": (int, type(None)),
    },
    "analysis": {
        "convergence": bool,
        "tolerances": dict,
        "convergence_confidence": _NUMBER,
        "convergence_max_structures": int,
        "cutoff": (str, int, float, dict),
        "cutoff_window": _NUMBER,
        "per_structure": bool,
        "check_dimers": bool,
        "total_cn": (str, list),
        "save_report": _OPTIONAL_STRING,
        "save_plot": _OPTIONAL_STRING,
        "rdf_pairs": (list, type(None)),
        "angle_triplets": (list, type(None)),
        "angle_style": str,
        "rmax": _OPTIONAL_NUMBER,
        "smearing": _NUMBER,
        "total_rdf": bool,
        "save_pdf": bool,
        "dpi": int,
        "show_title": bool,
        "pair_panels": bool,
        "sq": bool,
        "sq_weighting": str,
        "sq_method": str,
        "sq_smooth": _NUMBER,
        "sq_partials": bool,
        "sq_qmax": _NUMBER,
        "sq_nq": int,
        "tr": bool,
        "tr_qrange": list,
        "tr_window": str,
        "tr_scan": bool,
        "experiment_sq": _OPTIONAL_STRING,
        "experiment_tr": _OPTIONAL_STRING,
        "experiment_skiprows": int,
        "experiment_columns": (list, type(None)),
        "sq_fit_range": (list, type(None)),
        "tr_fit_range": (list, type(None)),
        "xrd": bool,
        "xrd_wavelength": _NUMBER,
        "xrd_qmax": _OPTIONAL_NUMBER,
        "xrd_nq": int,
        "rings": (bool, str, list, type(None)),
        "ring_bond_pair": (bool, str, list, type(None)),
        "connectivity": bool,
        "voronoi": (bool, str, type(None)),
        "voronoi_element": (bool, str, type(None)),
        "reference": _OPTIONAL_STRING,
        "energy_ranking": bool,
        "voids": bool,
        "bond_order": bool,
        "order_cutoff": (str, int, float, dict),
        "qbar6_threshold": _NUMBER,
        "order_min_neighbors": int,
        "void_samples": int,
        "void_probe_radius": _NUMBER,
        "void_bins": int,
        "void_seed": (int, type(None)),
        "void_radii": _OPTIONAL_DICT,
        "oxygen_speciation": bool,
        "network_formers": (str, list, type(None)),
        "elastic": bool,
        "elastic_strain": _NUMBER,
        "elastic_relax": bool,
        "vdos": bool,
        "vdos_displacement": _NUMBER,
        "vdos_sigma": _NUMBER,
        "vdos_npoints": int,
    },
    "classical_params": {
        "params": dict,
        "charges": _OPTIONAL_DICT,
        "cutoff": _NUMBER,
        "alpha": _OPTIONAL_NUMBER,
        "coulomb": bool,
        "coulomb_method": _OPTIONAL_STRING,
    },
    "convert": {
        "input": str,
        "format": str,
        "output_dir": _OPTIONAL_STRING,
    },
}

_VALID_ENSEMBLES = {"NVT", "NPT", "nvt", "npt"}
_VALID_NPT_METHODS = {"berendsen", "mtk", "parrinello-rahman"}
_VALID_DEVICES = {"cuda", "cpu", "mps", "auto"}


def _check_type(value, expected, key: str, errors: list[str]) -> bool:
    """Check YAML types without treating booleans as integers."""
    types = expected if isinstance(expected, tuple) else (expected,)
    if isinstance(value, types) and (not isinstance(value, bool) or bool in types):
        return True
    names = " or ".join(t.__name__ for t in types)
    errors.append(f"{key} has type {type(value).__name__}, expected {names}")
    return False


def _validate_keys(values: dict, schema: dict, prefix: str,
                   errors: list[str], path: str) -> None:
    for key, value in values.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if not prefix and key in _INTERNAL_KEYS:
            errors.append(
                f"'{key}' in {path} is set internally by AmorphGen and cannot be "
                f"given in a config file; use 'run_index' to label a run by hand")
        elif key not in schema:
            errors.append(f"Unknown key '{full_key}' in {path}")
        else:
            _check_type(value, schema[key], full_key, errors)


def _validate_data_map(value, kind: str, expected, prefix: str,
                       errors: list[str], path: str, *, cutoff=False) -> None:
    """Validate maps whose keys name elements/pairs rather than config fields."""
    if not isinstance(value, dict):
        return  # The containing schema checks dict/null types.
    for key, entry in value.items():
        full_key = f"{prefix}.{key}"
        if cutoff and key == "default":
            _check_type(entry, (str, int, float), full_key, errors)
            continue
        parts = key.split("-") if isinstance(key, str) else []
        size = 1 if kind == "element" else 2
        if len(parts) != size or any(part not in atomic_numbers for part in parts):
            errors.append(
                f"Unknown key '{full_key}' in {path}; expected an element symbol"
                if kind == "element" else
                f"Unknown key '{full_key}' in {path}; expected an Element-Element pair")
        if isinstance(expected, dict):
            if _check_type(entry, dict, full_key, errors):
                _validate_keys(entry, expected, full_key, errors, path)
        else:
            _check_type(entry, expected, full_key, errors)


def _validate_nested_values(cfg: dict, errors: list[str], path: str) -> None:
    # Element and pair names are open-ended data keys, but their values and
    # any settings inside them still obey a closed schema.
    maps = {
        "random_gen": {
            "composition": ("element", int),
            "target_cn": ("element", int),
            "minsep": ("pair", _NUMBER),
            "dmax": ("pair", _NUMBER),
        },
        "analysis": {
            "cutoff": ("pair", _NUMBER),
            "order_cutoff": ("pair", _NUMBER),
            "void_radii": ("element", _NUMBER),
        },
        "classical_params": {
            "charges": ("element", _NUMBER),
            "params": ("pair", _PAIR_PARAMS_SCHEMA),
        },
    }
    for block_name, fields in maps.items():
        block = cfg.get(block_name)
        if not isinstance(block, dict):
            continue
        for field, (kind, expected) in fields.items():
            _validate_data_map(
                block.get(field), kind, expected, f"{block_name}.{field}",
                errors, path, cutoff=(block_name == "analysis"
                                      and field in ("cutoff", "order_cutoff")))

    analysis = cfg.get("analysis")
    if not isinstance(analysis, dict):
        return
    tolerances = analysis.get("tolerances")
    if isinstance(tolerances, dict):
        for name, value in tolerances.items():
            key = f"analysis.tolerances.{name}"
            if not isinstance(name, str) or not name.strip():
                errors.append("analysis.tolerances keys must be nonempty descriptor names")
            if _check_type(value, _NUMBER, key, errors):
                if not math.isfinite(value) or value <= 0:
                    errors.append(f"{key} must be finite and positive")
    confidence = analysis.get("convergence_confidence")
    if type(confidence) in _NUMBER:
        if not math.isfinite(confidence) or not 0 < confidence < 1:
            errors.append("analysis.convergence_confidence must be finite and between 0 and 1")
    max_structures = analysis.get("convergence_max_structures")
    if type(max_structures) is int and max_structures < 2:
        errors.append("analysis.convergence_max_structures must be at least 2")
    cutoff_window = analysis.get("cutoff_window")
    if (type(cutoff_window) in _NUMBER
            and (not math.isfinite(cutoff_window) or cutoff_window <= 0)):
        errors.append("analysis.cutoff_window must be finite and positive")
    _validate_scattering_values(analysis, errors)
    for key in ("total_cn", "rdf_pairs", "angle_triplets", "tr_qrange",
                "rings", "ring_bond_pair", "network_formers"):
        value = analysis.get(key)
        if not isinstance(value, list):
            continue
        if key in ("tr_qrange", "rings", "ring_bond_pair") and len(value) != 2:
            errors.append(f"analysis.{key} must contain exactly two values")
        for index, entry in enumerate(value):
            _check_type(entry, _NUMBER if key == "tr_qrange" else str,
                        f"analysis.{key}[{index}]", errors)


def _validate_scattering_values(analysis: dict, errors: list[str]) -> None:
    """Validate comparison controls before a scattering calculation is started."""
    for name, allowed in (("sq_method", ("direct", "ft")),
                          ("sq_weighting", ("xray", "neutron", "unweighted")),
                          ("tr_window", ("lorch", "none"))):
        value = analysis.get(name)
        if isinstance(value, str) and value not in allowed:
            errors.append(f"analysis.{name} must be one of {', '.join(allowed)}")

    for name, minimum in (("experiment_skiprows", 0), ("sq_nq", 2), ("xrd_nq", 2)):
        value = analysis.get(name)
        if type(value) is int and value < minimum:
            errors.append(f"analysis.{name} must be at least {minimum}")
    for name, threshold, inclusive in (("sq_qmax", 0.1, False),
                                       ("sq_smooth", 0.0, True),
                                       ("xrd_wavelength", 0.0, False),
                                       ("xrd_qmax", 0.0, False)):
        value = analysis.get(name)
        if type(value) in _NUMBER and (
                not math.isfinite(value)
                or (value < threshold if inclusive else value <= threshold)):
            relation = "at least" if inclusive else "greater than"
            errors.append(f"analysis.{name} must be finite and {relation} {threshold}")

    qmax = analysis.get("xrd_qmax")
    if type(qmax) in _NUMBER and math.isfinite(qmax):
        if qmax > 24.0 * math.pi:
            errors.append("analysis.xrd_qmax exceeds the form-factor validity limit (24*pi)")
    # Physical accessibility depends on both qmax and wavelength, either of
    # which can be overridden on the CLI. compute_xrd_pattern validates that
    # relationship after the final values have been merged.

    for name in ("sq_fit_range", "tr_fit_range"):
        bounds = analysis.get(name)
        if not isinstance(bounds, list):
            continue
        valid = len(bounds) == 2
        if not valid:
            errors.append(f"analysis.{name} must contain exactly two values")
        for index, value in enumerate(bounds):
            if not _check_type(value, _NUMBER, f"analysis.{name}[{index}]", errors):
                valid = False
            elif not math.isfinite(value):
                errors.append(f"analysis.{name}[{index}] must be finite")
                valid = False
        if valid and bounds[0] > bounds[1]:
            errors.append(f"analysis.{name} must have lower <= upper")

    columns = analysis.get("experiment_columns")
    if isinstance(columns, list):
        if len(columns) not in (2, 3):
            errors.append("analysis.experiment_columns must select two or three columns")
        valid = True
        for index, value in enumerate(columns):
            if not _check_type(value, int, f"analysis.experiment_columns[{index}]", errors):
                valid = False
            elif value < 0:
                errors.append(f"analysis.experiment_columns[{index}] must be non-negative")
                valid = False
        if valid and len(set(columns)) != len(columns):
            errors.append("analysis.experiment_columns must select distinct columns")

    qrange = analysis.get("tr_qrange")
    if (isinstance(qrange, list) and len(qrange) == 2
            and all(type(value) in _NUMBER for value in qrange)):
        if (not all(math.isfinite(value) for value in qrange)
                or not 0 <= qrange[0] < qrange[1]):
            errors.append("analysis.tr_qrange must satisfy finite 0 <= qmin < qmax")


def _validate_config(cfg: dict, path: str) -> tuple[list[str], list[str]]:
    """Validate all YAML fields. Returns (warnings, errors) for compatibility."""
    errors = []
    _validate_keys(cfg, _VALID_TOP_KEYS, "", errors, path)
    for block_name, schema in _BLOCK_SCHEMA.items():
        block = cfg.get(block_name)
        if isinstance(block, dict):
            _validate_keys(block, schema, block_name, errors, path)
    _validate_nested_values(cfg, errors, path)

    # The same validation applies to YAML and direct Python API settings.
    from ..utils.safety import validate_safety_config
    from ..utils.repulsion import validate_repulsive_core_config
    for name, validator in (("safety", validate_safety_config),
                            ("repulsive_core", validate_repulsive_core_config)):
        if isinstance(cfg.get(name), dict):
            try:
                validator(cfg[name])
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))

    # Only check enum membership for strings, so malformed containers produce
    # validation errors rather than an unhashable-type exception.
    device = cfg.get("device")
    if isinstance(device, str) and device not in _VALID_DEVICES:
        errors.append(
            f"Invalid device '{device}'. "
            f"Choose from: {', '.join(sorted(_VALID_DEVICES))}")

    for stage_name in _STAGE_SCHEMA:
        stage = cfg.get(stage_name)
        if not isinstance(stage, dict):
            continue
        ensemble = stage.get("ensemble")
        if isinstance(ensemble, str) and ensemble not in _VALID_ENSEMBLES:
            errors.append(
                f"{stage_name}.ensemble = '{ensemble}' is invalid. Use 'NVT' or 'NPT'.")
        method = stage.get("npt_method")
        if isinstance(method, str) and method not in _VALID_NPT_METHODS:
            errors.append(
                f"{stage_name}.npt_method = '{method}' is invalid. "
                f"Choose from: {', '.join(sorted(_VALID_NPT_METHODS))}.")

        for key in ("T", "T_start", "T_end", "steps", "steps_per_T",
                    "timestep", "fmax", "max_steps",
                    "taup_factor", "compressibility_GPa"):
            value = stage.get(key)
            if type(value) in _NUMBER and value <= 0:
                errors.append(f"{stage_name}.{key} must be positive, got {value}")

    return [], errors


_SCI = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def _coerce_sci_notation(obj):
    """PyYAML 1.1 reads ``1e-3`` (no dot) as a string; turn such strings into floats."""
    if isinstance(obj, dict):
        return {k: _coerce_sci_notation(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_sci_notation(v) for v in obj]
    if isinstance(obj, str) and _SCI.match(obj.strip()):
        return float(obj)
    return obj


def load_yaml_config(path: str) -> dict:
    """
    Load a YAML configuration file and return it as a dict.

    Validates keys and types in every block, including element/pair maps.
    Unknown keys and invalid values raise ValueError before the config is used.

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
        If the YAML file is empty, is not a mapping, or fails validation.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

        cfg = _coerce_sci_notation(cfg)

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
            + "; ".join(errors)
        )

    return cfg
