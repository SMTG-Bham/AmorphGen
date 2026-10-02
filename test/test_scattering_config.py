"""YAML controls for experiment comparisons and coherent XRD profiles."""

import pytest
import yaml

from amorphgen.configs.yaml_config import load_yaml_config


def _load(tmp_path, analysis):
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump({"analysis": analysis}))
    return load_yaml_config(path)["analysis"]


def test_scattering_yaml_keys_roundtrip(tmp_path):
    analysis = {
        "experiment_sq": "measured_sq.csv", "experiment_tr": "measured_tr.dat",
        "experiment_skiprows": 1, "experiment_columns": [2, 0, 3],
        "sq_fit_range": [1., 8.], "tr_fit_range": [1.2, 6.],
        "sq_qmax": 9., "sq_nq": 500, "sq_method": "direct",
        "sq_weighting": "neutron", "sq_smooth": 0.,
        "tr_qrange": [0.3, 20], "tr_window": "none",
        "xrd": True, "xrd_wavelength": .71, "xrd_qmax": 12., "xrd_nq": 400,
    }
    assert _load(tmp_path, analysis) == analysis


def test_optional_scattering_yaml_settings_accept_null(tmp_path):
    analysis = dict.fromkeys(("experiment_sq", "experiment_tr", "experiment_columns",
                              "sq_fit_range", "tr_fit_range", "xrd_qmax"))
    assert _load(tmp_path, analysis) == analysis
    assert _load(tmp_path, {"sq_fit_range": [2, 2]})["sq_fit_range"] == [2, 2]


@pytest.mark.parametrize("name,value", [
    ("experiment_sq", []), ("experiment_tr", False),
    ("experiment_skiprows", -1), ("experiment_skiprows", True),
    ("experiment_skiprows", 1.5),
    ("experiment_columns", "0 1"), ("experiment_columns", [0]),
    ("experiment_columns", [0, 1, 2, 3]), ("experiment_columns", [0, 0]),
    ("experiment_columns", [-1, 1]), ("experiment_columns", [False, 1]),
    ("experiment_columns", [0, 1.0]), ("experiment_columns", [0, []]),
    ("sq_fit_range", [2, 1]), ("sq_fit_range", [1]),
    ("sq_fit_range", [1, 2, 3]), ("sq_fit_range", [False, 2]),
    ("sq_fit_range", [1, float("nan")]),
    ("tr_fit_range", [1, float("inf")]), ("tr_fit_range", ["1", "2"]),
    ("tr_fit_range", "1 2"),
    ("sq_qmax", 0.1), ("sq_qmax", float("inf")), ("sq_qmax", False),
    ("sq_nq", 1), ("sq_nq", 2.0), ("sq_nq", True),
    ("sq_method", "typo"), ("sq_weighting", "typo"),
    ("sq_smooth", -.1), ("sq_smooth", float("nan")),
    ("tr_window", "typo"), ("tr_qrange", [2, 1]),
    ("tr_qrange", [-.1, 2]), ("tr_qrange", [1, float("inf")]),
    ("xrd", "true"), ("xrd_wavelength", 0),
    ("xrd_wavelength", float("nan")), ("xrd_wavelength", True),
    ("xrd_qmax", 0), ("xrd_qmax", float("inf")), ("xrd_qmax", True),
    ("xrd_nq", 1), ("xrd_nq", True), ("xrd_nq", 2.5),
])
def test_invalid_scattering_yaml_is_a_named_value_error(tmp_path, name, value):
    with pytest.raises(ValueError, match=f"analysis.{name}"):
        _load(tmp_path, {name: value})


def test_xrd_qrange_validation_defers_wavelength_until_cli_merge(tmp_path):
    # Cross-setting validation belongs to the runtime, since a CLI wavelength
    # can make either of these otherwise-inaccessible configurations valid.
    settings = {"xrd_wavelength": 4, "xrd_qmax": 4}
    assert _load(tmp_path, settings) == settings
    assert _load(tmp_path, {"xrd_qmax": 10}) == {"xrd_qmax": 10}
    with pytest.raises(ValueError, match="validity limit"):
        _load(tmp_path, {"xrd_wavelength": .1, "xrd_qmax": 80})


def test_cli_wavelength_override_can_make_yaml_qmax_accessible(tmp_path, monkeypatch):
    import sys
    from ase import Atoms
    from ase.io import write
    from amorphgen.analysis import StructureAnalyser
    from amorphgen.cli import main

    structure = tmp_path / "structure.xyz"
    write(structure, Atoms("Si2", positions=[[1, 1, 1], [3, 3, 3]],
                          cell=[8] * 3, pbc=True))
    config = tmp_path / "analysis.yaml"
    config.write_text(yaml.safe_dump({"analysis": {"xrd": True, "xrd_qmax": 10}}))
    options = {}

    def capture_pattern(self, **kwargs):
        options.update(kwargs)
        return {"wavelength": kwargs["wavelength"], "q": []}

    monkeypatch.setattr(StructureAnalyser, "xrd_pattern", capture_pattern)
    monkeypatch.setattr(sys, "argv", [
        "amorphgen", "--analyse", str(structure), "--config", str(config),
        "--cutoff", "2.5", "--xrd-wavelength", ".71",
    ])
    main()
    assert options["qmax"] == 10
    assert options["wavelength"] == .71
