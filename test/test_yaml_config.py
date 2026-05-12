"""
tests/test_yaml_config.py
--------------------------
Tests for YAML configuration loading and validation.
"""

import os
import pytest
import tempfile

from amorphgen.configs import load_yaml_config
from amorphgen.configs.yaml_config import _validate_config


class TestLoadYamlConfig:

    def test_basic_load(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("model: chgnet\ndevice: cpu\n")
        cfg = load_yaml_config(str(cfg_file))
        assert cfg["model"] == "chgnet"
        assert cfg["device"] == "cpu"

    def test_nested_config(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "model: mace-mpa-0\n"
            "opt:\n"
            "  fmax: 0.01\n"
            "  max_steps: 500\n"
        )
        cfg = load_yaml_config(str(cfg_file))
        assert cfg["opt"]["fmax"] == 0.01
        assert cfg["opt"]["max_steps"] == 500

    def test_random_gen_block(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "random_gen:\n"
            "  composition:\n"
            "    Si: 16\n"
            "    O: 32\n"
            "  n_structures: 5\n"
            "  target_density: 2.2\n"
            "  target_cn:\n"
            "    Si: 4\n"
            "    O: 2\n"
        )
        cfg = load_yaml_config(str(cfg_file))
        assert cfg["random_gen"]["composition"] == {"Si": 16, "O": 32}
        assert cfg["random_gen"]["target_cn"]["Si"] == 4

    def test_empty_file_raises(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_yaml_config(str(cfg_file))

    def test_non_dict_raises(self, tmp_path):
        cfg_file = tmp_path / "list.yaml"
        cfg_file.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="mapping"):
            load_yaml_config(str(cfg_file))

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_yaml_config("/nonexistent/config.yaml")

    def test_all_stage_keys(self, tmp_path):
        cfg_file = tmp_path / "full.yaml"
        cfg_file.write_text(
            "model: chgnet\n"
            "device: cpu\n"
            "opt:\n  fmax: 0.05\n"
            "eq_premelt:\n  T: 300\n"
            "melt:\n  T_end: 3000\n"
            "eq_high:\n  T: 3000\n"
            "quench:\n  T_start: 3000\n"
            "eq_low:\n  T: 300\n"
        )
        cfg = load_yaml_config(str(cfg_file))
        assert "opt" in cfg
        assert "eq_premelt" in cfg
        assert "melt" in cfg
        assert "eq_high" in cfg
        assert "quench" in cfg
        assert "eq_low" in cfg


class TestConfigValidation:
    """Tests for YAML config schema validation."""

    def test_valid_config_no_warnings(self):
        cfg = {
            "model": "mace-mpa-0",
            "device": "cuda",
            "opt": {"fmax": 0.01, "max_steps": 500},
        }
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert len(warnings) == 0
        assert len(errors) == 0

    def test_unknown_top_key_warns(self):
        cfg = {"model": "chgnet", "banana": 42}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("banana" in w for w in warnings)

    def test_wrong_type_top_key_errors(self):
        cfg = {"model": 123}  # should be str
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("model" in e and "int" in e for e in errors)

    def test_invalid_device_errors(self):
        cfg = {"device": "gpu"}  # should be cuda/cpu/mps/auto
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("device" in e and "gpu" in e for e in errors)

    def test_valid_devices(self):
        for dev in ("cuda", "cpu", "mps", "auto"):
            warnings, errors = _validate_config({"device": dev}, "test.yaml")
            assert not any("device" in e for e in errors)

    def test_invalid_ensemble_errors(self):
        cfg = {"melt": {"ensemble": "NVE"}}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("ensemble" in e and "NVE" in e for e in errors)

    def test_valid_ensembles(self):
        for ens in ("NVT", "NPT"):
            warnings, errors = _validate_config({"melt": {"ensemble": ens}}, "test.yaml")
            assert not any("ensemble" in e for e in errors)

    def test_negative_temperature_errors(self):
        cfg = {"eq_premelt": {"T": -100}}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("T" in e and "positive" in e for e in errors)

    def test_negative_fmax_errors(self):
        cfg = {"opt": {"fmax": -0.01}}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("fmax" in e and "positive" in e for e in errors)

    def test_wrong_type_in_stage_errors(self):
        cfg = {"opt": {"fmax": "not a number"}}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("fmax" in e and "str" in e for e in errors)

    def test_unknown_stage_key_warns(self):
        cfg = {"opt": {"banana": 42}}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("banana" in w for w in warnings)

    def test_stage_not_dict_errors(self):
        cfg = {"opt": "not a dict"}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("opt" in e and "dict" in e for e in errors)

    def test_classical_params_accepted(self):
        cfg = {"model": "buckingham", "classical_params": {"cutoff": 10.0}}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert not any("classical_params" in w for w in warnings)
        assert not any("classical_params" in e for e in errors)

    def test_negative_steps_errors(self):
        cfg = {"eq_premelt": {"steps": -100}}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("steps" in e and "positive" in e for e in errors)

    def test_full_valid_config(self):
        cfg = {
            "model": "mace-mpa-0",
            "device": "cuda",
            "default_dtype": "float64",
            "opt": {"fmax": 0.01, "max_steps": 1000, "optimizer": "LBFGS"},
            "eq_premelt": {"ensemble": "NVT", "T": 300, "steps": 50000,
                           "timestep": 1.0, "friction": 0.01},
            "melt": {"ensemble": "NPT", "T_start": 300, "T_end": 3000,
                     "T_step": 100, "steps_per_T": 1000, "timestep": 1.0},
            "quench": {"ensemble": "NVT", "T_start": 3000, "T_end": 300,
                       "T_step": -100, "steps_per_T": 1000},
        }
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert len(warnings) == 0
        assert len(errors) == 0

    def test_invalid_yaml_raises(self, tmp_path):
        """YAML with errors should raise ValueError."""
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("model: 123\ndevice: gpu\n")
        with pytest.raises(ValueError, match="error"):
            load_yaml_config(str(cfg_file))
