"""
tests/test_utils.py
-------------------
Tier 1 unit tests for amorphgen.utils (no calculator needed).
"""

import os
import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk

from amorphgen.utils.common import (
    make_cubic, resolve_ramp, merge_config,
    MDLogger, TrajectoryWriter, TRAJ_FORMATS,
)
from amorphgen.utils.calculators import (
    _detect_backend, MACE_FOUNDATION_MODELS, MODEL_DESCRIPTIONS,
)
from amorphgen.configs import DEFAULT_CONFIG


# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════

class TestConfig:

    def test_default_config_has_required_keys(self):
        for key in ["model", "device", "opt", "melt", "eq_high", "quench", "eq_low"]:
            assert key in DEFAULT_CONFIG, f"Missing key: {key}"

    def test_default_model_is_mace(self):
        assert DEFAULT_CONFIG["model"] == "mace-mpa-0"

    def test_merge_config_basic(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        result = merge_config(base, {"b": {"c": 99}})
        assert result["a"] == 1
        assert result["b"]["c"] == 99
        assert result["b"]["d"] == 3  # preserved

    def test_merge_config_none_override(self):
        base = {"a": 1}
        result = merge_config(base, None)
        assert result == base

    def test_merge_config_does_not_mutate_original(self):
        base = {"a": {"b": 1}}
        result = merge_config(base, {"a": {"b": 2}})
        assert base["a"]["b"] == 1  # original unchanged
        assert result["a"]["b"] == 2


# ═════════════════════════════════════════════════════════════════════════════
# Temperature ramp
# ═════════════════════════════════════════════════════════════════════════════

class TestResolveRamp:

    def test_heating_ramp(self):
        temps = resolve_ramp(300, 3000, 100)
        assert temps[0] == 300
        assert temps[-1] == 3000
        assert all(temps[i] < temps[i+1] for i in range(len(temps)-1))

    def test_cooling_ramp(self):
        temps = resolve_ramp(3000, 300, -100)
        assert temps[0] == 3000
        assert temps[-1] == 300
        assert all(temps[i] > temps[i+1] for i in range(len(temps)-1))

    def test_single_step(self):
        temps = resolve_ramp(300, 300, 100)
        assert temps == [300]

    def test_zero_step_raises(self):
        with pytest.raises(ValueError):
            resolve_ramp(300, 3000, 0)

    def test_endpoint_always_included(self):
        temps = resolve_ramp(300, 2950, 100)
        assert 2950 in temps


# ═════════════════════════════════════════════════════════════════════════════
# Cell helpers
# ═════════════════════════════════════════════════════════════════════════════

class TestMakeCubic:

    def test_preserves_volume(self):
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
        vol_before = atoms.get_volume()
        atoms = make_cubic(atoms)
        vol_after = atoms.get_volume()
        assert abs(vol_before - vol_after) < 0.01

    def test_cell_is_cubic(self):
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
        atoms = make_cubic(atoms)
        L = atoms.cell[0, 0]
        assert abs(atoms.cell[1, 1] - L) < 1e-10
        assert abs(atoms.cell[2, 2] - L) < 1e-10
        # Off-diagonals should be zero
        assert abs(atoms.cell[0, 1]) < 1e-10
        assert abs(atoms.cell[0, 2]) < 1e-10

    def test_preserves_atom_count(self):
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
        n = len(atoms)
        atoms = make_cubic(atoms)
        assert len(atoms) == n


# ═════════════════════════════════════════════════════════════════════════════
# Trajectory writer
# ═════════════════════════════════════════════════════════════════════════════

class TestTrajectoryWriter:

    def test_supported_formats(self):
        assert "extxyz" in TRAJ_FORMATS
        assert "xyz" in TRAJ_FORMATS
        assert "traj" in TRAJ_FORMATS

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            TrajectoryWriter("test.xyz", fmt="invalid_format")


# ═════════════════════════════════════════════════════════════════════════════
# Backend detection
# ═════════════════════════════════════════════════════════════════════════════

class TestBackendDetection:

    def test_mace_models_detected(self):
        for name in ["mace-mpa-0", "mace-mp-0b3-medium", "mace-mh-1",
                      "mace-omat-0", "mace-matpes-pbe"]:
            assert _detect_backend(name) == "mace"

    def test_chgnet_detected(self):
        assert _detect_backend("chgnet") == "chgnet"

    def test_m3gnet_detected(self):
        assert _detect_backend("m3gnet") == "m3gnet"
        assert _detect_backend("matgl") == "m3gnet"
        assert _detect_backend("m3gnet-pes") == "m3gnet"

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unrecognised model"):
            _detect_backend("nonexistent-model-xyz")

    def test_case_insensitive(self):
        assert _detect_backend("MACE-MPA-0") == "mace"
        assert _detect_backend("CHGNet") == "chgnet"
        assert _detect_backend("M3GNet") == "m3gnet"


# ═════════════════════════════════════════════════════════════════════════════
# Model registry
# ═════════════════════════════════════════════════════════════════════════════

class TestModelRegistry:

    def test_mace_models_have_descriptions(self):
        for name in MACE_FOUNDATION_MODELS:
            if name in MODEL_DESCRIPTIONS:
                assert len(MODEL_DESCRIPTIONS[name]) > 10

    def test_descriptions_cover_all_backends(self):
        backends = {d.split()[0] for d in MODEL_DESCRIPTIONS.values()}
        assert "MACE-MPA-0" in backends or any("MACE" in b for b in backends)
        assert any("CHGNet" in b for b in backends)
        assert any("M3GNet" in b for b in backends)

    def test_default_model_in_registry(self):
        assert "mace-mpa-0" in MACE_FOUNDATION_MODELS
