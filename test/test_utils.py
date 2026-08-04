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
    resolve_device,
)


class TestResolveDevice:
    """resolve_device: 'auto' resolution and the torch-free fallback."""

    def test_explicit_device_passes_through(self):
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda") == "cuda"
        assert resolve_device("mps") == "mps"

    def test_auto_resolves_to_cuda_or_cpu(self):
        # With or without torch installed, auto must land on a concrete device.
        assert resolve_device("auto") in ("cuda", "cpu")

    def test_auto_without_torch_falls_back_to_cpu(self, monkeypatch):
        """The torch-free install contract: auto -> cpu, no ImportError."""
        import builtins
        real_import = builtins.__import__

        def no_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch not installed (simulated)")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_torch)
        assert resolve_device("auto") == "cpu"
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
# Snapshot extraction (with burn-in)
# ═════════════════════════════════════════════════════════════════════════════

class TestExtractSnapshots:
    """Verify burn_in_frames option of extract_snapshots."""

    @staticmethod
    def _write_dummy_traj(path, n_frames):
        """Write a fake extxyz trajectory of n_frames Cu fcc cells."""
        from ase.io import write
        atoms_list = [bulk("Cu", "fcc", a=3.6 + 0.001 * i, cubic=True) * (2, 2, 2)
                      for i in range(n_frames)]
        write(path, atoms_list, format="extxyz")

    def test_no_burn_in_default(self, tmp_path):
        from amorphgen.utils.common import extract_snapshots
        traj = str(tmp_path / "traj.xyz")
        self._write_dummy_traj(traj, n_frames=20)
        out = str(tmp_path / "snaps")
        paths = extract_snapshots(traj, n_snapshots=5, output_dir=out)
        assert len(paths) == 5
        # First snapshot starts at frame 0
        assert "frame00000" in paths[0]

    def test_burn_in_skips_leading_frames(self, tmp_path):
        from amorphgen.utils.common import extract_snapshots
        traj = str(tmp_path / "traj.xyz")
        self._write_dummy_traj(traj, n_frames=20)
        out = str(tmp_path / "snaps")
        paths = extract_snapshots(traj, n_snapshots=5,
                                  burn_in_frames=10, output_dir=out)
        assert len(paths) == 5
        # First sampled index is the burn-in cutoff (10), not 0
        assert "frame00010" in paths[0]
        # Last sampled index is the final trajectory frame (n_frames - 1)
        assert "frame00019" in paths[-1]

    def test_burn_in_equal_to_length_raises(self, tmp_path):
        from amorphgen.utils.common import extract_snapshots
        traj = str(tmp_path / "traj.xyz")
        self._write_dummy_traj(traj, n_frames=10)
        with pytest.raises(ValueError):
            extract_snapshots(traj, n_snapshots=2, burn_in_frames=10,
                              output_dir=str(tmp_path / "snaps"))


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

    def test_sevennet_detected(self):
        assert _detect_backend("sevennet") == "sevennet"
        assert _detect_backend("7net-mf-ompa") == "sevennet"
        assert _detect_backend("7net-l3i5") == "sevennet"
        assert _detect_backend("7net-omat") == "sevennet"

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unrecognised model"):
            _detect_backend("nonexistent-model-xyz")

    def test_case_insensitive(self):
        assert _detect_backend("MACE-MPA-0") == "mace"
        assert _detect_backend("CHGNet") == "chgnet"
        assert _detect_backend("SevenNet") == "sevennet"


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
        assert any("SevenNet" in b for b in backends)

    def test_default_model_in_registry(self):
        assert "mace-mpa-0" in MACE_FOUNDATION_MODELS


class TestResumeHelpers:
    """Shared frame-resume helpers (single home for the resume invariants)."""

    def test_ramp_resume_position(self):
        from amorphgen.utils.common import ramp_resume_position
        assert ramp_resume_position(0, 60, 4) == (0, 0)       # fresh
        assert ramp_resume_position(100, 60, 4) == (1, 40)    # mid-segment
        assert ramp_resume_position(120, 60, 4) == (2, 0)     # on boundary
        assert ramp_resume_position(240, 60, 4) == (4, 0)     # complete
        assert ramp_resume_position(999, 60, 4) == (4, 0)     # clamped

    def test_needs_velocity_init(self):
        import numpy as np
        from ase.build import bulk
        from amorphgen.utils.common import needs_velocity_init
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
        assert needs_velocity_init(atoms, 0)              # fresh run
        assert needs_velocity_init(atoms, 100)            # resumed, no momenta
        atoms.set_momenta(np.ones((len(atoms), 3)))
        assert not needs_velocity_init(atoms, 100)        # resumed w/ momenta

    def test_traj_log_interval_shared(self):
        """attach_outputs and read_md_checkpoint must share ONE interval."""
        import inspect
        from amorphgen.utils.common import (attach_outputs, read_md_checkpoint,
                                            TRAJ_LOG_INTERVAL)
        assert (inspect.signature(attach_outputs).parameters["interval"].default
                == TRAJ_LOG_INTERVAL)
        assert (inspect.signature(read_md_checkpoint).parameters["interval"].default
                == TRAJ_LOG_INTERVAL)
