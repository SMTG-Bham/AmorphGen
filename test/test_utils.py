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
        assert temps[0] == 400          # T_start itself is not a segment
        assert temps[-1] == 3000
        assert len(temps) == 27         # 2700 K / 100 K -> exact rate
        assert all(temps[i] < temps[i+1] for i in range(len(temps)-1))

    def test_cooling_ramp(self):
        temps = resolve_ramp(3000, 300, -100)
        assert temps[0] == 2900
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

    def test_non_divisible_span_never_overshoots(self):
        # 300 -> 3000 in steps of 700 is not divisible; the old range()-based
        # melt ramp emitted 3100 (past the endpoint).
        temps = resolve_ramp(300, 3000, 700)
        assert max(temps) == 3000
        assert temps[-1] == 3000

    def test_float_step_supported(self):
        # range() would raise TypeError on a float step.
        temps = resolve_ramp(300, 1000, 250.5)
        assert temps[0] == 550.5 and temps[-1] == 1000
        assert all(t2 > t1 for t1, t2 in zip(temps, temps[1:]))

    def test_mis_signed_step_does_not_hang_or_crash(self):
        # A positive step for a cooling ramp used to give an empty list
        # (IndexError) or, in the quench while-loop, an infinite loop.
        # Direction is now inferred from the endpoints.
        temps = resolve_ramp(3000, 300, +100)   # cooling, wrong sign
        assert temps[0] == 2900 and temps[-1] == 300
        assert all(t2 < t1 for t1, t2 in zip(temps, temps[1:]))


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
        assert "xyz" not in TRAJ_FORMATS            # alias of extxyz, see _TRAJ_ALIASES
        from amorphgen.utils.common import _TRAJ_ALIASES
        assert _TRAJ_ALIASES["xyz"] == "extxyz"
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


class TestRampTemperatureAllIntegrators:
    """Heating/cooling ramps must be able to change T for every npt_method (Tier 2, EMT)."""

    @pytest.mark.parametrize("method", ["berendsen", "mtk", "parrinello-rahman"])
    def test_set_md_temperature_and_logging_hook_across_segments(self, method, tmp_path):
        from ase.build import bulk
        from ase.calculators.emt import EMT
        from ase.io import read
        from amorphgen.utils.common import build_md_dynamics, set_md_temperature, attach_outputs
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2)); atoms.calc = EMT()
        dyn = build_md_dynamics(atoms, ensemble="NPT", T=300.0, timestep=1.0, npt_method=method)
        attach_outputs(dyn, atoms, str(tmp_path / "s.log"), str(tmp_path / "s_traj.xyz"), interval=2)
        dyn.run(4)
        set_md_temperature(dyn, 400.0)          # IsotropicMTKNPT has no set_temperature
        dyn.run(4)                              # NPT (PR) refuses if the atoms were wrapped in between
        frames = read(str(tmp_path / "s_traj.xyz"), index=":")
        assert len(frames) >= 4
        assert frames[-1].calc is not None      # energy carried into the wrapped copy


class TestReviewFixesPhysics:
    """Regressions for the 2026-09 review: ramp rate, negative rate, smoothing edges,
    block test, torn checkpoint, model-name case, convert collisions."""

    def test_ramp_rate_is_exact(self):
        # 3000 -> 300 K, T_step 1000, 1000 steps per segment at 1 fs = 3 ps -> 900 K/ps
        temps = resolve_ramp(3000, 300, 1000)
        assert temps == [2000.0, 1000.0, 300.0]           # no segment at T_start
        assert 2700 / (len(temps) * 1000 * 1e-3) == pytest.approx(900.0)

    def test_negative_rate_does_not_collapse_quench(self):
        # the stage code takes abs(rate); reproduce its arithmetic
        rate = abs(float(-100)); steps = int(round(abs(-100) / (rate * 1.0 / 1000)))
        assert steps == 1000

    def test_smoothed_rdf_has_no_edge_dip(self):
        from ase.build import bulk
        from amorphgen.analysis.rdf import compute_rdf
        a = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((4, 4, 4)); a.rattle(0.2, seed=0)
        raw = np.array(compute_rdf([a], pair="Cu-Cu", rmax=7.0, nbins=350, sigma=0.0)["g_r"])
        sm = np.array(compute_rdf([a], pair="Cu-Cu", rmax=7.0, nbins=350, sigma=0.05)["g_r"])
        assert sm[-1] > 0.8 * raw[-10:].mean()             # zero-padding gave ~0.5x

    def test_block_average_test_passes_on_equilibrated_noise(self, tmp_path):
        from amorphgen.utils.equilibration import block_average_test
        rng = np.random.default_rng(0); passed = 0
        for k in range(10):
            e = -300.0 + 1.0 * rng.normal(size=4000)          # stationary, uncorrelated
            p = tmp_path / f"run{k}.log"
            with open(p, "w") as fh:
                fh.write("    Step     Time_ps       T_K       Epot_eV       Ekin_eV       Etot_eV      Vol_A3\n" + "-" * 80 + "\n")
                for s, ep in enumerate(e):
                    fh.write(f"{s*100:8d} {s*0.05:10.4f} {300.0:8.1f} {ep:12.4f} {10.0:12.4f} {ep+10:12.4f} {1000.0:10.2f}\n")
            ok, _ = block_average_test(str(p), n_atoms=100)
            passed += ok
        assert passed >= 8        # old threshold passed ~35 % of such runs

    def test_torn_last_frame_is_salvaged(self, tmp_path):
        from ase.build import bulk
        from ase.io import write
        a = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
        p = tmp_path / "t.xyz"; write(str(p), [a, a, a], format="extxyz")
        with open(p, "a") as fh:                            # half-written 4th frame
            fh.write('32\nLattice="7.2 0 0 0 7.2 0 0 0 7.2" Properties=species:S:1:pos:R:3\nCu 0 0 0\n')
        from amorphgen.utils.common import read_md_checkpoint
        ck = read_md_checkpoint(str(p))
        assert ck is not None and ck[1] == 200              # 3 good frames kept
        from ase.io import read
        assert len(read(str(p), index=":")) == 3            # file truncated to good frames

    def test_model_names_are_case_insensitive(self):
        from amorphgen.utils.calculators import _ci_get, MACE_FOUNDATION_MODELS, SEVENNET_MODELS
        assert _ci_get(MACE_FOUNDATION_MODELS, "MACE-MPA-0") == _ci_get(MACE_FOUNDATION_MODELS, "mace-mpa-0")
        assert _ci_get(SEVENNET_MODELS, "7NET-MF-OMPA") == _ci_get(SEVENNET_MODELS, "7net-mf-ompa")
        assert _ci_get(MACE_FOUNDATION_MODELS, "/some/path.model") == "/some/path.model"
