"""Tests for --resume support in random structure generation."""

import os
import pytest
import numpy as np
from ase.io import read, write

from amorphgen.pipeline.random_gen import batch_random


class TestRandomGenResume:

    def test_resume_skips_existing(self, tmp_path):
        """Generate 5, delete 2, resume should produce 5 total."""
        out = str(tmp_path / "structures")
        initial = os.path.join(out, "random_initial")   # v1.0.0rc2 subdir
        comp = {"Si": 8, "O": 16}

        # First run: generate 5
        paths1 = batch_random(comp, n_structures=5, output_dir=out, seed=42)
        assert len(paths1) == 5

        # Read structure 0 and 1 for later comparison
        atoms0_before = read(os.path.join(initial, "random_0000.xyz"))
        atoms2_before = read(os.path.join(initial, "random_0002.xyz"))

        # Delete structures 3 and 4
        os.remove(os.path.join(initial, "random_0003.xyz"))
        os.remove(os.path.join(initial, "random_0004.xyz"))

        # Resume: should regenerate 3 and 4, skip 0-2
        paths2 = batch_random(comp, n_structures=5, output_dir=out,
                              seed=42, resume=True)

        # Should have 5 paths total (3 existing + 2 new)
        all_files = sorted(f for f in os.listdir(initial)
                           if f.startswith("random_") and f.endswith(".xyz")
                           and "_opt" not in f)
        assert len(all_files) == 5

        # Existing structures should be unchanged
        atoms0_after = read(os.path.join(initial, "random_0000.xyz"))
        atoms2_after = read(os.path.join(initial, "random_0002.xyz"))
        assert np.allclose(atoms0_before.positions, atoms0_after.positions)
        assert np.allclose(atoms2_before.positions, atoms2_after.positions)

    def test_resume_regenerates_corrupted(self, tmp_path):
        """Corrupted (empty) file should be regenerated."""
        out = str(tmp_path / "structures")
        initial = os.path.join(out, "random_initial")   # v1.0.0rc2 subdir
        comp = {"Si": 8, "O": 16}

        # Generate 3
        batch_random(comp, n_structures=3, output_dir=out, seed=42)

        # Corrupt structure 1 (make it empty)
        corrupt_path = os.path.join(initial, "random_0001.xyz")
        with open(corrupt_path, "w") as f:
            f.write("")

        # Resume: should regenerate structure 1
        batch_random(comp, n_structures=3, output_dir=out,
                     seed=42, resume=True)

        # File should now be valid
        atoms = read(os.path.join(initial, "random_0001.xyz"))
        assert len(atoms) == 24  # Si8O16

    def test_resume_empty_dir(self, tmp_path):
        """Resume on empty directory should behave like fresh run."""
        out = str(tmp_path / "structures")
        comp = {"Si": 8, "O": 16}

        paths = batch_random(comp, n_structures=3, output_dir=out,
                             seed=42, resume=True)
        assert len(paths) == 3

    def test_resume_all_complete(self, tmp_path):
        """Resume when all structures exist should skip everything."""
        out = str(tmp_path / "structures")
        comp = {"Si": 8, "O": 16}

        # First run
        batch_random(comp, n_structures=3, output_dir=out, seed=42)

        # Resume: everything exists
        paths = batch_random(comp, n_structures=3, output_dir=out,
                             seed=42, resume=True)
        assert len(paths) == 3

    def test_resume_composition_mismatch_warns(self, tmp_path):
        """Changing composition on resume should warn."""
        out = str(tmp_path / "structures")

        # First run with SiO2
        batch_random({"Si": 8, "O": 16}, n_structures=2,
                     output_dir=out, seed=42)

        # Resume with different composition
        with pytest.warns(UserWarning, match="composition changed"):
            batch_random({"Al": 8, "O": 12}, n_structures=2,
                         output_dir=out, seed=42, resume=True)


class TestSeedReproducibility:
    """Per-structure seeds are derived from the structure index, so a resumed
    run reproduces exactly the structures a fresh run would generate."""

    def test_seed_helper_index_stable_and_retry_varying(self):
        from amorphgen.pipeline.random_gen import _derive_structure_seed
        # Deterministic for a fixed (base_seed, index, attempt)
        assert _derive_structure_seed(42, 3, 0) == _derive_structure_seed(42, 3, 0)
        # Distinct per index
        seeds = {_derive_structure_seed(42, i, 0) for i in range(6)}
        assert len(seeds) == 6
        # Retries of the same index draw different randomness
        assert _derive_structure_seed(42, 3, 0) != _derive_structure_seed(42, 3, 1)

    def test_resume_reproduces_fresh_structures(self, tmp_path):
        comp = {"Si": 8, "O": 16}
        fresh = str(tmp_path / "fresh")
        resumed = str(tmp_path / "resumed")

        batch_random(comp, n_structures=4, output_dir=fresh, seed=123)

        # Simulate an interruption: pre-seed the resume dir with the first
        # two structures from the fresh run, then resume.
        init = os.path.join(resumed, "random_initial")
        os.makedirs(init, exist_ok=True)
        for i in (0, 1):
            src = os.path.join(fresh, "random_initial", f"random_{i:04d}.xyz")
            write(os.path.join(init, f"random_{i:04d}.xyz"), read(src))
        batch_random(comp, n_structures=4, output_dir=resumed, seed=123,
                     resume=True)

        # Indices generated on resume (2, 3) must match the fresh run exactly.
        for i in (2, 3):
            a = read(os.path.join(fresh, "random_initial", f"random_{i:04d}.xyz"))
            b = read(os.path.join(resumed, "random_initial", f"random_{i:04d}.xyz"))
            assert a.get_chemical_symbols() == b.get_chemical_symbols()
            assert np.allclose(a.get_positions(), b.get_positions())


def test_batch_path_keeps_auto_cn_tolerance_and_cn_aware_minsep(tmp_path, monkeypatch):
    """batch_random must forward the automatic CN tolerance and build the same
    CN-aware minsep table as generate_random (regression for the CLI path)."""
    import amorphgen.pipeline.random_gen as rg
    from ase import Atoms
    captured = {}
    comp = {"Cu": 8, "Zr": 8}                        # alloy: auto tolerance is 2

    def fake_generate(composition, seed=None, **kwargs):
        captured.update(kwargs)
        n = sum(composition.values())
        return Atoms("Cu8Zr8", positions=np.random.default_rng(0).uniform(0, 8, (n, 3)),
                     cell=[8, 8, 8], pbc=True)
    monkeypatch.setattr(rg, "generate_random", fake_generate)
    rg.batch_random(comp, n_structures=1, output_dir=str(tmp_path), output_format="xyz")
    target_cn, auto_tol = rg._auto_target_cn(comp)
    assert captured.get("target_cn") == target_cn
    assert captured.get("cn_tolerance") == auto_tol and auto_tol > 0
