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
        comp = {"Si": 8, "O": 16}

        # First run: generate 5
        paths1 = batch_random(comp, n_structures=5, output_dir=out, seed=42)
        assert len(paths1) == 5

        # Read structure 0 and 1 for later comparison
        atoms0_before = read(os.path.join(out, "random_0000.xyz"))
        atoms2_before = read(os.path.join(out, "random_0002.xyz"))

        # Delete structures 3 and 4
        os.remove(os.path.join(out, "random_0003.xyz"))
        os.remove(os.path.join(out, "random_0004.xyz"))

        # Resume: should regenerate 3 and 4, skip 0-2
        paths2 = batch_random(comp, n_structures=5, output_dir=out,
                              seed=42, resume=True)

        # Should have 5 paths total (3 existing + 2 new)
        all_files = sorted(f for f in os.listdir(out)
                           if f.startswith("random_") and f.endswith(".xyz")
                           and "_opt" not in f)
        assert len(all_files) == 5

        # Existing structures should be unchanged
        atoms0_after = read(os.path.join(out, "random_0000.xyz"))
        atoms2_after = read(os.path.join(out, "random_0002.xyz"))
        assert np.allclose(atoms0_before.positions, atoms0_after.positions)
        assert np.allclose(atoms2_before.positions, atoms2_after.positions)

    def test_resume_regenerates_corrupted(self, tmp_path):
        """Corrupted (empty) file should be regenerated."""
        out = str(tmp_path / "structures")
        comp = {"Si": 8, "O": 16}

        # Generate 3
        batch_random(comp, n_structures=3, output_dir=out, seed=42)

        # Corrupt structure 1 (make it empty)
        corrupt_path = os.path.join(out, "random_0001.xyz")
        with open(corrupt_path, "w") as f:
            f.write("")

        # Resume: should regenerate structure 1
        batch_random(comp, n_structures=3, output_dir=out,
                     seed=42, resume=True)

        # File should now be valid
        atoms = read(os.path.join(out, "random_0001.xyz"))
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
