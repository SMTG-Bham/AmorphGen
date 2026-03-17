"""
tests/test_random_gen.py
------------------------
Tier 1 + 2 tests for amorphgen.pipeline.random_gen
All tests are CPU-only and require no MACE model.
"""

import os
import pytest
import numpy as np
from itertools import combinations

from amorphgen.pipeline.random_gen import (
    generate_random, batch_random,
    _default_minsep, _get_minsep, _estimate_cell_length,
)


def check_min_distances(atoms, minsep, tolerance=0.01):
    """Verify all pairwise minimum-image distances satisfy constraints."""
    symbols = atoms.get_chemical_symbols()
    L = atoms.cell[0, 0]
    violations = 0
    for i, j in combinations(range(len(atoms)), 2):
        d = atoms.positions[i] - atoms.positions[j]
        d -= L * np.round(d / L)
        dist = np.linalg.norm(d)
        req = _get_minsep(symbols[i], symbols[j], minsep)
        if dist < req - tolerance:
            violations += 1
    return violations


class TestDefaultMinsep:

    def test_returns_dict(self):
        result = _default_minsep(["In", "O"])
        assert isinstance(result, dict)
        assert len(result) == 3  # In-In, In-O, O-O

    def test_values_positive(self):
        result = _default_minsep(["Si", "O"])
        assert all(v > 0 for v in result.values())

    def test_scale_factor(self):
        r1 = _default_minsep(["Cu"], scale=1.0)
        r2 = _default_minsep(["Cu"], scale=0.5)
        key = list(r1.keys())[0]
        assert abs(r2[key] / r1[key] - 0.5) < 0.01


class TestEstimateCellLength:

    def test_returns_positive(self):
        L = _estimate_cell_length({"In": 32, "O": 48})
        assert L > 0

    def test_density_gives_smaller_cell(self):
        L_default = _estimate_cell_length({"In": 32, "O": 48})
        L_dense = _estimate_cell_length({"In": 32, "O": 48}, target_density=7.0)
        assert L_dense < L_default

    def test_more_atoms_bigger_cell(self):
        L_small = _estimate_cell_length({"O": 10})
        L_large = _estimate_cell_length({"O": 100})
        assert L_large > L_small


class TestGenerateRandom:

    def test_basic_generation(self):
        atoms = generate_random(
            composition={"Cu": 10},
            cell_length_ang=8.0,
            seed=42,
        )
        assert len(atoms) == 10
        assert atoms.get_chemical_formula() == "Cu10"

    def test_binary_composition(self):
        atoms = generate_random(
            composition={"In": 4, "O": 6},
            cell_length_ang=8.0,
            seed=42,
        )
        assert len(atoms) == 10
        symbols = atoms.get_chemical_symbols()
        assert symbols.count("In") == 4
        assert symbols.count("O") == 6

    def test_pbc_set(self):
        atoms = generate_random({"Si": 5}, cell_length_ang=7.0, seed=1)
        assert all(atoms.pbc)

    def test_reproducible_with_seed(self):
        a1 = generate_random({"O": 8}, cell_length_ang=6.0, seed=123)
        a2 = generate_random({"O": 8}, cell_length_ang=6.0, seed=123)
        assert np.allclose(a1.positions, a2.positions)

    def test_different_seeds_differ(self):
        a1 = generate_random({"O": 8}, cell_length_ang=6.0, seed=1)
        a2 = generate_random({"O": 8}, cell_length_ang=6.0, seed=2)
        assert not np.allclose(a1.positions, a2.positions)

    def test_minsep_respected(self):
        minsep = {"Cu-Cu": 2.0}
        atoms = generate_random(
            composition={"Cu": 15},
            cell_length_ang=10.0,
            minsep=minsep,
            seed=42,
        )
        violations = check_min_distances(atoms, minsep)
        assert violations == 0

    def test_custom_minsep_binary(self):
        minsep = {"In-In": 2.8, "In-O": 1.9, "O-O": 2.5}
        atoms = generate_random(
            composition={"In": 8, "O": 12},
            cell_length_ang=10.0,
            minsep=minsep,
            seed=42,
        )
        violations = check_min_distances(atoms, minsep)
        assert violations == 0

    def test_too_dense_raises(self):
        """Packing too many atoms in a small box should fail."""
        with pytest.raises(RuntimeError, match="Could not place"):
            generate_random(
                composition={"Cu": 100},
                cell_length_ang=3.0,
                seed=42,
                max_attempts_per_atom=100,
            )

    def test_cell_length_auto_from_density(self):
        atoms = generate_random(
            composition={"In": 32, "O": 48},
            target_density=5.5,
            seed=42,
        )
        assert len(atoms) == 80


class TestBatchRandom:

    def test_generates_correct_count(self, tmp_work_dir):
        paths = batch_random(
            composition={"Cu": 5},
            n_structures=3,
            output_dir=str(tmp_work_dir / "batch"),
            cell_length_ang=6.0,
        )
        assert len(paths) == 3
        for p in paths:
            assert os.path.isfile(p)

    def test_output_files_readable(self, tmp_work_dir):
        from ase.io import read
        paths = batch_random(
            composition={"Cu": 5},
            n_structures=2,
            output_dir=str(tmp_work_dir / "batch2"),
            cell_length_ang=6.0,
        )
        for p in paths:
            atoms = read(p)
            assert len(atoms) == 5
