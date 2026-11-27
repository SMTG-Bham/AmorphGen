"""
test/test_neighbor_cache.py
---------------------------
Skin neighbour lists (amorphgen.utils.neighbor_cache): the kept list must give
exactly the pairs of a fresh neighbour list at every step, through atomic
motion and cell changes, and must rebuild itself when it can no longer
guarantee that.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import neighbor_list

from amorphgen.utils.neighbor_cache import (
    NeighborCache, SkinNeighborList, _pairs_numpy, pair_kernel,
)

CUTOFF = 3.0


def random_cell(n=40, box=7.0, seed=0):
    rng = np.random.default_rng(seed)
    cell = np.array([[box, 0, 0], [0.4, box * 1.1, 0], [0.2, -0.3, box * 0.9]])
    return Atoms("Cu" * n, positions=rng.uniform(0, box, (n, 3)), cell=cell, pbc=True)


def as_set(i, j, r, D):
    return sorted(zip(i.tolist(), j.tolist(), [tuple(row) for row in np.round(D, 8)]))


def assert_same_pairs(got, atoms):
    i, j, r, D = neighbor_list("ijdD", atoms, CUTOFF)
    assert as_set(*got) == as_set(i, j, r, D)
    assert np.allclose(got[2], np.linalg.norm(got[3], axis=1))
    assert (np.diff(got[0]) >= 0).all()             # grouped by the first index


KERNELS = [False] + ([True] if pair_kernel()[1] == "numba" else [])


@pytest.mark.parametrize("use_numba", KERNELS)
class TestSkinList:

    def test_follows_a_trajectory(self, use_numba):
        atoms = random_cell()
        kept = SkinNeighborList(CUTOFF, skin=0.5, use_numba=use_numba)
        rng = np.random.default_rng(1)
        for _ in range(60):
            atoms.positions += rng.normal(0, 0.02, atoms.positions.shape)
            assert_same_pairs(kept.pairs(atoms), atoms)
        assert kept.builds > 1 and kept.reuses > 3 * kept.builds

    def test_follows_a_changing_cell(self, use_numba):
        """Cell relaxation / NPT: positions scale with the cell, plus motion."""
        atoms = random_cell(seed=2)
        kept = SkinNeighborList(CUTOFF, skin=0.5, use_numba=use_numba)
        rng = np.random.default_rng(3)
        for step in range(40):
            strain = np.eye(3) + rng.normal(0, 0.002, (3, 3))
            atoms.set_cell(atoms.cell.array @ strain, scale_atoms=True)
            atoms.positions += rng.normal(0, 0.01, atoms.positions.shape)
            assert_same_pairs(kept.pairs(atoms), atoms)
        assert kept.reuses > kept.builds                # the strain alone costs little

    def test_rebuilds_after_a_jump_or_wrap(self, use_numba):
        atoms = random_cell(seed=4)
        kept = SkinNeighborList(CUTOFF, skin=0.5, use_numba=use_numba)
        kept.pairs(atoms)
        atoms.positions[5] += [1.5, 0, 0]               # far more than the skin
        assert_same_pairs(kept.pairs(atoms), atoms)
        atoms.positions[7] += atoms.cell.array[0]       # wrapped by a lattice vector
        assert_same_pairs(kept.pairs(atoms), atoms)
        assert kept.builds == 3 and kept.reuses == 0

    def test_compressed_cell_rebuilds(self, use_numba):
        atoms = random_cell(seed=5)
        kept = SkinNeighborList(CUTOFF, skin=0.3, use_numba=use_numba)
        kept.pairs(atoms)
        atoms.set_cell(atoms.cell.array * 0.9, scale_atoms=True)    # pairs come inside
        assert_same_pairs(kept.pairs(atoms), atoms)
        assert kept.builds == 2


def test_numba_and_numpy_agree():
    if pair_kernel()[1] != "numba":
        pytest.skip("Numba not installed")
    atoms = random_cell(seed=6)
    i, j, S = neighbor_list("ijS", atoms, CUTOFF + 0.5)
    args = (np.ascontiguousarray(atoms.positions), np.ascontiguousarray(atoms.cell.array),
            i.astype(np.int32), j.astype(np.int32), S.astype(float), CUTOFF)
    fast, plain = pair_kernel(True)[0](*args), _pairs_numpy(*args)
    for a, b in zip(fast, plain):
        assert np.allclose(a, b, atol=1e-13) and len(a) == len(b)


class TestCache:

    def test_one_list_per_key_and_statistics(self):
        cache = NeighborCache(CUTOFF, skin=0.5)
        first, second = random_cell(seed=7), random_cell(n=30, seed=8)
        for _ in range(5):
            first.rattle(0.01)
            second.rattle(0.01)
            assert_same_pairs(cache.pairs("a", first), first)
            assert_same_pairs(cache.pairs("b", second), second)
        assert cache.builds == 2 and cache.reuses == 8
        assert "2 built, 8 reused (80 %" in cache.summary()

    def test_a_key_that_changes_structure_is_still_right(self):
        """Keys need not be true identities: a mismatch costs one rebuild."""
        cache = NeighborCache(CUTOFF, skin=0.5)
        first, other = random_cell(seed=9), random_cell(seed=10)
        cache.pairs("slot", first)
        assert_same_pairs(cache.pairs("slot", other), other)       # same size, other atoms
        smaller = random_cell(n=25, seed=11)
        assert_same_pairs(cache.pairs("slot", smaller), smaller)   # other size
        assert cache.builds == 3

    def test_no_key_or_no_skin_builds_fresh(self):
        atoms = random_cell(seed=12)
        for cache, key in ((NeighborCache(CUTOFF, skin=0.5), None),
                           (NeighborCache(CUTOFF, skin=0.0), "a")):
            for _ in range(3):
                assert_same_pairs(cache.pairs(key, atoms), atoms)
            assert cache.builds == 3 and cache.reuses == 0

    def test_not_fully_periodic_is_not_kept(self):
        atoms = random_cell(seed=13)
        atoms.pbc = [True, True, False]
        cache = NeighborCache(CUTOFF, skin=0.5)
        for _ in range(2):
            assert_same_pairs(cache.pairs("a", atoms), atoms)
        assert cache.reuses == 0

    def test_bounded(self):
        cache = NeighborCache(CUTOFF, skin=0.5, max_lists=3)
        atoms = random_cell(n=10, seed=14)
        for key in range(8):
            cache.pairs(key, atoms)
        assert len(cache._lists) == 3 and cache.builds == 8
