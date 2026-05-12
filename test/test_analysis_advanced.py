"""
tests/test_analysis_advanced.py
--------------------------------
Tests for analysis modules with low coverage: energy ranking, rings, voronoi.
Uses pre-built Cu structures with EMT energies.
"""

import os
import pytest
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write


class TestEnergyRanking:
    """Test energy ranking module."""

    def test_ranking_from_info(self):
        from amorphgen.analysis.energy import compute_energy_ranking

        atoms_list = []
        for e in [-3.5, -3.2, -3.8, -3.1]:
            atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
            atoms.info["energy"] = e * len(atoms)
            atoms_list.append(atoms)

        result = compute_energy_ranking(atoms_list)
        assert "error" not in result
        assert result["best"] == 2       # -3.8 is lowest
        assert result["worst"] == 3      # -3.1 is highest
        assert result["best_energy"] == pytest.approx(-3.8)
        assert result["worst_energy"] == pytest.approx(-3.1)
        assert result["spread"] == pytest.approx(0.7)
        assert len(result["ranking"]) == 4

    def test_ranking_from_calculator(self):
        from amorphgen.analysis.energy import compute_energy_ranking

        atoms_list = []
        for disp in [0.0, 0.1, 0.2]:
            atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
            atoms.positions[0] += [disp, 0, 0]
            atoms.calc = EMT()
            atoms_list.append(atoms)

        result = compute_energy_ranking(atoms_list)
        assert "error" not in result
        assert result["best"] == 0  # undistorted has lowest energy

    def test_ranking_no_energy(self):
        from amorphgen.analysis.energy import compute_energy_ranking

        atoms_list = [bulk("Cu", "fcc", a=3.6, cubic=True) for _ in range(3)]
        result = compute_energy_ranking(atoms_list)
        assert "warning" in result
        assert result["best"] is None
        assert result["ranking"] == []

    def test_ranking_mixed_energy_sources(self):
        from amorphgen.analysis.energy import compute_energy_ranking

        a1 = bulk("Cu", "fcc", a=3.6, cubic=True)
        a1.info["energy"] = -10.0
        a2 = bulk("Cu", "fcc", a=3.6, cubic=True)
        # a2 has no energy — should be skipped
        result = compute_energy_ranking([a1, a2])
        assert len(result["ranking"]) == 1


class TestRingStatistics:
    """Test ring statistics module."""

    def test_rings_on_sio2_like(self):
        """Build a simple network and check rings are found."""
        from amorphgen.analysis.rings import compute_ring_statistics

        # Create a simple SiO2-like structure where we know rings exist
        # Use a cristobalite-like arrangement
        from ase import Atoms
        # 2x2x2 NaCl structure as proxy (has clear ring topology)
        atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True) * (2, 2, 2)
        result = compute_ring_statistics(
            [atoms], bond_pair=("Na", "Cl"), cutoff=3.5, max_ring=8
        )
        assert "ring_sizes" in result
        assert "counts" in result
        assert "fractions" in result
        assert result["bond_pair"] == ("Na", "Cl")

    def test_rings_auto_bond_pair(self):
        from amorphgen.analysis.rings import compute_ring_statistics
        atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True) * (2, 2, 2)
        result = compute_ring_statistics([atoms], cutoff=3.5)
        assert result["bond_pair"] is not None

    def test_rings_single_element(self):
        """Single element — should still work (metallic bonding)."""
        from amorphgen.analysis.rings import compute_ring_statistics
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
        result = compute_ring_statistics(
            [atoms], bond_pair=("Cu", "Cu"), cutoff=2.7
        )
        assert "total_rings" in result


class TestVoronoi:
    """Test Voronoi tessellation module."""

    def test_voronoi_basic(self):
        from amorphgen.analysis.voronoi import compute_voronoi
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
        result = compute_voronoi([atoms])
        assert result["total_atoms"] > 0
        assert result["mean_faces"] > 0
        assert len(result["top_10"]) > 0

    def test_voronoi_with_element_filter(self):
        from amorphgen.analysis.voronoi import compute_voronoi
        atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True) * (2, 2, 2)
        result = compute_voronoi([atoms], element="Na")
        assert result["total_atoms"] > 0
        # Should only count Na atoms
        n_na = sum(1 for s in atoms.get_chemical_symbols() if s == "Na")
        assert result["total_atoms"] <= n_na

    def test_voronoi_distribution_sums(self):
        from amorphgen.analysis.voronoi import compute_voronoi
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
        result = compute_voronoi([atoms])
        total_from_dist = sum(result["distribution"].values())
        assert total_from_dist == result["total_atoms"]

    def test_voronoi_multiple_structures(self):
        from amorphgen.analysis.voronoi import compute_voronoi
        atoms_list = []
        for _ in range(3):
            a = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
            a.rattle(0.05)
            atoms_list.append(a)
        result = compute_voronoi(atoms_list)
        assert result["total_atoms"] > len(atoms_list[0])


class TestAnalyserIntegration:
    """Test StructureAnalyser methods that had low coverage."""

    @pytest.fixture
    def analyser_with_energies(self, tmp_path):
        """Create structures with EMT energies for analysis."""
        from amorphgen.analysis import StructureAnalyser
        struct_dir = tmp_path / "structs"
        struct_dir.mkdir()

        for i in range(5):
            atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
            atoms.rattle(0.02 * (i + 1), seed=i)
            atoms.calc = EMT()
            e = atoms.get_potential_energy()
            atoms.calc = None  # detach to avoid extxyz write conflict
            atoms.info["energy"] = e
            write(str(struct_dir / f"sample_{i}.xyz"), atoms, format="extxyz")

        return StructureAnalyser(str(struct_dir), cutoff=2.7)

    def test_energy_ranking(self, analyser_with_energies):
        result = analyser_with_energies.energy_ranking()
        assert "error" not in result
        assert result["spread"] >= 0

    def test_ring_statistics(self, analyser_with_energies):
        result = analyser_with_energies.ring_statistics(
            bond_pair=("Cu", "Cu"), cutoff=2.7
        )
        assert "ring_sizes" in result

    def test_voronoi(self, analyser_with_energies):
        result = analyser_with_energies.voronoi(element="Cu")
        assert result["total_atoms"] > 0

    def test_per_structure_summary(self, analyser_with_energies):
        text = analyser_with_energies.per_structure_summary()
        assert isinstance(text, str)
        assert len(text) > 0
