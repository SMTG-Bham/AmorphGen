"""
tests/test_analysis.py
-----------------------
Tier 1 tests for amorphgen.utils.analysis (StructureAnalyser).
"""

import os
import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk


class TestStructureAnalyser:

    @pytest.fixture
    def sio2_atoms(self):
        """Create a simple SiO2-like structure for testing."""
        # 3 Si + 6 O in a 6 Å cubic cell
        positions = [
            [1.0, 1.0, 1.0],  # Si
            [3.0, 3.0, 1.0],  # Si
            [1.0, 3.0, 3.0],  # Si
            [1.8, 1.0, 1.8],  # O
            [1.0, 1.8, 1.0],  # O
            [3.8, 3.0, 1.0],  # O
            [3.0, 3.8, 1.0],  # O
            [1.0, 3.0, 3.8],  # O
            [1.8, 3.0, 3.0],  # O
        ]
        atoms = Atoms(
            symbols=["Si", "Si", "Si", "O", "O", "O", "O", "O", "O"],
            positions=positions,
            cell=[6, 6, 6],
            pbc=True,
        )
        return atoms

    @pytest.fixture
    def sio2_dir(self, tmp_path, sio2_atoms):
        """Write SiO2 structures to a temp directory."""
        from ase.io import write
        d = tmp_path / "sio2"
        d.mkdir()
        for i in range(3):
            atoms = sio2_atoms.copy()
            # Add small random noise
            atoms.positions += np.random.default_rng(i).normal(0, 0.05, atoms.positions.shape)
            write(str(d / f"struct_{i:04d}.xyz"), atoms, format="extxyz")
        return str(d)

    def test_load_directory(self, sio2_dir):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(sio2_dir)
        assert len(sa.atoms_list) == 3

    def test_load_single_file(self, sio2_dir):
        from amorphgen.analysis import StructureAnalyser
        files = sorted(os.listdir(sio2_dir))
        sa = StructureAnalyser(os.path.join(sio2_dir, files[0]))
        assert len(sa.atoms_list) == 1

    def test_load_list(self, sio2_dir):
        from amorphgen.analysis import StructureAnalyser
        files = [os.path.join(sio2_dir, f) for f in sorted(os.listdir(sio2_dir))]
        sa = StructureAnalyser(files)
        assert len(sa.atoms_list) == 3

    def test_density(self, sio2_dir):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(sio2_dir)
        d = sa.density()
        assert d["mean"] > 0
        assert d["std"] >= 0
        assert len(d["values"]) == 3

    def test_coordination(self, sio2_dir):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(sio2_dir, cutoff=2.5)
        cn = sa.coordination()
        assert isinstance(cn, dict)
        for pair, data in cn.items():
            assert "mean" in data
            assert "distribution" in data
            assert "total_atoms" in data

    def test_bond_distances(self, sio2_dir):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(sio2_dir, cutoff=2.5)
        bd = sa.bond_distances()
        assert isinstance(bd, dict)
        for pair, data in bd.items():
            assert data["mean"] > 0
            assert data["count"] > 0

    def test_bond_angles(self, sio2_dir):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(sio2_dir, cutoff=2.5)
        ba = sa.bond_angles()
        assert isinstance(ba, dict)
        for triplet, data in ba.items():
            assert 0 < data["mean"] < 180
            assert data["count"] > 0

    def test_rdf_auto_rmax(self, sio2_dir):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(sio2_dir, cutoff=2.5)
        rdf = sa.rdf()
        r = np.array(rdf["r"])
        g = np.array(rdf["g_r"])
        # Auto rmax should be <= half cell (3.0 Å)
        assert r[-1] <= 3.0
        assert len(r) == len(g)

    def test_rdf_manual_rmax(self, sio2_dir):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(sio2_dir, cutoff=2.5)
        rdf = sa.rdf(rmax=2.5)
        r = np.array(rdf["r"])
        assert r[-1] <= 2.5

    def test_rdf_partial(self, sio2_dir):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(sio2_dir, cutoff=2.5)
        rdf = sa.rdf(pair="O-Si")
        assert len(rdf["r"]) > 0
        assert len(rdf["g_r"]) > 0

    def test_summary_returns_string(self, sio2_dir):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(sio2_dir, cutoff=2.5)
        text = sa.summary()
        assert isinstance(text, str)
        assert "Density" in text

    def test_save_report(self, sio2_dir, tmp_path):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(sio2_dir, cutoff=2.5)
        report_path = str(tmp_path / "report.txt")
        sa.save_report(report_path)
        assert os.path.isfile(report_path)
        with open(report_path) as f:
            assert "Density" in f.read()

    def test_plot_creates_files(self, sio2_dir, tmp_path):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(sio2_dir, cutoff=2.5)
        plot_dir = str(tmp_path / "plots")
        sa.plot(output_dir=plot_dir, prefix="test", save_csv=True)
        assert os.path.isfile(os.path.join(plot_dir, "test_rdf.png"))
        assert os.path.isfile(os.path.join(plot_dir, "test_rdf.csv"))

    def test_cutoff_auto(self, sio2_dir):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(sio2_dir, cutoff="auto")
        assert isinstance(sa.cutoff, dict)
        assert len(sa.cutoff) > 0

    def test_cutoff_auto_rdf(self, sio2_dir):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(sio2_dir, cutoff="auto-rdf")
        assert isinstance(sa.cutoff, dict)
        assert len(sa.cutoff) > 0

    def test_empty_dir_raises(self, tmp_path):
        from amorphgen.analysis import StructureAnalyser
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            StructureAnalyser(str(empty))


class TestRDFNormalisation:

    def test_single_element_total_equals_partial(self):
        """For a single element, total RDF should equal the partial."""
        from amorphgen.analysis import StructureAnalyser
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
        sa = StructureAnalyser([atoms], cutoff=3.0)
        total = sa.rdf(rmax=3.0)
        partial = sa.rdf(pair="Cu-Cu", rmax=3.0)
        g_total = np.array(total["g_r"])
        g_partial = np.array(partial["g_r"])
        # Should be identical for single element
        np.testing.assert_allclose(g_total, g_partial, atol=0.01)

    def test_rdf_converges_to_one(self):
        """g(r) should converge to ~1 at large r for a bulk structure."""
        from amorphgen.analysis import StructureAnalyser
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (3, 3, 3)
        sa = StructureAnalyser([atoms], cutoff=3.0)
        rdf = sa.rdf(rmax=5.0)
        r = np.array(rdf["r"])
        g = np.array(rdf["g_r"])
        mask = (r > 4.0) & (r < 5.0)
        if np.any(mask):
            assert abs(np.mean(g[mask]) - 1.0) < 0.3
