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


# ─── Dimer detection ────────────────────────────────────────────────────────

class TestDimerReport:
    def _peroxide_structure(self):
        """SiO2-ish box with one deliberately planted O-O peroxide (1.45 A)."""
        from ase import Atoms
        import numpy as np
        pos = np.array([
            [0.0, 0.0, 0.0], [5.0, 5.0, 5.0],          # Si, far apart
            [2.5, 2.5, 2.5], [2.5, 2.5, 3.95],          # O-O at 1.45 A (dimer)
            [7.0, 7.0, 7.0], [1.0, 7.0, 1.0],           # isolated O
        ])
        return Atoms("Si2O4", positions=pos, cell=[10, 10, 10], pbc=True)

    def test_planted_peroxide_found(self):
        from amorphgen.analysis.structure import compute_dimers
        res = compute_dimers([self._peroxide_structure()])
        assert res["total"] == 1
        assert res["pairs"]["O-O"]["count"] == 1
        assert abs(res["pairs"]["O-O"]["min_distance"] - 1.45) < 0.01
        assert res["per_structure"] == [1]

    def test_clean_structure_dimer_free(self):
        from ase.build import bulk
        from amorphgen.analysis.structure import compute_dimers
        atoms = bulk("Cu", "fcc", a=3.6).repeat(2)   # normal metal, no dimers
        res = compute_dimers([atoms])
        assert res["total"] == 0

    def test_format_report_mentions_pair(self):
        from amorphgen.analysis.structure import compute_dimers, format_dimer_report
        text = format_dimer_report(compute_dimers([self._peroxide_structure()]))
        assert "O-O" in text and "1.45" in text
        clean = format_dimer_report(
            {"pairs": {}, "per_structure": [0], "total": 0,
             "n_structures": 1, "threshold_frac": 0.85})
        assert "DIMER-FREE" in clean

    def test_analyser_method(self):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser([self._peroxide_structure()])
        assert sa.dimer_report()["total"] == 1


class TestSqNormalisation:
    """Direct-method S(q) must satisfy the Faber-Ziman S(q->inf)=1 limit
    for weighted MULTI-element compositions (regression: the raw form
    plateaued at <f^2>/<f^2> ~ 1.8 for heavy/light element mixes)."""

    def _random_binary(self):
        """Random Na/Ta box — max f contrast (Z=11 vs 73)."""
        import numpy as np
        from ase import Atoms
        rng = np.random.default_rng(7)
        n = 60
        pos = rng.uniform(0, 12.0, (n, 3))
        return Atoms("Na30Ta30", positions=pos, cell=[12.0] * 3, pbc=True)

    def test_xray_high_q_plateau_is_one(self):
        import numpy as np
        from amorphgen.analysis.rdf import compute_structure_factor_direct
        res = compute_structure_factor_direct([self._random_binary()],
                                              qmax=14.0, nq=140,
                                              weighting="xray")
        q = np.array(res["q"]); s = np.array(res["s_q"])
        hi = s[(q > 9) & ~np.isnan(s)]
        # random positions ~ ideal gas: S(q) ~ 1 everywhere above q_min
        assert abs(hi.mean() - 1.0) < 0.15

    def test_unweighted_unchanged_by_offset(self):
        """For f=1 the self-scattering offset is exactly zero — the fix must
        not alter unweighted results."""
        import numpy as np
        from amorphgen.analysis.rdf import compute_structure_factor_direct
        res = compute_structure_factor_direct([self._random_binary()],
                                              qmax=14.0, nq=140,
                                              weighting="unweighted")
        q = np.array(res["q"]); s = np.array(res["s_q"])
        hi = s[(q > 9) & ~np.isnan(s)]
        assert abs(hi.mean() - 1.0) < 0.15

    def test_tiny_cell_image_dedup(self):
        """In a cell smaller than 2x the threshold, a close pair's periodic
        images must not double the dimer count (min-image dedup)."""
        from ase import Atoms
        from amorphgen.analysis.structure import compute_dimers
        # SiO2 context => O-O threshold 0.85*2.24 = 1.90 A. Cell L=3.3:
        # the O-O pair sits at 1.45 A directly AND at 3.3-1.45 = 1.85 A via
        # the periodic image — BOTH under threshold, so neighbor_list yields
        # two entries for the same (i, j) pair. Must count ONE dimer.
        atoms = Atoms("SiO2", positions=[[1.65, 1.65, 1.65],
                                         [0.0, 0.0, 0.0],
                                         [1.45, 0.0, 0.0]],
                      cell=[3.3, 3.3, 3.3], pbc=True)
        res = compute_dimers([atoms])
        assert res["pairs"]["O-O"]["count"] == 1
        assert abs(res["pairs"]["O-O"]["min_distance"] - 1.45) < 0.01
        assert res["total"] == 1

    def test_metal_self_pairs_skipped_when_anions_present(self):
        """Li-Li at ionic-matrix distances (2.2-2.4 A) must NOT be flagged —
        the metallic-radius threshold is the wrong yardstick for cations
        packed around shared anions (regression: relaxed a-Li3OCl produced
        13 Li-Li false positives)."""
        from ase import Atoms
        from amorphgen.analysis.structure import compute_dimers
        atoms = Atoms("Li2O", positions=[[0, 0, 0], [2.3, 0, 0],
                                         [1.15, 1.6, 0]],
                      cell=[8, 8, 8], pbc=True)   # Li-Li 2.3 A, physical
        assert compute_dimers([atoms])["total"] == 0

    def test_metal_self_pairs_checked_in_alloys(self):
        """Anion-free systems keep the metal-metal check (real dimers)."""
        from ase import Atoms
        from amorphgen.analysis.structure import compute_dimers
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [1.5, 0, 0]],
                      cell=[8, 8, 8], pbc=True)   # far below metallic contact
        assert compute_dimers([atoms])["total"] == 1

    def test_polyanion_bonds_not_flagged(self):
        """A real phosphate P-O bond (~1.5 A) must NOT be flagged as a dimer
        (P is a nonmetal, but P-O is the structure, not a defect). Regression:
        relaxed a-KTiOPO4 reported 23 false P-O 'dimers'."""
        from ase import Atoms
        from amorphgen.analysis.structure import compute_dimers
        # PO4-like: central P with 4 O at 1.53 A
        atoms = Atoms("PO4", positions=[[0, 0, 0], [1.53, 0, 0], [-1.53, 0, 0],
                                        [0, 1.53, 0], [0, -1.53, 0]],
                      cell=[10, 10, 10], pbc=True)
        assert compute_dimers([atoms])["total"] == 0

    def test_homonuclear_defect_still_flagged_with_polyanion(self):
        """A genuine S-S disulfide is still caught even in a P/S system."""
        from ase import Atoms
        from amorphgen.analysis.structure import compute_dimers
        atoms = Atoms("PS2", positions=[[0, 0, 0], [2.0, 0, 0], [2.0, 2.05, 0]],
                      cell=[10, 10, 10], pbc=True)   # S-S at 2.05 A (disulfide)
        res = compute_dimers([atoms])
        assert res["total"] == 1 and "S-S" in res["pairs"]
