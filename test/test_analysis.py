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

    def test_default_smearing_and_sq_isolation(self, sio2_dir):
        """Default RDF smearing is DEFAULT_SMEARING (0.05 A): peak position
        unchanged, height reduced. S(q) must be unaffected by the default
        (it always transforms the raw g(r))."""
        from amorphgen.analysis import StructureAnalyser
        from amorphgen.analysis.rdf import compute_rdf, DEFAULT_SMEARING
        assert DEFAULT_SMEARING == 0.05
        sa = StructureAnalyser(sio2_dir)
        raw = sa.rdf(pair="Si-O", rmax=3.0, nbins=300, sigma=0.0)
        dflt = sa.rdf(pair="Si-O", rmax=3.0, nbins=300)                 # default
        expl = compute_rdf(sa.atoms_list, pair="Si-O", rmax=3.0, nbins=300,
                           sigma=DEFAULT_SMEARING)
        g_raw, g_dflt, g_expl = (np.array(x["g_r"]) for x in (raw, dflt, expl))
        assert np.allclose(g_dflt, g_expl)            # default == 0.05
        assert g_dflt.max() < g_raw.max()             # smearing lowers the peak
        # ...and moves its position by at most ~sigma (the sparse 9-atom
        # fixture has isolated spikes that merge under the kernel; dr=0.01 A)
        assert abs(np.argmax(g_dflt) - np.argmax(g_raw)) <= 5
        # S(q) isolation from the smeared default is covered by
        # test_ft_sq_uses_r_squared_transform, whose reference uses sigma=0.0.

    def test_save_report_creates_parent_dir(self, sio2_dir, tmp_path):
        """--save-report into a not-yet-existing folder must not abort the
        run (regression: FileNotFoundError; --save-plot already created its
        directory, save_report did not)."""
        import os
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(sio2_dir)
        target = tmp_path / "new" / "nested" / "report.txt"
        sa.save_report(str(target), text="hello")
        assert os.path.isfile(target) and target.read_text() == "hello"

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

    def test_partial_cutoff_dict_is_completed_from_auto_rdf(self, sio2_dir):
        """A dict naming only Si-O keeps auto-rdf for Si-Si and O-O instead of
        treating them as unbonded; 'default' sets the base rule."""
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(sio2_dir, cutoff={"Si-O": 1.9})
        assert sa._get_cutoff("Si", "O") == 1.9 == sa._get_cutoff("O", "Si")
        assert sa._get_cutoff("O", "O") > 0 and sa._get_cutoff("Si", "Si") > 0
        assert "overrides Si-O=1.90" in sa._cutoff_mode
        sa2 = StructureAnalyser(sio2_dir, cutoff={"default": 3.0, "Si-O": 1.9})
        assert sa2._get_cutoff("O", "O") == 3.0 and sa2._get_cutoff("Si", "O") == 1.9
        tot = sa.total_coordination()
        assert set(tot) == {"O", "Si"} and tot["Si"]["mean"] > 0
        only_o = sa.total_coordination(centre="O", partners=["Si"])
        assert set(only_o) == {"O"} and only_o["O"]["mean"] == tot["O"]["mean"]


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

    def test_ft_sq_uses_r_squared_transform(self):
        """FT S(q) must use the 3D isotropic integrand r^2 (g-1) sinc(qr).
        Regression for the missing factor of r (integrand was r (g-1) sinc)."""
        import numpy as np
        from amorphgen.analysis.rdf import (compute_structure_factor,
                                            compute_rdf)
        atoms = self._random_binary()
        out = compute_structure_factor([atoms], weighting="unweighted",
                                       qmax=10.0, nq=60)
        q = np.array(out["q"]); s = np.array(out["s_q"])

        # Independent reference with the correct r^2 integrand.
        rdf = compute_rdf([atoms], nbins=500, sigma=0.0)   # S(q) uses raw g(r)
        r = np.array(rdf["r"]); g = np.array(rdf["g_r"]); dr = r[1] - r[0]
        rho = (len(atoms) - 1) / atoms.get_volume()
        trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        ref = np.ones_like(q)
        for i, qi in enumerate(q):
            qr = qi * r
            sinc = np.where(qr > 1e-12, np.sin(qr) / qr, 1.0)
            ref[i] = 1.0 + 4.0 * np.pi * rho * trapz(r ** 2 * (g - 1.0) * sinc,
                                                     dx=dr)
        assert np.allclose(s, ref, atol=1e-6)

    def test_ft_partials_use_total_density_fz_identity(self, monkeypatch):
        """With equal scattering lengths the Faber-Ziman weighted total must
        equal the unweighted total. This holds only if the FT partials use
        the TOTAL number density rho0; using the partner density n_b/V scales
        each partial by c_b and gives exactly 0.5 for a 50/50 binary.
        Regression for that bug. Uses a real (rattled) crystal so g-1 != 0
        and checks the strong-signal ratio, which tolerates the O(1/N)
        self-exclusion finite-size effect but cleanly rejects 0.5."""
        import numpy as np
        from ase.build import bulk
        import amorphgen.analysis.rdf as rdf_mod

        monkeypatch.setitem(rdf_mod._NEUTRON_B, "Na", 5.0)
        monkeypatch.setitem(rdf_mod._NEUTRON_B, "Cl", 5.0)
        atoms = bulk("NaCl", "rocksalt", a=5.64).repeat((3, 3, 3))
        atoms.rattle(0.15, seed=0)

        u = np.array(rdf_mod.compute_structure_factor(
            [atoms], weighting="unweighted", qmax=10.0, nq=120)["s_q"])
        w = np.array(rdf_mod.compute_structure_factor(
            [atoms], weighting="neutron", qmax=10.0, nq=120)["s_q"])
        strong = np.abs(u - 1.0) > 0.3
        ratio = (w - 1.0)[strong] / (u - 1.0)[strong]
        assert abs(ratio.mean() - 1.0) < 0.1   # old partner-density code: 0.5

    def test_scattering_tables_match_printed_sources(self):
        """Pin table entries to the printed primary sources so they cannot
        drift. Neutron b_c: Sears, Neutron News 3(3), 26 (1992), Table 1
        (all 55 entries were checked; Cd, W, Au had been wrong).
        X-ray f0: Waasmaier & Kirfel (1995) Table 1(a), rows read
        digit-for-digit from the printed table."""
        from amorphgen.analysis.rdf import _NEUTRON_B, _WK95_XRAY
        sears = {"H": -3.7390, "O": 5.803, "Si": 4.1491, "Cl": 9.5770,
                 "Ti": -3.438, "Ga": 7.288, "Cd": 4.87, "In": 4.065,
                 "Hf": 7.77, "W": 4.86, "Au": 7.63, "Pb": 9.405, "Bi": 8.532}
        for s, b in sears.items():
            assert _NEUTRON_B[s] == b, (s, _NEUTRON_B[s], b)
        wk95 = {
            "O":  ((2.960427, 2.508818, 0.637853, 0.722838, 1.142756),
                   (14.182259, 5.936858, 0.112726, 34.958481, 0.390240), 0.027014),
            "Ga": ((15.758946, 6.841123, 4.121016, 2.714681, 2.395246),
                   (3.121754, 0.226057, 12.482196, 66.203622, 0.007238), -0.847395),
            "Cu": ((14.014192, 4.784577, 5.056806, 1.457971, 6.932996),
                   (3.738280, 0.003744, 13.034982, 72.554793, 0.265666), -3.254477),
        }
        for s, (a, b, c) in wk95.items():
            A, B, C = _WK95_XRAY[s]
            assert tuple(A) == a and tuple(B) == b and C == c, s

    def test_xray_form_factor_is_z_at_q0_and_decays(self):
        """Waasmaier-Kirfel f0(q): f0(0) = Z for every tabulated element,
        monotonically decreasing, and O falls off faster than Ga (which is
        why constant-Z weighting mis-weights partials at finite q)."""
        import numpy as np
        from ase.data import atomic_numbers
        from amorphgen.analysis.rdf import xray_form_factor, _WK95_XRAY

        assert len(_WK95_XRAY) >= 90
        for s in _WK95_XRAY:
            assert abs(float(xray_form_factor(s, 0.0)) - atomic_numbers[s]) < 0.05, s
        q = np.linspace(0.0, 12.0, 50)
        for s in ("O", "Si", "Ga", "In"):
            f = xray_form_factor(s, q)
            assert np.all(np.diff(f) <= 1e-9), s        # non-increasing
        # relative fall-off at the a-Ga2O3 first peak
        assert xray_form_factor("O", 2.45) / 8.0 < xray_form_factor("Ga", 2.45) / 31.0

    def test_xray_weighting_uses_q_dependent_f(self):
        """With q-dependent f(q) the x-ray FZ total must differ from the
        constant-Z one at finite q (regression: it was Z-weighted). Also
        the FZ weights must still sum to 1 so S(q->inf) = 1."""
        import numpy as np
        from amorphgen.analysis.rdf import (compute_structure_factor,
                                            xray_form_factor)
        atoms = self._random_binary()
        out = compute_structure_factor([atoms], weighting="xray",
                                       qmax=14.0, nq=140)
        q = np.array(out["q"]); s = np.array(out["s_q"])
        hi = s[(q > 9) & ~np.isnan(s)]
        assert abs(hi.mean() - 1.0) < 0.15
        # Rebuild the Z-weighted total from the partials and confirm the
        # q-dependent result is not simply the Z-weighted one.
        syms = atoms.get_chemical_symbols(); n = len(syms)
        c = {e: syms.count(e) / n for e in ("Na", "Ta")}
        parts = {p: np.array(compute_structure_factor([atoms], pair=p, qmax=14.0,
                                                      nq=140, weighting="unweighted")["s_q"])
                 for p in ("Na-Na", "Na-Ta", "Ta-Ta")}
        Z = {"Na": 11.0, "Ta": 73.0}; zm = c["Na"] * Z["Na"] + c["Ta"] * Z["Ta"]
        s_z = (c["Na"]**2 * Z["Na"]**2 * parts["Na-Na"]
               + 2 * c["Na"] * c["Ta"] * Z["Na"] * Z["Ta"] * parts["Na-Ta"]
               + c["Ta"]**2 * Z["Ta"]**2 * parts["Ta-Ta"]) / zm**2
        fNa, fTa = xray_form_factor("Na", q), xray_form_factor("Ta", q)
        fm = c["Na"] * fNa + c["Ta"] * fTa
        s_f = (c["Na"]**2 * fNa**2 * parts["Na-Na"]
               + 2 * c["Na"] * c["Ta"] * fNa * fTa * parts["Na-Ta"]
               + c["Ta"]**2 * fTa**2 * parts["Ta-Ta"]) / fm**2
        assert np.allclose(s, s_f, atol=1e-8)      # implementation == f(q) FZ
        assert not np.allclose(s, s_z, atol=1e-3)  # and != constant-Z FZ

    def test_direct_sq_weighted_rebinning(self):
        """sigma_q re-binning: identity at 0; reduces high-q speckle; keeps
        the first-peak position; ignores empty (NaN) shells; raw kept."""
        import numpy as np
        from amorphgen.analysis.rdf import (compute_structure_factor_direct,
                                            _smooth_sq_weighted)
        atoms = self._random_binary()
        raw = compute_structure_factor_direct([atoms], qmax=14.0, nq=140,
                                              weighting="unweighted")
        sm = compute_structure_factor_direct([atoms], qmax=14.0, nq=140,
                                             weighting="unweighted", sigma_q=0.1)
        q = np.array(raw["q"]); s0 = np.array(raw["s_q"]); s1 = np.array(sm["s_q"])
        assert "s_q_raw" not in raw and np.allclose(np.array(sm["s_q_raw"]), s0,
                                                    equal_nan=True)
        # sigma_q = 0 is the identity
        n = np.array(raw["n_per_bin"])
        assert np.allclose(_smooth_sq_weighted(q, s0, n, 0.0), s0, equal_nan=True)
        # empty shells stay NaN, populated shells become finite
        assert np.all(np.isnan(s1[n == 0])) and np.all(np.isfinite(s1[n > 0]))
        # noise (rms of successive differences) drops at high q
        hi = (q > 8) & np.isfinite(s0) & np.isfinite(s1)
        assert np.std(np.diff(s1[hi])) < np.std(np.diff(s0[hi]))

    def test_direct_partials_recombine_to_total_exactly(self):
        """Faber-Ziman partials from the direct method must rebuild the
        weighted total to machine precision (neutron weights are constant
        within a shell, so the identity is exact after shell averaging)."""
        import numpy as np
        from amorphgen.analysis.rdf import (compute_structure_factor_direct,
                                            _NEUTRON_B)
        atoms = self._random_binary()                     # Na30Ta30
        r = compute_structure_factor_direct([atoms], qmax=12.0, nq=120,
                                            weighting="neutron", partials=True)
        S = np.array(r["s_q"]); P = {k: np.array(v) for k, v in r["partials"].items()}
        assert set(P) == {"Na-Na", "Na-Ta", "Ta-Ta"}
        c = 0.5; bN, bT = _NEUTRON_B["Na"], _NEUTRON_B["Ta"]; bm = c * (bN + bT)
        rec = (c*c*bN*bN*P["Na-Na"] + 2*c*c*bN*bT*P["Na-Ta"] + c*c*bT*bT*P["Ta-Ta"]) / bm**2
        ok = np.isfinite(S)
        assert np.max(np.abs(rec[ok] - S[ok])) < 1e-10
        q = np.array(r["q"])
        for v in P.values():                              # each partial -> 1
            assert abs(np.nanmean(v[q > 9]) - 1.0) < 0.2

    def test_ft_sq_high_q_plateau_is_one(self):
        """FT S(q) on random positions must plateau at 1 at high q."""
        import numpy as np
        from amorphgen.analysis.rdf import compute_structure_factor
        out = compute_structure_factor([self._random_binary()],
                                       weighting="unweighted",
                                       qmax=14.0, nq=140)
        q = np.array(out["q"]); s = np.array(out["s_q"])
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


# ── auto-rdf cutoff robustness ───────────────────────────────────────────────

class TestAutoCutoffRdf:
    """auto_cutoff_rdf must find the real first shell, not placement noise."""

    def test_unrelaxed_random_structure_cutoff_beyond_first_peak(self):
        # Unrelaxed random placement: broad first shell with a noisy rising
        # edge.  The old detector latched onto the first bump above threshold
        # and returned a cutoff BELOW the Si-O bond length (CN ~ 0.2).
        from amorphgen.pipeline.random_gen import generate_random
        from amorphgen.analysis.cutoff import auto_cutoff_rdf
        from amorphgen.analysis.rdf import compute_rdf
        st = [generate_random({"Si": 24, "O": 48}, seed=s) for s in (1, 2, 3)]
        cut = auto_cutoff_rdf(st)
        key = "Si-O" if "Si-O" in cut else "O-Si"
        r = compute_rdf(st, pair="Si-O", rmax=4.0, nbins=200, sigma=0.0)
        rr, g = np.array(r["r"]), np.array(r["g_r"])
        r_peak = rr[np.argmax(g)]
        assert r_peak < cut[key] < r_peak * 1.6, (cut[key], r_peak)
        # and the resulting coordination is that of a bonded network, not ~0
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(st)
        cn = sa.coordination()["Si-O"]
        cn = cn["mean"] if isinstance(cn, dict) else cn
        assert cn > 2.5

    def test_rattled_crystal_cutoff_between_shells(self):
        # Rocksalt MgO: Mg-O shells at 2.10 and 3.64 A; cutoff must sit between.
        from amorphgen.analysis.cutoff import auto_cutoff_rdf
        rng = np.random.default_rng(0)
        st = []
        for k in range(3):
            a = bulk("MgO", "rocksalt", a=4.21).repeat((3, 3, 3))
            a.positions += rng.normal(scale=0.06, size=a.positions.shape)
            st.append(a)
        cut = auto_cutoff_rdf(st)
        key = "Mg-O" if "Mg-O" in cut else "O-Mg"
        assert 2.3 < cut[key] < 3.4, cut[key]

    def test_fallback_warns_when_no_minimum(self):
        # rmax below the first Mg-O shell (2.10 A): g(r) has no peak at all in
        # range -> radii-table fallback with a warning, never a silent nonsense
        # cutoff.
        import warnings
        from amorphgen.analysis.cutoff import auto_cutoff_rdf, auto_cutoff_minsep
        st = [bulk("MgO", "rocksalt", a=4.21).repeat((2, 2, 2))]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cut = auto_cutoff_rdf(st, rmax=1.8, nbins=90)
        key = "Mg-O" if "Mg-O" in cut else "O-Mg"
        assert cut[key] == pytest.approx(auto_cutoff_minsep(st)[key])
        assert any("no clear first minimum" in str(x.message) for x in w)


def test_analyser_counts_each_stem_once(tmp_path):
    """The optimiser writes s_opt.xyz AND s_opt.cif; --analyse must not count both."""
    from ase.build import bulk
    from ase.io import write
    from amorphgen.analysis import StructureAnalyser
    a = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    write(str(tmp_path / "s_opt.xyz"), a, format="extxyz")
    write(str(tmp_path / "s_opt.cif"), a, format="cif")
    write(str(tmp_path / "t_opt.cif"), a, format="cif")
    sa = StructureAnalyser(str(tmp_path))
    assert len(sa.atoms_list) == 2
    assert [os.path.basename(f) for f in sa.file_paths] == ["s_opt.xyz", "t_opt.cif"] if hasattr(sa, "file_paths") else True


def test_is_bonding_pair_rules():
    """Cation-anion pairs bond; cation-cation (Ga-In, Al-Si, Na-Si) and
    anion-anion pairs do not, whatever the radii table calls them; systems
    without an anion fall back to the radii classification."""
    from amorphgen.analysis.structure import is_bonding_pair as b
    igzo = {"In", "Ga", "Zn", "O"}
    assert b("Ga", "O", igzo) and b("O", "In", igzo)
    assert not b("Ga", "In", igzo) and not b("O", "O", igzo) and not b("Zn", "Zn", igzo)
    nas = {"Na", "Al", "Si", "O"}
    assert b("Si", "O", nas) and b("Al", "O", nas) and b("Na", "O", nas)
    assert not b("Al", "Si", nas) and not b("Na", "Si", nas)
    assert b("P", "O", {"Li", "P", "O"}) and b("Si", "N", {"Si", "N"})
    assert b("Si", "Si", {"Si"}) and b("Si", "C", {"Si", "C"}) and b("Ga", "As", {"Ga", "As"})
    assert b("Cu", "Zr", {"Cu", "Zr"}) and b("Cu", "Cu", {"Cu", "Zr"})


class TestBondingPairRules:
    """Review round 4 (2026-09-29): which contacts count as first-shell bonds."""

    @staticmethod
    def _b(a, b, els):
        from amorphgen.analysis.structure import is_bonding_pair
        return is_bonding_pair(a, b, els)

    def test_compound_of_two_anion_elements_has_a_cation(self):
        """TeO2 and SO3 are made only of elements on the anion list; the least
        electronegative one is the cation, so Te-O and S-O are bonds and the
        coordination report is not empty."""
        assert self._b("Te", "O", {"Te", "O"}) and not self._b("O", "O", {"Te", "O"})
        assert self._b("S", "O", {"S", "O"}) and not self._b("S", "S", {"S", "O"})
        assert self._b("Se", "O", {"Se", "O"})

    def test_hydrogen_is_an_anion_only_in_a_hydride(self):
        assert self._b("Li", "H", {"Li", "H"})                    # hydride
        assert not self._b("H", "H", {"Mg", "H"})
        assert self._b("O", "H", {"Na", "O", "H"})                # hydroxide: H is the cation
        assert self._b("Na", "O", {"Na", "O", "H"})
        assert not self._b("Na", "H", {"Na", "O", "H"})

    def test_several_true_anions_all_bond_to_the_cation(self):
        els = {"Bi", "O", "Cl"}
        assert self._b("Bi", "O", els) and self._b("Bi", "Cl", els)
        assert not self._b("O", "Cl", els)

    def test_same_element_bonds_only_in_elements_and_metal_alloys(self):
        assert self._b("Si", "Si", {"Si"})                        # a-Si
        assert self._b("Cu", "Cu", {"Cu", "Zr"})                  # metallic glass
        assert not self._b("Ga", "Ga", {"Ga", "As"})              # III-V, not a metal
        assert not self._b("In", "In", {"In", "P"})
        assert not self._b("Ti", "Ti", {"Ti", "C"})               # carbide
        assert not self._b("Ga", "In", {"In", "Ga", "Zn", "O"})   # cation-cation in an oxide

    def test_anion_set_follows_charge_balance(self):
        """Review round 5: membership of the anion table is not enough. The
        same element is an anion or a cation depending on what it is with."""
        from amorphgen.analysis.structure import anion_elements as ae
        assert ae({"Cd": 1, "Te": 1}) == {"Te"}          # telluride: Te is the anion
        assert ae({"Te": 1, "O": 2}) == {"O"}            # tellurite: Te is the cation
        assert ae({"Na": 2, "Te": 1, "O": 4}) == {"O"}   # ... even with a real cation present
        assert ae({"La": 2, "O": 2, "S": 1}) == {"O", "S"}   # oxysulfide: both balance
        assert ae({"H": 2, "S": 1, "O": 4}) == {"O"}     # sulfate: H and S are cations
        assert ae({"Na": 1, "N": 1, "O": 3}) == {"O"}    # nitrate
        assert ae({"Li": 1, "H": 1}) == {"H"}            # hydride
        assert ae({"Na": 1, "O": 1, "H": 1}) == {"O"}    # hydroxide
        assert ae({"Bi": 1, "O": 1, "Cl": 1}) == {"O", "Cl"}
        assert ae({"Cu": 1, "Zr": 1}) == set()

    def test_metal_rich_glasses_keep_their_metal_metal_bonds(self):
        """Ni80P20, Fe80B20 and Pd80Si20 are classified as alloys, so their
        metal-metal contacts must count as bonds; GaAs and TiC are not metals."""
        assert self._b("Ni", "Ni", {"Ni": 80, "P": 20})
        assert self._b("Fe", "Fe", {"Fe": 80, "B": 20})
        assert self._b("Pd", "Pd", {"Pd": 80, "Si": 20})
        assert self._b("Cu", "Cu", {"Cu": 50, "Zr": 50})
        assert not self._b("Ga", "Ga", {"Ga": 1, "As": 1})
        assert not self._b("Ti", "Ti", {"Ti": 1, "C": 1})
        assert not self._b("Be", "Be", {"Be": 2, "C": 1})

    def test_off_stoichiometry_cells_keep_an_anion(self):
        """A charge-unbalanced composition (a random test cell, a defective
        model) must not end up with no anion at all: the most electronegative
        element is never demoted."""
        from amorphgen.analysis.structure import anion_elements as ae
        assert ae({"Ga": 16, "Zn": 16, "O": 48}) == {"O"}
        assert ae({"Si": 30, "O": 40}) == {"O"}
        assert self._b("Ga", "O", {"Ga": 16, "Zn": 16, "O": 48})
        assert not self._b("Ga", "Zn", {"Ga": 16, "Zn": 16, "O": 48})

    def test_mixed_chalcogen_glasses_keep_every_chalcogen_as_an_anion(self):
        """Review round 6: with two chalcogens and no oxidiser present, neither
        may be promoted to cation (Ge-S-Se, Ge-Se-Te, Ge-Sb-Te)."""
        from amorphgen.analysis.structure import anion_elements as ae
        assert ae({"Ge": 20, "S": 10, "Se": 70}) == {"S", "Se"}
        assert ae({"Ge": 20, "Se": 40, "Te": 40}) == {"Se", "Te"}
        assert ae({"Ge": 2, "Sb": 2, "Te": 5}) == {"Te"}          # one chalcogen only
        assert ae({"As": 40, "S": 30, "Se": 30}) == {"S", "Se"}   # two, neither promoted
        assert self._b("As", "S", {"As": 40, "S": 30, "Se": 30})
        assert self._b("As", "Se", {"As": 40, "S": 30, "Se": 30})
        assert not self._b("S", "Se", {"As": 40, "S": 30, "Se": 30})
        assert self._b("Ge", "Se", {"Ge": 20, "S": 10, "Se": 70})
        assert self._b("Ge", "Te", {"Ge": 20, "Se": 40, "Te": 40})
        assert not self._b("S", "Se", {"Ge": 20, "S": 10, "Se": 70})
        assert not self._b("Se", "Te", {"Ge": 20, "Se": 40, "Te": 40})

    def test_dopants_and_defects_do_not_flip_the_major_anion(self):
        """A single dopant or defect atom must not turn the major anion into a
        cation: promoting it would overshoot charge balance, so it is kept."""
        from amorphgen.analysis.structure import anion_elements as ae
        # promoting the major anion here would overshoot balance, so it is kept
        assert ae({"Si": 32, "O": 64, "F": 2}) == {"O", "F"}      # F-doped silica
        assert ae({"Na": 32, "Cl": 32, "O": 1}) == {"Cl", "O"}    # O impurity in NaCl
        assert ae({"Li": 29, "P": 10, "O": 33, "N": 5}) == {"O", "N"}   # LiPON
        assert self._b("Si", "O", {"Si": 32, "O": 64, "F": 2})
        assert self._b("Na", "Cl", {"Na": 32, "Cl": 32, "O": 1})
        assert self._b("P", "N", {"Li": 29, "P": 10, "O": 33, "N": 5})
        assert not self._b("N", "O", {"Li": 29, "P": 10, "O": 33, "N": 5})
        # ... while a real tellurite still promotes Te, oxidiser present
        assert ae({"Na": 2, "Te": 1, "O": 4}) == {"O"}


def test_metal_rich_glass_through_the_analyser_and_plotter(tmp_path):
    """Review round 11: the analyser and the plotter must pass element COUNTS,
    not a bare set, or the metal-fraction test cannot fire and Ni80P20 loses its
    Ni-Ni bonds. Drives the real report and plot, so reverting the Counter call
    sites fails here."""
    import numpy as np
    from ase import Atoms
    from amorphgen.analysis import StructureAnalyser
    rng = np.random.default_rng(0)
    n_ni, n_p = 80, 20
    a = Atoms("Ni80P20", positions=rng.uniform(0, 11, (n_ni + n_p, 3)),
              cell=[11] * 3, pbc=True)
    sa = StructureAnalyser([a], cutoff=3.0)
    text = sa.summary()
    body = text if isinstance(text, str) else "\n".join(text)
    assert "Ni-Ni" in body.split("Non-bonded contacts")[0], body[:400]
    cn = sa.coordination()
    assert cn["Ni-Ni"]["mean"] > 0
    assert sa.total_coordination(centre="Ni")["Ni"]["mean"] > 0
    sa.plot(output_dir=str(tmp_path))          # the plotter takes the same path
    assert (tmp_path / "analysis_cn.csv").exists()
    assert "Ni-Ni" in (tmp_path / "analysis_cn.csv").read_text()


class TestTotalCorrelationFunction:
    """T(r) = 4 pi r rho g(r) in the diffraction convention."""

    @staticmethod
    def _fcc_cu(n=3, a=3.61):
        from ase.build import bulk
        return bulk("Cu", "fcc", a=a, cubic=True).repeat((n, n, n))

    def test_first_peak_gives_the_known_coordination(self):
        """For fcc Cu the first shell holds 12 atoms, and the area under
        r*T(r) over that peak must recover it. This is what a diffraction
        paper integrates, so it pins the normalisation, the density and the
        transform together."""
        from amorphgen.analysis import StructureAnalyser
        from amorphgen.analysis.rdf import coordination_from_Tr
        sa = StructureAnalyser([self._fcc_cu()])
        tr = sa.total_correlation(weighting="unweighted", qmin=0.5, qmax=25.0,
                                  nq=600, rmax=6.0, nr=600)
        n = coordination_from_Tr(tr, 2.0, 3.2)
        assert 10.5 <= n <= 13.5, n          # 12 with transform broadening

    def test_peak_position_and_density(self):
        from amorphgen.analysis import StructureAnalyser
        import numpy as np
        cu = self._fcc_cu()
        sa = StructureAnalyser([cu])
        tr = sa.total_correlation(weighting="unweighted", qmin=0.5, qmax=25.0, rmax=6.0)
        r = np.asarray(tr["r"]); T = np.asarray(tr["T_r"])
        first = r[np.argmax(np.where(r < 3.2, T, -np.inf))]
        assert abs(first - 3.61 / np.sqrt(2)) < 0.12, first     # fcc nearest neighbour
        assert abs(tr["rho"] - len(cu) / cu.get_volume()) < 1e-9
        # G(r) = T(r) - 4 pi r rho, by definition
        G = np.asarray(tr["G_r"]); g = np.asarray(tr["g_r"])
        assert np.allclose(G, T - 4 * np.pi * r * tr["rho"], atol=1e-8)
        assert np.allclose(T, 4 * np.pi * r * tr["rho"] * g, atol=1e-8)

    def test_weighting_changes_the_curve_for_a_multi_element_system(self):
        """The weighted g(r) is NOT the unweighted one: in IGZO the indium
        correlations dominate the X-ray weighting."""
        import numpy as np
        from ase import Atoms
        from amorphgen.analysis import StructureAnalyser
        rng = np.random.default_rng(0)
        a = Atoms("In8Ga8Zn8O32", positions=rng.uniform(0, 12, (56, 3)),
                  cell=[12] * 3, pbc=True)
        sa = StructureAnalyser([a])
        x = np.asarray(sa.total_correlation(weighting="xray", qmin=0.6, qmax=18.0)["g_r"])
        u = np.asarray(sa.total_correlation(weighting="unweighted", qmin=0.6, qmax=18.0)["g_r"])
        assert not np.allclose(x, u, atol=0.05)

    def test_window_and_bad_arguments(self):
        import pytest
        from amorphgen.analysis import StructureAnalyser
        import numpy as np
        sa = StructureAnalyser([self._fcc_cu()])
        lo = np.asarray(sa.total_correlation(qmin=0.5, qmax=25.0, window="lorch")["T_r"])
        no = np.asarray(sa.total_correlation(qmin=0.5, qmax=25.0, window=None)["T_r"])
        assert not np.allclose(lo, no)          # the window changes the ripple
        with pytest.raises(ValueError, match="window must be"):
            sa.total_correlation(window="hann")
        with pytest.raises(ValueError, match="usable S\\(Q\\) points"):
            sa.total_correlation(qmin=30.0, qmax=20.0)   # no S(Q) survives qmin

    def test_cli_tr_flag_writes_the_plot_and_csv(self, tmp_path, monkeypatch, capsys):
        import sys
        from ase.io import write
        from amorphgen.cli import main
        src = tmp_path / "cu.xyz"
        write(str(src), self._fcc_cu(), format="extxyz")
        plots = tmp_path / "p"
        report = tmp_path / "report.txt"
        monkeypatch.setattr(sys, "argv", ["amorphgen", "--analyse", str(src), "--tr",
                                          "--tr-qrange", "0.5", "25", "--tr-window", "none",
                                          "--save-plot", str(plots),
                                          "--save-report", str(report)])
        main()
        out = capsys.readouterr().out
        assert "T(r): xray weighting, q = 0.5-25.0" in out and "none window" in out
        assert "first peak at r =" in out
        # the summary belongs in the saved report too, not only on screen
        saved = report.read_text()
        assert "T(r) (xray, q = 0.5-25.0 1/A, none window)" in saved
        assert "first peak at r =" in saved or "no resolved first peak" in saved
        assert (plots / "analysis_tr.png").exists()
        head = (plots / "analysis_tr.csv").read_text().splitlines()
        assert "window=None" in head[0] and "qmax=25.0" in head[0]
        assert head[1].startswith("r_A,g_r_weighted,T_r_invA2,G_r")

    def test_first_peak_is_the_first_not_the_tallest(self):
        """In an oxide the second shell is taller than the first, so a maximum
        over a fixed window lands on the second peak's rising edge. The helper
        must return the first LOCAL maximum and the minima either side."""
        import numpy as np
        from ase import Atoms
        from amorphgen.analysis import StructureAnalyser
        from amorphgen.analysis.rdf import first_Tr_peak
        rng = np.random.default_rng(3)
        a = Atoms("In8Ga8Zn8O32", positions=rng.uniform(0, 12, (56, 3)),
                  cell=[12] * 3, pbc=True)
        tr = StructureAnalyser([a]).total_correlation(qmin=0.6, qmax=22.0)
        r = np.asarray(tr["r"]); T = np.asarray(tr["T_r"])
        pk, lo, hi = first_Tr_peak(tr)
        assert pk is not None and lo < pk < hi
        # it is a genuine local maximum, and earlier than the global one
        i = int(np.argmin(np.abs(r - pk)))
        assert T[i] >= T[i - 1] and T[i] >= T[i + 1]
        assert pk <= r[int(np.argmax(T))]

    def test_qmax_window_sensitivity_scan(self):
        """The q range and the window belong to the measurement, not the model,
        and both move T(r). The scan reports that spread; with a Lorch window
        the first peak is stable, without one the truncation ripple narrows the
        integration window and the count drifts down."""
        from ase import Atoms
        import numpy as np
        from amorphgen.analysis.rdf import scan_Tr_qmax, format_Tr_scan
        rng = np.random.default_rng(5)
        a = Atoms("In8Ga8Zn8O32", positions=rng.uniform(0, 12, (56, 3)),
                  cell=[12] * 3, pbc=True)
        rows = scan_Tr_qmax([a], qmax_values=(14.0, 18.0, 22.0), qmin=0.6)
        assert len(rows) == 6                       # two windows x three qmax
        lorch = [r for r in rows if r["window"] == "lorch" and r.get("r_peak")]
        assert len(lorch) == 3
        pk = [r["r_peak"] for r in lorch]
        assert max(pk) - min(pk) < 0.25             # stable under the window
        txt = format_Tr_scan(rows)
        assert "sensitivity" in txt and "lorch" in txt and "spread" in txt
        # a range with nothing in it is reported, not raised
        bad = scan_Tr_qmax([a], qmax_values=(5.0,), qmin=30.0, windows=("lorch",))
        assert "error" in bad[0] and "usable S(Q)" in bad[0]["error"]

    def test_first_peak_rejects_the_truncation_ripple(self):
        """Without a Lorch window T(r) carries a ripple before the first shell.
        T grows as 4 pi rho r, so a height threshold set as a fraction of the
        maximum is a threshold on r and the ripple clears it; the ripple is
        rejected on prominence instead. This is the a-Al2O3 failure, which used
        to report 1.3 to 1.5 A with a count of 0.01 for an Al-O shell at 1.82."""
        import numpy as np
        from amorphgen.analysis.rdf import first_Tr_peak

        r = np.linspace(0.05, 8.0, 800)
        rho = 0.07

        def shell(r0, amp, w):
            return amp * np.exp(-((r - r0) / w) ** 2)

        g = shell(1.20, 0.55, 0.10) + shell(1.90, 4.00, 0.12) + shell(3.20, 2.20, 0.25)
        res = {"r": r.tolist(), "g_r": g.tolist(), "rho": rho,
               "T_r": (4 * np.pi * r * rho * g).tolist()}
        T = np.asarray(res["T_r"])

        # the ripple is a local maximum that clears a plain height threshold,
        # which is why height alone picked it
        i_ripple = int(np.argmin(np.abs(r - 1.20)))
        assert T[i_ripple] > 0.05 * T.max()

        pk, lo, hi = first_Tr_peak(res)
        assert abs(pk - 1.90) < 0.08, f"picked {pk}, expected the shell at 1.90"
        assert lo < 1.90 < hi

    def test_first_peak_keeps_a_real_shell_below_the_mean_density(self):
        """A heavy scatterer can put the real first shell below g = 1: in Cu2O
        the X-ray weighting is dominated by Cu-Cu and the Cu-O shell has a
        weighted g of 0.81. The density test must not reject it."""
        import numpy as np
        from amorphgen.analysis.rdf import first_Tr_peak

        r = np.linspace(0.05, 8.0, 800)
        rho = 0.08

        def shell(r0, amp, w):
            return amp * np.exp(-((r - r0) / w) ** 2)

        g = shell(1.95, 0.80, 0.13) + shell(2.65, 2.40, 0.20)
        res = {"r": r.tolist(), "g_r": g.tolist(), "rho": rho,
               "T_r": (4 * np.pi * r * rho * g).tolist()}
        pk, _, _ = first_Tr_peak(res)
        assert abs(pk - 1.95) < 0.08, f"picked {pk}, expected the weak shell at 1.95"

    def test_cli_tr_scan_flag(self, tmp_path, monkeypatch, capsys):
        import sys
        from ase.io import write
        from amorphgen.cli import main
        src = tmp_path / "cu.xyz"
        write(str(src), self._fcc_cu(), format="extxyz")
        monkeypatch.setattr(sys, "argv", ["amorphgen", "--analyse", str(src),
                                          "--tr", "--tr-scan", "--tr-qrange", "0.5", "25"])
        main()
        out = capsys.readouterr().out
        assert "T(r) transform sensitivity" in out and "spread:" in out
