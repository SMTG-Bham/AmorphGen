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
