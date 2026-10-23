"""Tests for amorphgen.analysis.exafs (EXAFS first-shell descriptors).

Tier 1 — pure unit tests, no calculator. The physics assertions are against
closed-form values (a perfect crystal, hand-placed shells, the analytic MSRD
limit of an uncorrelated rattle) rather than against previously recorded
output, so a regression in the cumulant accumulation fails loudly.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from amorphgen.analysis.exafs import (
    DEFAULT_BOOTSTRAP,
    DEFAULT_MFP,
    KURT_THRESHOLD,
    compute_exafs_descriptors,
    format_exafs_report,
    shell_cumulants,
    write_exafs_csv,
)


def _flat_cutoff(value):
    """A get_cutoff_fn that ignores the pair and returns a fixed cutoff."""
    return lambda s1, s2: value


def _shell(distances, absorber="Si", scatterer="O", box=12.0):
    """One absorber at the box centre with scatterers at exact distances.

    Scatterers are spread over distinct directions so that only the
    absorber-scatterer distances are the ones asked for.
    """
    centre = np.array([box / 2] * 3)
    dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                     [0, 0, 1], [0, 0, -1]], dtype=float)
    if len(distances) > len(dirs):
        raise ValueError("at most 6 scatterers")
    positions = [centre] + [centre + d * dirs[k]
                            for k, d in enumerate(distances)]
    return Atoms(symbols=[absorber] + [scatterer] * len(distances),
                 positions=positions, cell=[box] * 3, pbc=False)


# ─── exact cumulants from a hand-placed shell ─────────────────────────────

class TestCumulantsExact:
    def test_symmetric_shell(self):
        """{1.5, 1.5, 1.7, 1.7}: mean 1.6, variance 0.01, C3 exactly 0."""
        atoms = _shell([1.5, 1.5, 1.7, 1.7])
        res = compute_exafs_descriptors([atoms], 2.0, _flat_cutoff(2.0))
        d = res["Si-O"]
        assert d["N"] == pytest.approx(4.0)
        assert d["R"] == pytest.approx(1.6)
        assert d["sigma2"] == pytest.approx(0.01)
        assert d["C3"] == pytest.approx(0.0, abs=1e-12)
        assert d["skew"] == pytest.approx(0.0, abs=1e-9)

    def test_asymmetric_shell_positive_c3(self):
        """{1.5, 1.5, 1.5, 1.9}: a tail to long R gives C3 > 0."""
        atoms = _shell([1.5, 1.5, 1.5, 1.9])
        d = compute_exafs_descriptors([atoms], 2.0, _flat_cutoff(2.0))["Si-O"]
        assert d["R"] == pytest.approx(1.6)
        assert d["sigma2"] == pytest.approx(0.03)
        assert d["C3"] == pytest.approx(0.006)
        assert d["skew"] > 0

    def test_asymmetric_shell_negative_c3(self):
        """Mirror image {1.3, 1.7, 1.7, 1.7}: tail to short R gives C3 < 0."""
        atoms = _shell([1.3, 1.7, 1.7, 1.7])
        d = compute_exafs_descriptors([atoms], 2.0, _flat_cutoff(2.0))["Si-O"]
        assert d["R"] == pytest.approx(1.6)
        assert d["C3"] == pytest.approx(-0.006)
        assert d["skew"] < 0

    def test_sample_sizes(self):
        atoms = _shell([1.5, 1.5, 1.7, 1.7])
        d = compute_exafs_descriptors([atoms, atoms], 2.0,
                                      _flat_cutoff(2.0))["Si-O"]
        assert d["n_paths"] == 8           # 4 scatterers x 2 structures
        assert d["n_absorbers"] == 2       # 1 absorber x 2 structures
        assert d["n_structures"] == 2


# ─── a perfect crystal has zero configurational disorder ──────────────────

class TestFourthCumulant:
    """C4 and the excess kurtosis C4 / sigma^4, against closed forms.

    Stratified quantiles reproduce a distribution's moments far better
    than a random draw, so these compare against exact targets.
    """

    Q = (np.arange(20000) + 0.5) / 20000

    def _cum(self, d):
        return shell_cumulants(np.asarray(d, dtype=float), np.array([4.0]))

    def test_gaussian_has_zero_excess_kurtosis(self):
        from scipy.stats import norm
        assert self._cum(norm.ppf(self.Q, 2.0, 0.1))["kurt"] == \
            pytest.approx(0.0, abs=0.02)

    def test_uniform_is_minus_six_fifths(self):
        from scipy.stats import uniform
        assert self._cum(uniform.ppf(self.Q, 1.8, 0.4))["kurt"] == \
            pytest.approx(-1.2, abs=0.01)

    def test_c4_is_kurtosis_times_sigma_to_the_fourth(self):
        from scipy.stats import skewnorm
        c = self._cum(1.9 + 0.1 * skewnorm.ppf(self.Q, 4.0))
        assert c["C4"] == pytest.approx(c["kurt"] * c["sigma2"] ** 2)

    def test_two_point_shell_through_the_descriptor_pipeline(self):
        """Two bond lengths, two each: kurtosis is exactly -2, C4 -2 sigma^4.

        A split shell (4+2 Jahn-Teller, say) is the textbook case a single
        Gaussian shell cannot describe, though its skew is zero.
        """
        res = compute_exafs_descriptors([_shell([1.5, 1.5, 1.7, 1.7])], 2.0,
                                        _flat_cutoff(2.0))["Si-O"]
        assert res["skew"] == pytest.approx(0.0, abs=1e-9)
        assert res["kurt"] == pytest.approx(-2.0)
        assert res["C4"] == pytest.approx(-2.0 * 0.01 ** 2)

    def test_zero_width_shell_is_not_a_division_by_zero(self):
        c = self._cum([1.6, 1.6, 1.6])
        assert c["kurt"] == 0.0 and c["skew"] == 0.0


class TestPerfectCrystal:
    def test_nearly_constant_distances_have_finite_normalised_histogram(self):
        # Periodic copies of a crystal can differ by a few rounding units.
        # Unlike exactly equal values, NumPy does not widen this tiny range.
        atoms = _shell([2.0, 2.0 + 4 * np.spacing(2.0)], absorber="Nb", scatterer="O")
        d = compute_exafs_descriptors([atoms], 2.5, _flat_cutoff(2.5),
                                      pair="Nb-O", bootstrap=0)["Nb-O"]
        r, p = np.asarray(d["P_R"]["r"]), np.asarray(d["P_R"]["p"])
        assert np.isfinite(p).all()
        assert (np.diff(r) > 0).all()
        assert np.sum(p) * (r[1] - r[0]) == pytest.approx(1.0)
        assert d["N"] == 2
        assert d["R"] == pytest.approx(2.0)
        assert d["sigma2"] < 1e-25

    @pytest.fixture
    def diamond_si(self):
        return bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)

    def test_exact_n_and_r(self, diamond_si):
        d = compute_exafs_descriptors([diamond_si], 2.8,
                                      _flat_cutoff(2.8))["Si-Si"]
        assert d["N"] == pytest.approx(4.0)
        assert d["R"] == pytest.approx(5.43 * np.sqrt(3) / 4)

    def test_zero_disorder(self, diamond_si):
        d = compute_exafs_descriptors([diamond_si], 2.8,
                                      _flat_cutoff(2.8))["Si-Si"]
        assert d["sigma2"] == pytest.approx(0.0, abs=1e-20)
        assert d["C3"] == pytest.approx(0.0, abs=1e-20)
        assert d["skew"] == 0.0            # guarded against 0/0

    def test_r_eff_equals_r_when_ordered(self, diamond_si):
        """The effective-distribution shift is proportional to sigma^2."""
        d = compute_exafs_descriptors([diamond_si], 2.8,
                                      _flat_cutoff(2.8))["Si-Si"]
        assert d["R_eff"] == pytest.approx(d["R"])


class TestMSRDLimit:
    """sigma^2 must be the mean-square RELATIVE displacement along the bond.

    Rattling every atom by an uncorrelated Gaussian of width s per Cartesian
    component gives Var(bond length) -> 2 s^2 to first order. Summing two
    independent 3D mean-square displacements instead would give 6 s^2, so
    this test would catch that confusion.
    """

    def test_matches_two_sigma_squared(self):
        s = 0.05
        base = bulk("Si", "diamond", a=5.43, cubic=True) * (3, 3, 3)
        frames = []
        for seed in range(20):
            at = base.copy()
            at.positions += np.random.default_rng(seed).normal(
                0, s, at.positions.shape)
            frames.append(at)

        d = compute_exafs_descriptors(frames, 2.9, _flat_cutoff(2.9))["Si-Si"]
        assert d["N"] == pytest.approx(4.0)
        assert d["sigma2"] == pytest.approx(2 * s ** 2, rel=0.05)


# ─── effective-distribution correction ────────────────────────────────────

class TestEffectiveDistribution:
    @pytest.fixture
    def descriptors(self):
        atoms = _shell([1.5, 1.5, 1.7, 1.7])
        return compute_exafs_descriptors(
            [atoms], 2.0, _flat_cutoff(2.0))["Si-O"]

    def test_shift_is_negative(self, descriptors):
        """1/R^2 and exp(-2R/lambda) both favour short R, so R_eff < R."""
        assert descriptors["R_eff"] < descriptors["R"]

    def test_shift_matches_formula(self, descriptors):
        r, s2 = descriptors["R"], descriptors["sigma2"]
        expected = r - 2 * s2 * (1 / r + 1 / DEFAULT_MFP)
        assert descriptors["R_eff"] == pytest.approx(expected)

    def test_shorter_mfp_gives_larger_shift(self):
        atoms = _shell([1.5, 1.5, 1.7, 1.7])
        short = compute_exafs_descriptors([atoms], 2.0, _flat_cutoff(2.0),
                                          mfp=3.0)["Si-O"]
        long = compute_exafs_descriptors([atoms], 2.0, _flat_cutoff(2.0),
                                         mfp=8.0)["Si-O"]
        assert short["R_eff"] < long["R_eff"]
        assert short["mfp"] == 3.0


# ─── ordered pair semantics ───────────────────────────────────────────────

class TestOrderedPairs:
    @pytest.fixture
    def sio2_like(self):
        """Two Si, each with 2 O; every O bridges both (Si-O 2, O-Si 2)."""
        box = 12.0
        return Atoms(
            symbols=["Si", "Si", "O", "O"],
            positions=[[5.0, 6.0, 6.0], [7.0, 6.0, 6.0],
                       [6.0, 6.8, 6.0], [6.0, 5.2, 6.0]],
            cell=[box] * 3, pbc=False,
        )

    def test_both_directions_present(self, sio2_like):
        res = compute_exafs_descriptors([sio2_like], 2.0, _flat_cutoff(2.0))
        assert "Si-O" in res and "O-Si" in res

    def test_distance_statistics_are_symmetric(self, sio2_like):
        """Same set of distances seen from either end — only N differs."""
        res = compute_exafs_descriptors([sio2_like], 2.0, _flat_cutoff(2.0))
        for key in ("R", "sigma2", "C3", "R_eff"):
            assert res["Si-O"][key] == pytest.approx(res["O-Si"][key])

    def test_n_is_directional(self):
        """SiO2 stoichiometry: N(Si-O) must be twice N(O-Si)."""
        atoms = Atoms(
            symbols=["Si", "O", "O", "O", "O"],
            positions=[[6, 6, 6], [7.6, 6, 6], [4.4, 6, 6],
                       [6, 7.6, 6], [6, 4.4, 6]],
            cell=[12] * 3, pbc=False,
        )
        res = compute_exafs_descriptors([atoms], 2.0, _flat_cutoff(2.0))
        assert res["Si-O"]["N"] == pytest.approx(4.0)
        assert res["O-Si"]["N"] == pytest.approx(1.0)

    def test_pair_filter_returns_only_that_pair(self, sio2_like):
        res = compute_exafs_descriptors([sio2_like], 2.0, _flat_cutoff(2.0),
                                        pair="Si-O")
        assert list(res) == ["Si-O"]

    def test_pair_filter_is_directional(self, sio2_like):
        res = compute_exafs_descriptors([sio2_like], 2.0, _flat_cutoff(2.0),
                                        pair="O-Si")
        assert list(res) == ["O-Si"]
        assert res["O-Si"]["N"] == pytest.approx(2.0)

    def test_malformed_pair_raises(self, sio2_like):
        with pytest.raises(ValueError, match="absorber"):
            compute_exafs_descriptors([sio2_like], 2.0, _flat_cutoff(2.0),
                                      pair="Si")

    def test_pair_with_no_neighbours_omitted(self):
        """Nothing inside the cutoff means no row, not a zero row."""
        far = Atoms(symbols=["Si", "O"], positions=[[1, 1, 1], [8, 8, 8]],
                    cell=[12] * 3, pbc=False)
        res = compute_exafs_descriptors([far], 2.0, _flat_cutoff(2.0))
        assert res == {}


# ─── consistency with the rest of the analyser ────────────────────────────

class TestAnalyserIntegration:
    @pytest.fixture
    def rattled_si(self):
        base = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
        out = []
        for seed in range(3):
            at = base.copy()
            at.positions += np.random.default_rng(seed).normal(
                0, 0.05, at.positions.shape)
            out.append(at)
        return out

    def test_n_matches_coordination(self, rattled_si):
        """N is the same quantity coordination() reports — by construction."""
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(rattled_si, cutoff=2.9)
        assert (sa.exafs_descriptors()["Si-Si"]["N"]
                == pytest.approx(sa.coordination()["Si-Si"]["mean"]))

    def test_r_matches_bond_distances(self, rattled_si):
        """R is the first cumulant of the bond_distances() distance set."""
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(rattled_si, cutoff=2.9)
        assert (sa.exafs_descriptors()["Si-Si"]["R"]
                == pytest.approx(sa.bond_distances()["Si-Si"]["mean"]))

    def test_sigma2_matches_bond_distance_variance(self, rattled_si):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(rattled_si, cutoff=2.9)
        bd = sa.bond_distances()["Si-Si"]
        assert (sa.exafs_descriptors()["Si-Si"]["sigma2"]
                == pytest.approx(bd["std"] ** 2, rel=1e-6))

    def test_analyser_passes_mfp_through(self, rattled_si):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(rattled_si, cutoff=2.9)
        assert sa.exafs_descriptors(mfp=3.0)["Si-Si"]["mfp"] == 3.0

    def test_cutoff_recorded(self, rattled_si):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(rattled_si, cutoff=2.9)
        assert sa.exafs_descriptors()["Si-Si"]["cutoff"] == pytest.approx(2.9)

    def test_p_r_distribution_normalised(self, rattled_si):
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser(rattled_si, cutoff=2.9)
        p_r = sa.exafs_descriptors()["Si-Si"]["P_R"]
        r, p = np.array(p_r["r"]), np.array(p_r["p"])
        assert len(r) == len(p) == 120
        # np.trapz was renamed np.trapezoid in NumPy 2; support both.
        trapz = getattr(np, "trapezoid", None) or np.trapz
        assert trapz(p, r) == pytest.approx(1.0, rel=0.02)


# ─── report and CSV ───────────────────────────────────────────────────────

class TestReport:
    @pytest.fixture
    def result(self):
        atoms = _shell([1.5, 1.5, 1.7, 1.7])
        return compute_exafs_descriptors([atoms], 2.0, _flat_cutoff(2.0))

    def test_report_has_headers_and_row(self, result):
        out = format_exafs_report(result)
        assert "EXAFS first-shell descriptors" in out
        assert "sigma2" in out
        assert "Si-O" in out

    def test_report_states_static_caveat(self, result):
        """The 0 K sigma^2 caveat must be impossible to miss."""
        out = format_exafs_report(result)
        assert "CONFIGURATIONAL" in out
        assert "thermal" in out

    def test_report_names_the_mfp(self, result):
        assert "lambda = 5.0 A" in format_exafs_report(result)

    def test_empty_result_message(self):
        assert "No EXAFS descriptors" in format_exafs_report({})

    def test_bonded_only_drops_contacts(self):
        """O-O is a second-shell contact, not a first-shell EXAFS path."""
        atoms = Atoms(
            symbols=["Si", "O", "O"],
            positions=[[6, 6, 6], [7.6, 6, 6], [6, 7.6, 6]],
            cell=[12] * 3, pbc=False,
        )
        res = compute_exafs_descriptors([atoms], 2.6, _flat_cutoff(2.6))
        assert "O-O" in res
        assert "O-O" not in format_exafs_report(res, bonded_only=True)
        assert "O-O" in format_exafs_report(res, bonded_only=False)

    def test_single_element_still_reported(self):
        """With no bonded pair at all, fall back to showing every pair."""
        si = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
        res = compute_exafs_descriptors([si], 2.8, _flat_cutoff(2.8))
        assert "Si-Si" in format_exafs_report(res, bonded_only=True)

    def test_skew_warning_appears_for_asymmetric_shell(self):
        atoms = _shell([1.45, 1.45, 1.45, 1.95])
        res = compute_exafs_descriptors([atoms], 2.2, _flat_cutoff(2.2))
        assert abs(res["Si-O"]["skew"]) > 0.3
        assert "harmonic" in format_exafs_report(res)

    def test_no_skew_warning_for_symmetric_shell(self, result):
        assert "Asymmetric shell" not in format_exafs_report(result)

    @staticmethod
    def _entry(skew, kurt, absorber="Ca", scatterer="Cl"):
        """One descriptor row with only what the report reads."""
        return {"absorber": absorber, "scatterer": scatterer, "N": 5.75,
                "R": 2.80, "R_eff": 2.785, "sigma2": 0.016, "C3": 0.0,
                "skew": skew, "kurt": kurt, "mfp": DEFAULT_MFP,
                "n_structures": 4}

    def test_symmetric_flat_shell_is_warned(self):
        """The a-CaZrCl6 Ca-Cl case: skew -0.04 passes the skew test, but
        kurtosis -0.83 makes it flat-topped -- and its fit failed."""
        out = format_exafs_report({"Ca-Cl": self._entry(-0.04, -0.83)})
        assert "Asymmetric shell" not in out
        assert "Symmetric but non-Gaussian shell(s): Ca-Cl (kurtosis -0.83)" \
            in out
        assert "R-factor" in out

    def test_heavy_tailed_symmetric_shell_is_warned_too(self):
        out = format_exafs_report({"Ca-Cl": self._entry(0.1, +1.4)})
        assert "Ca-Cl (kurtosis +1.40)" in out

    def test_near_gaussian_symmetric_shell_is_not_warned(self):
        out = format_exafs_report(
            {"Ca-Cl": self._entry(0.05, 0.9 * KURT_THRESHOLD)})
        assert "non-Gaussian" not in out

    def test_skewed_shell_is_left_to_the_skew_warning(self):
        """A skewed shell with large kurtosis is already flagged once; the
        symmetric-shell line is only for what the skew test misses."""
        out = format_exafs_report({"Zr-Cl": self._entry(1.26, 1.63, "Zr")})
        assert "Asymmetric shell(s): Zr-Cl" in out
        assert "non-Gaussian" not in out

    def test_result_without_kurt_is_not_warned(self):
        """Results saved before C4 existed must still render."""
        entry = self._entry(0.0, 0.0)
        del entry["kurt"]
        assert "non-Gaussian" not in format_exafs_report({"Ca-Cl": entry})


class TestCSV:
    def test_csv_round_trip(self, tmp_path):
        atoms = _shell([1.5, 1.5, 1.7, 1.7])
        res = compute_exafs_descriptors([atoms], 2.0, _flat_cutoff(2.0))
        path = tmp_path / "exafs.csv"
        write_exafs_csv(res, str(path))

        lines = path.read_text().strip().split("\n")
        header = lines[0].split(",")
        assert header[0] == "pair"
        for col in ("N", "R", "R_eff", "sigma2", "C3", "skew", "C4", "kurt"):
            assert col in header
        assert len(lines) == 1 + len(res)

        row = dict(zip(header, lines[1].split(",")))
        assert row["pair"] in res
        assert float(row["R"]) == pytest.approx(res[row["pair"]]["R"])
        assert float(row["kurt"]) == pytest.approx(res[row["pair"]]["kurt"],
                                                   rel=1e-5)


# ─── reference-YAML validation ────────────────────────────────────────────

class _StubAnalyser:
    """Stand-in exposing only what validate_against_reference calls."""

    def exafs_descriptors(self):
        return {
            "Ga-O": {"absorber": "Ga", "scatterer": "O", "N": 4.42,
                     "R": 1.895, "R_eff": 1.880, "sigma2": 0.0062,
                     "C3": 1.2e-4, "skew": 0.25},
        }


class TestValidateExafsSection:
    def test_all_three_metrics_produce_rows(self):
        from amorphgen.analysis.validate import validate_against_reference
        ref = {"exafs": {"Ga-O": {"N_expected": [4.0, 4.8],
                                  "R_expected": [1.86, 1.90],
                                  "sigma2_expected": [0.004, 0.008]}}}
        rows = validate_against_reference(_StubAnalyser(), ref)["rows"]
        assert [r[0] for r in rows] == ["EXAFS N Ga-O", "EXAFS R_eff Ga-O",
                                        "EXAFS sigma2 Ga-O"]
        assert all(r[5] == "match" for r in rows)

    def test_r_compared_against_effective_by_default(self):
        from amorphgen.analysis.validate import validate_against_reference
        ref = {"exafs": {"Ga-O": {"R_expected": [1.86, 1.90]}}}
        row = validate_against_reference(_StubAnalyser(), ref)["rows"][0]
        assert row[0] == "EXAFS R_eff Ga-O"
        assert row[1] == pytest.approx(1.880)

    def test_real_convention_compares_against_plain_mean(self):
        from amorphgen.analysis.validate import validate_against_reference
        ref = {"exafs": {"r_convention": "real",
                         "Ga-O": {"R_expected": [1.86, 1.90]}}}
        row = validate_against_reference(_StubAnalyser(), ref)["rows"][0]
        assert row[0] == "EXAFS R Ga-O"
        assert row[1] == pytest.approx(1.895)

    def test_bad_convention_raises(self):
        from amorphgen.analysis.validate import validate_against_reference
        ref = {"exafs": {"r_convention": "nonsense",
                         "Ga-O": {"R_expected": [1.8, 1.9]}}}
        with pytest.raises(ValueError, match="r_convention"):
            validate_against_reference(_StubAnalyser(), ref)

    def test_unknown_pair_reported_as_na_not_skipped(self):
        # An absorber the ensemble does not contain must be visible in the
        # table, not dropped (a missing row reads as a pass). The pair is
        # *ordered*, so there is no reversed-key fallback here.
        from amorphgen.analysis.validate import validate_against_reference
        ref = {"exafs": {"In-O": {"N_expected": [4.0, 6.0]}}}
        with pytest.warns(UserWarning, match="matched nothing"):
            rows = validate_against_reference(_StubAnalyser(), ref)["rows"]
        assert len(rows) == 1
        assert rows[0][0] == "EXAFS N In-O"
        assert rows[0][1] is None
        assert rows[0][5] == "n/a"

    def test_missing_metric_produces_no_row(self):
        from amorphgen.analysis.validate import validate_against_reference
        ref = {"exafs": {"Ga-O": {"N_expected": [4.0, 4.8]}}}
        rows = validate_against_reference(_StubAnalyser(), ref)["rows"]
        assert len(rows) == 1

    def test_sigma2_renders_without_losing_precision(self):
        """A .3f table would print every sigma^2 as 0.006 or 0.005."""
        from amorphgen.analysis.validate import (validate_against_reference,
                                                 format_validation_report)
        ref = {"exafs": {"Ga-O": {"sigma2_expected": [0.004, 0.008]}}}
        out = format_validation_report(
            validate_against_reference(_StubAnalyser(), ref))
        assert "6.200e-03" in out
        assert "4.000e-03" in out


# ─── CLI ──────────────────────────────────────────────────────────────────

class TestCLI:
    @pytest.fixture
    def structure_dir(self, tmp_path):
        from ase.io import write
        base = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
        d = tmp_path / "structs"
        d.mkdir()
        for seed in range(2):
            at = base.copy()
            at.positions += np.random.default_rng(seed).normal(
                0, 0.05, at.positions.shape)
            write(str(d / f"s_{seed}.xyz"), at, format="extxyz")
        return d

    def test_exafs_flag_prints_table(self, structure_dir, tmp_path, capsys):
        from unittest.mock import patch
        from amorphgen.cli import main
        argv = ["amorphgen", "--analyse", "--input-dir", str(structure_dir),
                "--exafs", "--save-plot", str(tmp_path / "plots")]
        with patch("sys.argv", argv):
            main()
        out = capsys.readouterr().out
        assert "EXAFS first-shell descriptors" in out
        assert (tmp_path / "plots" / "analysis_exafs.csv").exists()

    def test_exafs_named_pair(self, structure_dir, capsys):
        from unittest.mock import patch
        from amorphgen.cli import main
        argv = ["amorphgen", "--analyse", "--input-dir", str(structure_dir),
                "--exafs", "Si-Si"]
        with patch("sys.argv", argv):
            main()
        assert "Si-Si" in capsys.readouterr().out

    def test_exafs_off_by_default(self, structure_dir, capsys):
        from unittest.mock import patch
        from amorphgen.cli import main
        argv = ["amorphgen", "--analyse", "--input-dir", str(structure_dir)]
        with patch("sys.argv", argv):
            main()
        assert "EXAFS first-shell descriptors" not in capsys.readouterr().out


class TestBootstrap:
    """Ensemble-sampling error bars, by resampling whole structures.

    Added after a real comparison of two a-SiO2 ensembles: the quench-rate
    differences I reported as results were all within about one standard
    error, and nothing in the output said so.
    """

    def _ensemble(self, n, sigma=0.06, seed0=0):
        base = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
        out = []
        for s in range(n):
            at = base.copy()
            at.positions += np.random.default_rng(seed0 + s).normal(
                0, sigma, at.positions.shape)
            out.append(at)
        return out

    def _desc(self, frames, **kw):
        from amorphgen.analysis import StructureAnalyser
        return StructureAnalyser(frames, cutoff=2.9).exafs_descriptors(**kw)

    def test_errors_present_by_default(self):
        d = self._desc(self._ensemble(4))["Si-Si"]
        for key in ("N_err", "R_err", "sigma2_err", "C3_err", "skew_err"):
            assert key in d
        assert d["R_err"] > 0 and d["sigma2_err"] > 0
        assert d["n_boot"] == DEFAULT_BOOTSTRAP

    def test_exactly_coordinated_shell_has_zero_n_error(self):
        """Rattled diamond Si keeps exactly 4 neighbours in every
        structure, so N has no ensemble spread. Zero is the right answer,
        not a missing key."""
        d = self._desc(self._ensemble(4))["Si-Si"]
        assert d["N"] == pytest.approx(4.0)
        assert d["N_err"] == pytest.approx(0.0, abs=1e-15)

    def test_bootstrap_zero_disables(self):
        d = self._desc(self._ensemble(4), bootstrap=0)["Si-Si"]
        assert "N_err" not in d and "n_boot" not in d

    def test_single_structure_has_no_error(self):
        """One structure means nothing to resample — silence beats a zero."""
        d = self._desc(self._ensemble(1))["Si-Si"]
        assert "N_err" not in d

    def test_identical_structures_give_zero_error(self):
        """The decisive test of resampling STRUCTURES rather than paths.

        Every structure is the same, so every resample is the same and the
        spread must vanish. Resampling individual paths would instead
        report a large spurious error.
        """
        base = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
        base.positions += np.random.default_rng(0).normal(
            0, 0.06, base.positions.shape)
        d = self._desc([base.copy() for _ in range(5)])["Si-Si"]
        assert d["sigma2_err"] == pytest.approx(0.0, abs=1e-15)
        assert d["R_err"] == pytest.approx(0.0, abs=1e-15)
        assert d["sigma2"] > 0          # the shell itself is disordered

    def test_reproducible_for_a_given_seed(self):
        frames = self._ensemble(5)
        a = self._desc(frames, seed=7)["Si-Si"]
        b = self._desc(frames, seed=7)["Si-Si"]
        assert a["sigma2_err"] == b["sigma2_err"]

    def test_seed_changes_the_estimate(self):
        frames = self._ensemble(5)
        a = self._desc(frames, seed=1)["Si-Si"]
        b = self._desc(frames, seed=2)["Si-Si"]
        assert a["sigma2_err"] != b["sigma2_err"]

    def test_error_shrinks_with_more_structures(self):
        """More structures, better-determined mean — the point of the bar."""
        few = self._desc(self._ensemble(3, seed0=0))["Si-Si"]
        many = self._desc(self._ensemble(24, seed0=0))["Si-Si"]
        assert many["R_err"] < few["R_err"]

    def test_r_eff_error_propagated(self):
        d = self._desc(self._ensemble(4))["Si-Si"]
        assert "R_eff_err" in d and d["R_eff_err"] > 0

    def test_report_shows_the_error_row(self):
        out = format_exafs_report(self._desc(self._ensemble(4)))
        assert "+/-" in out
        assert "bootstrap resamples" in out

    def test_report_warns_when_too_few_structures(self):
        """A bootstrap over 3 items cannot reach its own tails."""
        out = format_exafs_report(self._desc(self._ensemble(3)))
        assert "UNDERESTIMATE" in out
        assert "Only 3 structures" in out

    def test_report_quiet_when_enough_structures(self):
        out = format_exafs_report(self._desc(self._ensemble(10)))
        assert "+/-" in out
        assert "UNDERESTIMATE" not in out

    def test_csv_carries_the_errors(self, tmp_path):
        res = self._desc(self._ensemble(4))
        path = tmp_path / "exafs.csv"
        write_exafs_csv(res, str(path))
        header = path.read_text().splitlines()[0].split(",")
        for col in ("N_err", "R_err", "sigma2_err", "n_boot"):
            assert col in header

    def test_csv_omits_errors_when_disabled(self, tmp_path):
        res = self._desc(self._ensemble(4), bootstrap=0)
        path = tmp_path / "exafs.csv"
        write_exafs_csv(res, str(path))
        assert "N_err" not in path.read_text().splitlines()[0]


class TestAnionAnionPairs:
    """Hetero-anion pairs are contacts, not EXAFS paths.

    Found while checking AmorphGen against a Na-Nb-O-Cl oxychloride: the
    package's "ionic, or hetero-covalent" rule excludes O-O and Cl-Cl
    because the symbols match, but lets O-Cl through as a bond. Both are
    anions around Nb and Na and are never bonded to each other, so the
    pair is not an EXAFS path at all.
    """

    @pytest.mark.parametrize("a,b", [
        ("O", "Cl"),     # oxychloride
        ("O", "N"),      # oxynitride
        ("O", "F"),      # oxyfluoride
        ("Cl", "Br"),    # mixed halide
    ])
    def test_anion_pairs_are_contacts(self, a, b):
        from amorphgen.analysis.exafs import is_bonded_pair
        assert not is_bonded_pair(a, b)
        assert not is_bonded_pair(b, a)

    @pytest.mark.parametrize("a,b", [
        ("Si", "O"),     # silicate
        ("P", "O"),      # phosphate
        ("S", "O"),      # sulfate
        ("C", "O"),      # carbonate
        ("B", "O"),      # borate
        ("P", "S"),      # thiophosphate
        ("P", "N"),      # LiPON
    ])
    def test_network_formers_still_bond_to_their_anions(self, a, b):
        """The fix must not demote a polyanion former's own bonds."""
        from amorphgen.analysis.exafs import is_bonded_pair
        assert is_bonded_pair(a, b)
        assert is_bonded_pair(b, a)

    @pytest.mark.parametrize("a,b", [("Zr", "Cl"), ("Ca", "Cl"),
                                     ("Nb", "O"), ("Na", "Cl")])
    def test_cation_anion_pairs_are_bonds(self, a, b):
        from amorphgen.analysis.exafs import is_bonded_pair
        assert is_bonded_pair(a, b)

    def test_homonuclear_pairs_remain_contacts(self):
        from amorphgen.analysis.exafs import is_bonded_pair
        for sym in ("O", "Cl", "Si", "Na"):
            assert not is_bonded_pair(sym, sym)

    def test_report_drops_the_anion_pair(self):
        """End to end: O-Cl must not appear as an edge path."""
        from ase import Atoms
        rng = np.random.default_rng(0)
        cell = 12.0
        at = Atoms("Nb4O8Cl12",
                   positions=rng.uniform(0, cell, (24, 3)),
                   cell=[cell] * 3, pbc=True)
        from amorphgen.analysis import StructureAnalyser
        sa = StructureAnalyser([at], cutoff=3.0)
        out = format_exafs_report(sa.exafs_descriptors(bootstrap=0))
        assert "O-Cl" not in out and "Cl-O" not in out
        assert "Nb-O" in out or "Nb-Cl" in out
