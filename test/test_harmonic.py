"""Tests for amorphgen.analysis.harmonic (harmonic-fit bias).

Tier 1 — no calculator, no FEFF. The fit is checked against closed-form
cases: a delta shell, a Gaussian shell (whose answer the tier-1 cumulant
correction independently predicts), and skewed shells whose bias has a
known sign and a known dependence on the fit range.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk
from scipy.stats import norm, skewnorm, uniform

from amorphgen.analysis.exafs import DEFAULT_MFP
from amorphgen.analysis.harmonic import (
    DEFAULT_KMAX,
    DEFAULT_KMIN,
    R_FACTOR_FAILED,
    R_FACTOR_POOR,
    backscattering_amplitude,
    fit_quality_flags,
    effective_amplitude,
    fit_harmonic,
    fit_quality,
    format_harmonic_report,
    harmonic_amplitude,
    write_harmonic_csv,
)

#: Stratified quantiles. Sampling a distribution at evenly spaced
#: quantiles reproduces its moments far more accurately than a random
#: draw of the same size, so the tests below compare against the exact
#: target values rather than against a sampling tolerance.
Q = (np.arange(8000) + 0.5) / 8000

N_PER_ABSORBER = 4.0


def _gaussian_shell(r0, sd):
    return norm.ppf(Q, r0, sd)


def _skewed_shell(r0, sd, shape):
    """A shell with the given mean and width but a deliberate skew."""
    raw = skewnorm.ppf(Q, shape)
    return r0 + sd * (raw - raw.mean()) / raw.std()


def _cumulants(d):
    c1 = float(d.mean())
    delta = d - c1
    return c1, float((delta ** 2).mean()), float((delta ** 3).mean())


def _fit(d, **kw):
    return fit_harmonic(d, len(d) / N_PER_ABSORBER, **kw)


# ─── amplitude functions ──────────────────────────────────────────────────

class TestAmplitudes:
    def test_single_distance_is_the_closed_form(self):
        k = np.linspace(3.0, 12.0, 50)
        got = effective_amplitude(k, [1.85], n_absorbers=1.0)
        want = 1.85 ** -2 * np.exp(-2 * 1.85 / DEFAULT_MFP) * np.exp(
            2j * k * 1.85)
        assert np.allclose(got, want)

    def test_harmonic_with_zero_width_is_a_single_distance(self):
        """sigma^2 = 0 collapses the Gaussian model onto one distance."""
        k = np.linspace(3.0, 12.0, 50)
        assert np.allclose(harmonic_amplitude(k, 1.0, 1.85, 0.0),
                           effective_amplitude(k, [1.85], n_absorbers=1.0))

    def test_amplitude_scales_with_coordination(self):
        k = np.linspace(3.0, 12.0, 20)
        one = effective_amplitude(k, [1.85], n_absorbers=1.0)
        four = effective_amplitude(k, [1.85] * 4, n_absorbers=1.0)
        assert np.allclose(four, 4.0 * one)

    def test_empty_shell_is_zero(self):
        k = np.linspace(3.0, 12.0, 20)
        assert np.allclose(effective_amplitude(k, [], n_absorbers=1.0), 0.0)

    def test_e0_shift_changes_the_phase(self):
        k = np.linspace(3.0, 12.0, 20)
        base = harmonic_amplitude(k, 4.0, 1.85, 0.005)
        shifted = harmonic_amplitude(k, 4.0, 1.85, 0.005, e0_shift=5.0)
        assert not np.allclose(np.angle(base), np.angle(shifted))


# ─── closed-form shells ───────────────────────────────────────────────────

class TestDeltaShell:
    """Every scatterer at exactly the same distance: nothing to bias."""

    @pytest.fixture
    def fit(self):
        return _fit(np.full(4000, 1.85))

    def test_recovers_n_exactly(self, fit):
        assert fit["N"] == pytest.approx(N_PER_ABSORBER, rel=1e-6)

    def test_recovers_r_exactly(self, fit):
        assert fit["R"] == pytest.approx(1.85, abs=1e-6)

    def test_zero_disorder(self, fit):
        assert fit["sigma2"] == pytest.approx(0.0, abs=1e-8)

    def test_perfect_fit(self, fit):
        assert fit["r_factor"] < 1e-12


class TestGaussianShell:
    """A Gaussian shell is exactly what the harmonic model assumes, so the
    fit should be near-perfect — and its R must agree with the first-order
    cumulant correction that exafs.py derives independently."""

    @pytest.mark.parametrize("r0,sd", [(1.85, 0.07), (2.10, 0.09)])
    def test_r_matches_the_tier1_effective_distance(self, r0, sd):
        d = _gaussian_shell(r0, sd)
        c1, c2, _ = _cumulants(d)
        r_eff = c1 - 2 * c2 * (1 / c1 + 1 / DEFAULT_MFP)
        assert _fit(d)["R"] == pytest.approx(r_eff, abs=1e-3)

    def test_fit_is_essentially_perfect(self):
        assert _fit(_gaussian_shell(1.85, 0.07))["r_factor"] < 1e-6

    def test_sigma2_recovered(self):
        d = _gaussian_shell(1.85, 0.07)
        _, c2, _ = _cumulants(d)
        assert _fit(d)["sigma2"] == pytest.approx(c2, rel=0.02)

    def test_small_intrinsic_n_offset(self):
        """Even a symmetric shell loses ~0.5% of N: the harmonic model
        evaluates R^-2 exp(-2R/lambda) at the shell distance instead of
        carrying it inside the integral. Documented, not a bug."""
        n_fit = _fit(_gaussian_shell(1.85, 0.07))["N"]
        assert n_fit < N_PER_ABSORBER
        assert n_fit == pytest.approx(N_PER_ABSORBER, rel=0.02)


class TestSkewedShell:
    """An asymmetric shell is where the harmonic model breaks, and the
    whole point of the module."""

    @pytest.mark.parametrize("shape", [3.0, 6.0])
    def test_positive_skew_undercounts_n(self, shape):
        d = _skewed_shell(1.90, 0.08, shape)
        assert _fit(d)["N"] < 0.95 * N_PER_ABSORBER

    @pytest.mark.parametrize("shape", [-3.0, -6.0])
    def test_negative_skew_also_undercounts_n(self, shape):
        """The amplitude loss comes from the spread of the phase, which
        does not care about the sign of C3 — only R moves the other way."""
        d = _skewed_shell(1.90, 0.08, shape)
        assert _fit(d)["N"] < 0.95 * N_PER_ABSORBER

    def test_r_shift_follows_the_sign_of_c3(self):
        pos = _skewed_shell(1.90, 0.08, 6.0)
        neg = _skewed_shell(1.90, 0.08, -6.0)
        assert _cumulants(pos)[2] > 0 and _cumulants(neg)[2] < 0
        assert _fit(pos)["R"] < _cumulants(pos)[0]      # tail to long R
        assert _fit(neg)["R"] > _cumulants(neg)[0]      # tail to short R

    def test_sigma2_underestimated(self):
        d = _skewed_shell(1.90, 0.08, 6.0)
        _, c2, _ = _cumulants(d)
        assert _fit(d)["sigma2"] < c2

    def test_misfit_far_worse_than_gaussian(self):
        skewed = _fit(_skewed_shell(1.90, 0.08, 6.0))["r_factor"]
        gaussian = _fit(_gaussian_shell(1.90, 0.08))["r_factor"]
        assert skewed > 1000 * gaussian

    def test_bias_grows_with_kmax(self):
        """The C3 phase term grows as k^3, so a wider window costs more N.
        This is why a comparison must use the paper's own k-range."""
        d = _skewed_shell(1.90, 0.08, 6.0)
        n = [_fit(d, kmax=km)["N"] for km in (8.0, 10.0, 12.0, 14.0, 16.0)]
        assert np.all(np.diff(n) < 0)


class TestFloatedE0:
    """Floating E0 flatters the fit without fixing the coordination number
    — worth pinning, because it is the natural thing to try."""

    @pytest.fixture
    def shell(self):
        return _skewed_shell(1.90, 0.08, 6.0)

    def test_improves_the_misfit(self, shell):
        assert (_fit(shell, fit_e0=True)["r_factor"]
                < 0.5 * _fit(shell, fit_e0=False)["r_factor"])

    def test_leaves_n_essentially_unchanged(self, shell):
        assert (_fit(shell, fit_e0=True)["N"]
                == pytest.approx(_fit(shell, fit_e0=False)["N"], rel=0.02))

    def test_reports_a_nonzero_shift(self, shell):
        assert abs(_fit(shell, fit_e0=True)["e0_shift"]) > 1.0

    def test_fixed_e0_reports_zero(self, shell):
        assert _fit(shell, fit_e0=False)["e0_shift"] == 0.0


class TestFitInvariants:
    def test_doubling_paths_doubles_n_only(self):
        d = _skewed_shell(1.90, 0.08, 6.0)
        one = fit_harmonic(d, len(d) / N_PER_ABSORBER)
        two = fit_harmonic(np.concatenate([d, d]), len(d) / N_PER_ABSORBER)
        assert two["N"] == pytest.approx(2 * one["N"], rel=1e-6)
        assert two["R"] == pytest.approx(one["R"], rel=1e-9)
        assert two["sigma2"] == pytest.approx(one["sigma2"], rel=1e-6)

    def test_empty_shell_raises(self):
        with pytest.raises(ValueError, match="empty shell"):
            fit_harmonic([], 1.0)

    def test_settings_echoed(self):
        f = _fit(_gaussian_shell(1.85, 0.07), kmin=2.5, kmax=14.0, kweight=3)
        assert (f["kmin"], f["kmax"], f["kweight"]) == (2.5, 14.0, 3)
        assert f["mfp"] == DEFAULT_MFP
        assert f["converged"]

    def test_defaults(self):
        f = _fit(_gaussian_shell(1.85, 0.07))
        assert (f["kmin"], f["kmax"]) == (DEFAULT_KMIN, DEFAULT_KMAX)


# ─── fit_shells over a real ensemble ──────────────────────────────────────

class TestFitShells:
    @pytest.fixture
    def rattled_si(self):
        base = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
        out = []
        for seed in range(3):
            at = base.copy()
            at.positions += np.random.default_rng(seed).normal(
                0, 0.06, at.positions.shape)
            out.append(at)
        return out

    @pytest.fixture
    def analyser(self, rattled_si):
        from amorphgen.analysis import StructureAnalyser
        return StructureAnalyser(rattled_si, cutoff=2.9)

    # These exercise the fit itself, not the error bars, so they turn the
    # bootstrap off: 50 extra least-squares fits per assertion is a real
    # cost in CI and buys nothing here. TestFitBootstrap covers the bars.
    BOOT = {"bootstrap": 0}

    def test_geometric_column_matches_the_descriptor_table(self, analyser):
        """N_geom, R_geom and sigma2_geom must be the same numbers
        exafs_descriptors() reports — one is the baseline for the other."""
        ex = analyser.exafs_descriptors()["Si-Si"]
        hf = analyser.harmonic_fit(**self.BOOT)["Si-Si"]
        assert hf["N_geom"] == pytest.approx(ex["N"])
        assert hf["R_geom"] == pytest.approx(ex["R"])
        assert hf["sigma2_geom"] == pytest.approx(ex["sigma2"])
        assert hf["skew"] == pytest.approx(ex["skew"])

    def test_n_geom_matches_coordination(self, analyser):
        assert (analyser.harmonic_fit(**self.BOOT)["Si-Si"]["N_geom"]
                == pytest.approx(analyser.coordination()["Si-Si"]["mean"]))

    def test_bias_fields_are_consistent(self, analyser):
        d = analyser.harmonic_fit(**self.BOOT)["Si-Si"]
        assert d["N_bias"] == pytest.approx(d["N_fit"] - d["N_geom"])
        assert d["N_bias_percent"] == pytest.approx(
            100 * d["N_bias"] / d["N_geom"])
        assert d["R_bias"] == pytest.approx(d["R_fit"] - d["R_geom"])

    def test_pair_filter(self, analyser):
        assert list(analyser.harmonic_fit(pair="Si-Si", **self.BOOT)) == ["Si-Si"]

    def test_settings_passed_through(self, analyser):
        d = analyser.harmonic_fit(kmax=14.0, fit_e0=True, **self.BOOT)["Si-Si"]
        assert d["kmax"] == 14.0
        assert d["fit_e0"] is True

    def test_empty_pair_omitted(self):
        from amorphgen.analysis import StructureAnalyser
        from ase import Atoms
        far = Atoms(symbols=["Si", "O"], positions=[[1, 1, 1], [8, 8, 8]],
                    cell=[12] * 3, pbc=False)
        assert StructureAnalyser([far], cutoff=2.0).harmonic_fit(bootstrap=0) == {}


# ─── report and CSV ───────────────────────────────────────────────────────

class TestReport:
    @pytest.fixture
    def result(self):
        from amorphgen.analysis import StructureAnalyser
        base = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
        frames = []
        for seed in range(2):
            at = base.copy()
            at.positions += np.random.default_rng(seed).normal(
                0, 0.06, at.positions.shape)
            frames.append(at)
        return StructureAnalyser(frames, cutoff=2.9).harmonic_fit(bootstrap=0)

    def test_shows_both_columns(self, result):
        out = format_harmonic_report(result)
        assert "geometric" in out and "harmonic fit" in out
        assert "sigma2 (A^2)" in out

    def test_states_the_fit_window(self, result):
        assert "Fit window k = 3.0-12.0 A^-1" in format_harmonic_report(result)

    def test_says_which_column_to_compare_against(self, result):
        """The headline point: a published N belongs against the fit."""
        out = format_harmonic_report(result)
        assert "Compare a published EXAFS N against the 'harmonic fit'" in out

    def test_warns_that_the_bias_depends_on_kmax(self, result):
        assert "bias grows with kmax" in format_harmonic_report(result)

    def test_e0_row_only_when_floated(self):
        from amorphgen.analysis import StructureAnalyser
        base = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
        at = base.copy()
        at.positions += np.random.default_rng(0).normal(
            0, 0.06, at.positions.shape)
        sa = StructureAnalyser([at], cutoff=2.9)
        assert "dE0 (eV)" not in format_harmonic_report(
            sa.harmonic_fit(bootstrap=0))
        assert "dE0 (eV)" in format_harmonic_report(
            sa.harmonic_fit(fit_e0=True, bootstrap=0))

    def test_empty_result_message(self):
        assert "No harmonic fits" in format_harmonic_report({})

    def test_csv_round_trip(self, result, tmp_path):
        path = tmp_path / "harmonic.csv"
        write_harmonic_csv(result, str(path))
        lines = path.read_text().strip().split("\n")
        header = lines[0].split(",")
        for col in ("N_geom", "N_fit", "N_bias_percent", "R_fit", "skew",
                    "r_factor"):
            assert col in header
        assert len(lines) == 1 + len(result)
        row = dict(zip(header, lines[1].split(",")))
        assert row["pair"] in result
        assert float(row["N_geom"]) == pytest.approx(
            result[row["pair"]]["N_geom"], rel=1e-5)


# ─── |f(k)| weighting: what a real chi(k) fit weights the misfit by ───────

class TestBackscatterTable:
    K = np.linspace(3.0, 12.0, 50)

    def test_light_and_heavy_scatterers_are_tabulated(self):
        for el in ("Li", "O", "Cl", "Zr", "In", "Bi"):
            amp = backscattering_amplitude(el, self.K)
            assert amp is not None and np.all(amp > 0)

    def test_untabulated_element_gives_none(self):
        assert backscattering_amplitude("H", self.K) is None
        assert backscattering_amplitude("U", self.K) is None

    def test_oxygen_falls_more_than_tenfold_across_the_window(self):
        """The reason the weighting matters at all for oxides."""
        amp = backscattering_amplitude("O", self.K)
        assert amp[0] / amp[-1] > 10

    def test_heavy_scatterer_is_flatter_than_a_light_one(self):
        o = backscattering_amplitude("O", self.K)
        pb = backscattering_amplitude("Pb", self.K)
        assert pb[0] / pb[-1] < o[0] / o[-1] / 3


class TestAmplitudeWeighting:
    def test_gaussian_shell_is_indifferent_to_the_weighting(self):
        """A correct model fits at any weighting -- which is why tests on
        Gaussian shells could never have caught the missing |f(k)|."""
        d = _gaussian_shell(1.95, 0.08)
        bare, weighted = _fit(d), _fit(d, scatterer="O")
        assert weighted["N"] == pytest.approx(bare["N"], rel=0.01)
        assert weighted["amplitude_weighted"] and not bare["amplitude_weighted"]

    def test_weighting_shrinks_the_bias_of_a_skewed_oxide_shell(self):
        """|f(k)| of O moves weight to low k, where an asymmetric shell
        loses the least amplitude: a-Ga2O3 went from -22% to -12%,
        matching Larch fitting FEFF chi(k) (docs/notes/exafs_crosscheck)."""
        d = _skewed_shell(1.93, 0.10, 6.0)
        bare = _fit(d)["N"] - N_PER_ABSORBER
        weighted = _fit(d, scatterer="O")["N"] - N_PER_ABSORBER
        assert bare < weighted < 0

    def test_untabulated_scatterer_warns_and_falls_back(self):
        d = _gaussian_shell(1.0, 0.05)
        with pytest.warns(UserWarning, match="no backscattering amplitude"):
            f = _fit(d, scatterer="H")
        assert f["amplitude_weighted"] is False
        assert f["scatterer"] == "H"

    def test_fit_shells_weights_by_default_and_can_be_turned_off(self):
        from amorphgen.analysis import StructureAnalyser
        at = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
        at.positions += np.random.default_rng(0).normal(0, 0.06,
                                                        at.positions.shape)
        sa = StructureAnalyser([at], cutoff=2.9)
        assert sa.harmonic_fit(bootstrap=0)["Si-Si"]["amplitude_weighted"]
        assert not sa.harmonic_fit(bootstrap=0, backscatter=False)[
            "Si-Si"]["amplitude_weighted"]

    def test_report_says_how_the_residual_was_weighted(self):
        weighted = _report_row("Zr", "Cl", 5.49, 4.99, 0.004)
        bare = dict(weighted, amplitude_weighted=False)
        assert "Residual weighted by k^(w-1) |f(k)|" in \
            format_harmonic_report({"Zr-Cl": weighted})
        assert "NOT weighted by |f(k)|: Zr-Cl" not in \
            format_harmonic_report({"Zr-Cl": weighted})
        assert "NOT weighted by |f(k)|: Zr-Cl" in \
            format_harmonic_report({"Zr-Cl": bare})

    def test_csv_records_the_weighting(self, tmp_path):
        path = tmp_path / "h.csv"
        write_harmonic_csv({"Zr-Cl": _report_row("Zr", "Cl", 5.49, 4.99,
                                                 0.004)}, str(path))
        header, row = path.read_text().strip().split("\n")
        assert dict(zip(header.split(","), row.split(",")))[
            "amplitude_weighted"] == "true"


# ─── fit quality: when a fit means nothing ────────────────────────────────

class TestFitQuality:
    @pytest.mark.parametrize("r_factor, expected", [
        (0.003, "good"),              # a-Ga2O3 Ga-O, weighted residual
        (R_FACTOR_POOR, "good"),      # bands are inclusive at the top
        (0.05, "poor"),
        (R_FACTOR_FAILED, "poor"),
        (0.81, "failed"),             # a-CaZrCl6 Ca-Cl, bare residual
    ])
    def test_r_factor_bands(self, r_factor, expected):
        assert fit_quality(r_factor) == expected

    # Real shells, |f(k)|-weighted residual, against Larch fits of FEFF
    # chi(k) for the same sites (docs/notes/exafs_crosscheck).
    SHELLS = {   # r_factor, skew, kurt, sigma2_geom
        "Ga2O3 Ga-O": (0.003, 1.17, 2.12, 0.00795),    # matched to ~1 point
        "IGZO In-O": (0.009, 1.31, 2.27, 0.01224),
        "TiO2 Ti-O": (0.017, 1.03, 1.23, 0.02066),     # off by 4-8 points
        "CaZrCl6 Zr-Cl": (0.011, 1.26, 1.63, 0.01477),  # off by ~4 points
        "CaZrCl6 Ca-Cl": (0.048, -0.04, -0.83, 0.01599),  # off by ~30
    }

    @pytest.mark.parametrize("shell, expected", [
        ("Ga2O3 Ga-O", "good"), ("IGZO In-O", "good"),
        ("TiO2 Ti-O", "poor"), ("CaZrCl6 Zr-Cl", "poor"),
        ("CaZrCl6 Ca-Cl", "failed"),
    ])
    def test_real_shells(self, shell, expected):
        rf, skew, kurt, s2 = self.SHELLS[shell]
        assert fit_quality(rf, skew=skew, kurt=kurt, sigma2=s2,
                           kmax=12.0) == expected

    def test_flat_shell_fails_on_shape_though_its_r_factor_looks_fine(self):
        """Weighting brought Ca-Cl's R-factor down to 0.02-0.05; only its
        shape and width say the fit means nothing."""
        flags = fit_quality_flags(0.012, skew=-0.04, kurt=-0.83,
                                  sigma2=0.016, kmax=12.0)
        assert ("failed", "non-gaussian") in flags

    def test_narrow_flat_shell_is_not_condemned(self):
        """The C4 correction scales as sigma^4 kmax^4: a narrow flat shell
        is non-Gaussian but harmlessly so."""
        assert fit_quality(0.001, skew=0.0, kurt=-1.2, sigma2=0.0033,
                           kmax=12.0) == "good"

    def test_broad_shell_depends_on_kmax(self):
        """Broad means little signal left *at kmax*: the same shell is fine
        over a shorter window."""
        shape = dict(skew=1.0, kurt=1.2, sigma2=0.0207)
        assert fit_quality(0.005, kmax=12.0, **shape) == "poor"
        assert fit_quality(0.005, kmax=8.0, **shape) == "good"

    def test_not_converged_is_failed_whatever_the_r_factor(self):
        assert fit_quality(0.001, converged=False) == "failed"

    def test_non_finite_r_factor_is_failed(self):
        assert fit_quality(float("nan")) == "failed"


class TestFlatToppedShell:
    """Symmetric is not Gaussian: a flat-topped shell has zero skew, so the
    skew test passes it, yet a harmonic fit of it fails outright.

    Reproduces a-CaZrCl6 Ca-Cl from first principles: same sigma^2
    (0.016 A^2), kurtosis -1.2 against the real shell's -0.83.
    """

    @pytest.fixture
    def flat(self):
        return uniform.ppf(Q, 2.8 - 0.22, 0.44)

    def test_its_n_depends_on_the_fitting_choices(self, flat):
        """Ill-conditioned: weighting the residual as a chi(k) fit does
        moves N by tens of percent for the flat shell, by nothing for a
        Gaussian of the same width. Larch fits of real Ca-Cl moved by as
        much across window and fitting-space choices."""
        gaussian = _gaussian_shell(2.8, float(flat.std()))
        for shell, spread in ((flat, 0.3), (gaussian, 0.02)):
            bare, weighted = _fit(shell)["N"], _fit(shell, scatterer="Cl")["N"]
            moved = abs(bare - weighted) / N_PER_ABSORBER
            assert (moved > spread) if shell is flat else (moved < spread)

    def test_the_bare_fit_runs_n_away(self, flat):
        assert _fit(flat)["N"] > 1.5 * N_PER_ABSORBER

    def test_it_is_classified_failed_from_its_own_moments(self, flat):
        c1 = flat.mean()
        c2 = ((flat - c1) ** 2).mean()
        skew = ((flat - c1) ** 3).mean() / c2 ** 1.5
        kurt = ((flat - c1) ** 4).mean() / c2 ** 2 - 3.0
        r = _fit(flat, scatterer="Cl")["r_factor"]
        assert fit_quality(r, skew=skew, kurt=kurt, sigma2=c2,
                           kmax=12.0) == "failed"

    def test_a_narrow_flat_shell_still_fits(self):
        narrow = uniform.ppf(Q, 2.8 - 0.10, 0.20)
        assert fit_quality(_fit(narrow, scatterer="Cl")["r_factor"]) == "good"


def _report_row(absorber, scatterer, n_geom, n_fit, r_factor, converged=True,
                with_quality=True):
    """One fit_shells entry carrying only what the report reads."""
    d = {"absorber": absorber, "scatterer": scatterer,
         "N_geom": n_geom, "N_fit": n_fit,
         "N_bias": n_fit - n_geom,
         "N_bias_percent": 100.0 * (n_fit - n_geom) / n_geom,
         "C3": 0.0, "R_geom": 2.0, "R_fit": 1.98, "R_bias": -0.02,
         "sigma2_geom": 0.01, "sigma2_fit": 0.008,
         "sigma2_bias_percent": -20.0, "skew": 0.5,
         "r_factor": r_factor, "converged": converged,
         "amplitude_weighted": True, "kmin": 3.0, "kmax": 12.0, "kweight": 2, "fit_e0": False,
         "e0_shift": 0.0, "mfp": DEFAULT_MFP, "n_structures": 10}
    d["kurt"] = 0.0
    if with_quality:
        d["fit_quality"] = fit_quality(r_factor, converged)
        d["quality_flags"] = ";".join(
            c for _, c in fit_quality_flags(r_factor, converged))
    return d


class TestReportQualityGate:
    """The headline quotes the pair furthest from its geometric N. Ranked
    naively, a failed fit wins *because* it is broken: a-CaZrCl6 reported
    "+97%" for Ca-Cl off an R-factor of 0.81."""

    def test_headline_skips_a_failed_fit(self):
        out = format_harmonic_report({
            "Ca-Cl": _report_row("Ca", "Cl", 5.75, 11.35, 0.81),
            "Zr-Cl": _report_row("Zr", "Cl", 5.49, 4.50, 0.015)})
        assert "For Zr-Cl a fit of this" in out
        assert "For Ca-Cl" not in out
        assert "FIT FAILED for Ca-Cl" in out
        assert "R-factor 8.1e-01, FIT FAILED: r-factor)" in out

    def test_good_fit_preferred_over_a_poor_one_with_larger_bias(self):
        out = format_harmonic_report({
            "Ti-O": _report_row("Ti", "O", 5.33, 3.06, 0.05),
            "Zr-Cl": _report_row("Zr", "Cl", 5.49, 4.50, 0.011)})
        assert "For Zr-Cl a fit of this" in out
        assert "R-factor 5.0e-02, POOR FIT: r-factor)" in out
        assert "That fit is POOR" not in out

    def test_a_poor_fit_headline_carries_a_caveat(self):
        out = format_harmonic_report(
            {"Ti-O": _report_row("Ti", "O", 5.33, 3.06, 0.05)})
        assert "For Ti-O a fit of this" in out
        assert "That fit is POOR (r-factor)" in out

    def test_flags_are_explained_once_with_their_pairs(self):
        broad = _report_row("Ti", "O", 5.39, 4.66, 0.005)
        broad.update(fit_quality="poor", quality_flags="broad")
        out = format_harmonic_report({"Ti-O": broad})
        assert "POOR FIT: broad)" in out
        assert "broad (Ti-O): broad shell" in out

    def test_nothing_usable_says_so_instead_of_quoting_a_number(self):
        out = format_harmonic_report(
            {"Ca-Cl": _report_row("Ca", "Cl", 5.75, 11.35, 0.81)})
        assert "No pair has a usable harmonic fit" in out
        assert "would report N =" not in out

    def test_unconverged_fit_is_flagged_and_excluded(self):
        out = format_harmonic_report({
            "Ca-Cl": _report_row("Ca", "Cl", 5.75, 2.0, 0.01, converged=False),
            "Zr-Cl": _report_row("Zr", "Cl", 5.49, 4.50, 0.015)})
        assert "FIT FAILED: not-converged" in out
        assert "For Zr-Cl a fit of this" in out

    def test_headline_prefers_the_cation_edge(self):
        """With |f(k)| weighting O-Ga is fitted against Ga's amplitude and
        Ga-O against O's, so their biases differ -- on a-Ga2O3 -15% and
        -8%. The O K edge is the larger and almost nobody measures it."""
        out = format_harmonic_report({
            "Ga-O": _report_row("Ga", "O", 4.381, 4.048, 0.004),
            "O-Ga": _report_row("O", "Ga", 2.921, 2.485, 0.005)})
        assert "For Ga-O a fit of this" in out

    def test_largest_bias_among_cation_edges(self):
        out = format_harmonic_report({
            "Ga-O": _report_row("Ga", "O", 4.35, 3.97, 0.005),
            "In-O": _report_row("In", "O", 5.20, 4.53, 0.009),
            "O-In": _report_row("O", "In", 1.30, 0.95, 0.009)})
        assert "For In-O a fit of this" in out

    def test_an_anion_edge_when_no_cation_edge_is_usable(self):
        out = format_harmonic_report({
            "Ga-O": _report_row("Ga", "O", 4.38, 3.66, 0.5),
            "O-Ga": _report_row("O", "Ga", 2.92, 2.48, 0.005)})
        assert "For O-Ga a fit of this" in out

    def test_results_without_fit_quality_are_still_gated(self):
        """Results built before the field existed classify off r_factor."""
        out = format_harmonic_report({
            "Ca-Cl": _report_row("Ca", "Cl", 5.75, 11.35, 0.81,
                                 with_quality=False),
            "Zr-Cl": _report_row("Zr", "Cl", 5.49, 4.50, 0.015,
                                 with_quality=False)})
        assert "For Zr-Cl a fit of this" in out
        assert "FIT FAILED for Ca-Cl" in out

    def test_fit_shells_records_the_quality(self):
        from amorphgen.analysis import StructureAnalyser
        at = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
        at.positions += np.random.default_rng(0).normal(0, 0.06,
                                                        at.positions.shape)
        res = StructureAnalyser([at], cutoff=2.9).harmonic_fit(bootstrap=0)
        d = res["Si-Si"]
        assert d["fit_quality"] == fit_quality(d["r_factor"], d["converged"])

    def test_csv_carries_the_quality(self, tmp_path):
        path = tmp_path / "h.csv"
        write_harmonic_csv(
            {"Ca-Cl": _report_row("Ca", "Cl", 5.75, 11.35, 0.81)}, str(path))
        header, row = path.read_text().strip().split("\n")
        assert dict(zip(header.split(","), row.split(",")))["fit_quality"] \
            == "failed"


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
                0, 0.06, at.positions.shape)
            write(str(d / f"s_{seed}.xyz"), at, format="extxyz")
        return d

    def _run(self, argv):
        from unittest.mock import patch
        from amorphgen.cli import main
        with patch("sys.argv", argv):
            main()

    def test_flag_prints_table_and_saves_csv(self, structure_dir, tmp_path,
                                             capsys):
        plots = tmp_path / "plots"
        self._run(["amorphgen", "--analyse", "--input-dir",
                   str(structure_dir), "--harmonic-fit",
                   "--save-plot", str(plots)])
        assert "Harmonic-fit bias" in capsys.readouterr().out
        assert (plots / "analysis_harmonic.csv").exists()

    def test_kmax_flag_changes_the_window(self, structure_dir, capsys):
        self._run(["amorphgen", "--analyse", "--input-dir",
                   str(structure_dir), "--harmonic-fit", "Si-Si",
                   "--harmonic-kmax", "16.0", "--harmonic-bootstrap", "0"])
        assert "Fit window k = 3.0-16.0 A^-1" in capsys.readouterr().out

    def test_float_e0_flag(self, structure_dir, capsys):
        self._run(["amorphgen", "--analyse", "--input-dir",
                   str(structure_dir), "--harmonic-fit",
                   "--harmonic-float-e0", "--harmonic-bootstrap", "0"])
        out = capsys.readouterr().out
        assert "E0 floated" in out
        assert "dE0 (eV)" in out

    def test_off_by_default(self, structure_dir, capsys):
        self._run(["amorphgen", "--analyse", "--input-dir",
                   str(structure_dir)])
        assert "Harmonic-fit bias" not in capsys.readouterr().out


class TestValidateAgainstTheFit:
    """A reference taken from a published fit should be scored against the
    fit, not against the geometry — that is the whole argument."""

    class _Stub:
        def exafs_descriptors(self):
            return {"Ga-O": {"N": 4.42, "R": 1.895, "R_eff": 1.880,
                             "sigma2": 0.0062}}

        def harmonic_fit(self):
            return {"Ga-O": {"N_fit": 3.98, "R_fit": 1.872,
                             "sigma2_fit": 0.0048}}

    def test_default_source_is_geometric(self):
        from amorphgen.analysis.validate import validate_against_reference
        ref = {"exafs": {"Ga-O": {"N_expected": [4.0, 4.8]}}}
        row = validate_against_reference(self._Stub(), ref)["rows"][0]
        assert row[0] == "EXAFS N Ga-O"
        assert row[1] == pytest.approx(4.42)

    def test_harmonic_fit_source_scores_the_fitted_values(self):
        from amorphgen.analysis.validate import validate_against_reference
        ref = {"exafs": {"source": "harmonic-fit",
                         "Ga-O": {"N_expected": [3.8, 4.2],
                                  "R_expected": [1.86, 1.89],
                                  "sigma2_expected": [0.004, 0.006]}}}
        rows = validate_against_reference(self._Stub(), ref)["rows"]
        assert [r[0] for r in rows] == ["EXAFS N(fit) Ga-O",
                                        "EXAFS R(fit) Ga-O",
                                        "EXAFS sigma2(fit) Ga-O"]
        assert rows[0][1] == pytest.approx(3.98)
        assert all(r[5] == "match" for r in rows)

    def test_the_two_sources_can_disagree(self):
        """N = 3.98 passes a [3.8, 4.2] reference; the geometric 4.42 does
        not. That gap is exactly what the harmonic fit exists to expose."""
        from amorphgen.analysis.validate import validate_against_reference
        band = {"N_expected": [3.8, 4.2]}
        geom = validate_against_reference(
            self._Stub(), {"exafs": {"Ga-O": band}})["rows"][0]
        fitted = validate_against_reference(
            self._Stub(),
            {"exafs": {"source": "harmonic-fit", "Ga-O": band}})["rows"][0]
        assert geom[5] == "fail"
        assert fitted[5] == "match"

    def test_bad_source_raises(self):
        from amorphgen.analysis.validate import validate_against_reference
        ref = {"exafs": {"source": "nonsense",
                         "Ga-O": {"N_expected": [3.8, 4.2]}}}
        with pytest.raises(ValueError, match="source must be"):
            validate_against_reference(self._Stub(), ref)

    @staticmethod
    def _fit_stub(r_factor, converged=True):
        class Stub:
            def harmonic_fit(self):
                return {"Ga-O": {"N_fit": 3.98, "R_fit": 1.872,
                                 "sigma2_fit": 0.0048, "r_factor": r_factor,
                                 "converged": converged,
                                 "fit_quality": fit_quality(r_factor,
                                                            converged)}}
        return Stub()

    REF = {"exafs": {"source": "harmonic-fit",
                     "Ga-O": {"N_expected": [3.8, 4.2]}}}

    def test_a_failed_fit_is_not_scored(self):
        """A verdict off a failed fit would judge the fitting, not the
        structure -- the a-CaZrCl6 Ca-Cl case."""
        from amorphgen.analysis.validate import validate_against_reference
        with pytest.warns(UserWarning, match="is failed"):
            rows = validate_against_reference(self._fit_stub(0.81),
                                              self.REF)["rows"]
        assert rows[0][1] is None and rows[0][5] == "n/a"

    def test_an_unconverged_fit_is_not_scored(self):
        from amorphgen.analysis.validate import validate_against_reference
        with pytest.warns(UserWarning, match="is failed"):
            rows = validate_against_reference(
                self._fit_stub(0.001, converged=False), self.REF)["rows"]
        assert rows[0][5] == "n/a"

    def test_a_poor_fit_is_scored_with_a_warning(self):
        from amorphgen.analysis.validate import validate_against_reference
        with pytest.warns(UserWarning, match="is poor.*indicative"):
            rows = validate_against_reference(self._fit_stub(0.05),
                                              self.REF)["rows"]
        assert rows[0][1] == pytest.approx(3.98)
        assert rows[0][5] == "match"

    def test_a_good_fit_is_scored_silently(self):
        import warnings
        from amorphgen.analysis.validate import validate_against_reference
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rows = validate_against_reference(self._fit_stub(0.005),
                                              self.REF)["rows"]
        assert rows[0][5] == "match"

    def test_unknown_pair_reported_as_na_not_skipped(self):
        # As on the geometric path: an absorber the ensemble does not
        # contain gets a visible "n/a" row and a warning, never a silent
        # drop (a metric missing from the table reads as a pass).
        from amorphgen.analysis.validate import validate_against_reference
        ref = {"exafs": {"source": "harmonic-fit",
                         "In-O": {"N_expected": [4.0, 6.0]}}}
        with pytest.warns(UserWarning, match="matched nothing"):
            rows = validate_against_reference(self._Stub(), ref)["rows"]
        assert len(rows) == 1
        assert rows[0][0] == "EXAFS N(fit) In-O"
        assert rows[0][1] is None
        assert rows[0][5] == "n/a"

    def test_unknown_pair_yields_one_na_row_per_requested_descriptor(self):
        from amorphgen.analysis.validate import validate_against_reference
        ref = {"exafs": {"source": "harmonic-fit",
                         "In-O": {"N_expected": [4.0, 6.0],
                                  "R_expected": [2.1, 2.2],
                                  "sigma2_expected": [0.003, 0.006]}}}
        with pytest.warns(UserWarning, match="matched nothing"):
            rows = validate_against_reference(self._Stub(), ref)["rows"]
        assert [r[0] for r in rows] == ["EXAFS N(fit) In-O",
                                        "EXAFS R(fit) In-O",
                                        "EXAFS sigma2(fit) In-O"]
        assert all(r[1] is None and r[5] == "n/a" for r in rows)

    def test_known_pair_does_not_warn(self):
        import warnings
        from amorphgen.analysis.validate import validate_against_reference
        ref = {"exafs": {"source": "harmonic-fit",
                         "Ga-O": {"N_expected": [3.8, 4.2]}}}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rows = validate_against_reference(self._Stub(), ref)["rows"]
        assert rows[0][5] == "match"


class TestChunking:
    """effective_amplitude sums over an (nk x n_distances) array, so it
    processes distances in blocks to bound memory on long trajectories.
    Blocking must not change the answer."""

    def test_result_is_independent_of_block_size(self, monkeypatch):
        from amorphgen.analysis import harmonic
        k = np.linspace(3.0, 12.0, 40)
        d = _gaussian_shell(1.85, 0.07)          # 8000 distances

        monkeypatch.setattr(harmonic, "_CHUNK", 10 ** 9)   # one block
        whole = harmonic.effective_amplitude(k, d, 2000.0)
        monkeypatch.setattr(harmonic, "_CHUNK", 137)       # many odd blocks
        chunked = harmonic.effective_amplitude(k, d, 2000.0)

        assert np.allclose(whole, chunked, rtol=1e-12, atol=1e-14)

    def test_block_boundary_is_not_dropped(self, monkeypatch):
        """A block size that does not divide the count must still sum all
        of it — an off-by-one here would silently lower N."""
        from amorphgen.analysis import harmonic
        k = np.linspace(3.0, 12.0, 10)
        d = np.full(1000, 1.85)
        monkeypatch.setattr(harmonic, "_CHUNK", 333)
        got = harmonic.effective_amplitude(k, d, 250.0)
        want = 4.0 * 1.85 ** -2 * np.exp(-2 * 1.85 / 5.0) * np.exp(2j * k * 1.85)
        assert np.allclose(got, want)


class TestFitBootstrap:
    """Error bars on the fitted column, from the same structure resamples
    as the geometric column — so the bias error is a paired spread rather
    than a quadrature guess."""

    def _ensemble(self, n, seed0=0):
        base = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
        out = []
        for s in range(n):
            at = base.copy()
            at.positions += np.random.default_rng(seed0 + s).normal(
                0, 0.06, at.positions.shape)
            out.append(at)
        return out

    def _fit(self, frames, **kw):
        from amorphgen.analysis import StructureAnalyser
        return StructureAnalyser(frames, cutoff=2.9).harmonic_fit(
            pair="Si-Si", **kw)["Si-Si"]

    def test_both_columns_get_errors(self):
        d = self._fit(self._ensemble(4), bootstrap=25)
        for key in ("N_geom_err", "N_fit_err", "sigma2_fit_err",
                    "N_bias_percent_err"):
            assert key in d

    def test_bootstrap_zero_disables(self):
        d = self._fit(self._ensemble(4), bootstrap=0)
        assert "N_fit_err" not in d

    def test_identical_structures_give_zero_error(self):
        base = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
        base.positions += np.random.default_rng(0).normal(
            0, 0.06, base.positions.shape)
        d = self._fit([base.copy() for _ in range(4)], bootstrap=15)
        assert d["N_fit_err"] == pytest.approx(0.0, abs=1e-9)

    def test_reproducible_for_a_given_seed(self):
        frames = self._ensemble(4)
        a = self._fit(frames, bootstrap=20, seed=3)
        b = self._fit(frames, bootstrap=20, seed=3)
        assert a["N_fit_err"] == b["N_fit_err"]

    def test_report_shows_paired_errors(self):
        out = format_harmonic_report(
            {"Si-Si": self._fit(self._ensemble(4), bootstrap=20)},
            bonded_only=False)
        assert "+/-" in out
        assert "bootstrap" in out

    def test_report_warns_when_too_few_structures(self):
        out = format_harmonic_report(
            {"Si-Si": self._fit(self._ensemble(3), bootstrap=20)},
            bonded_only=False)
        assert "themselves unreliable" in out

    def test_csv_carries_the_errors(self, tmp_path):
        res = {"Si-Si": self._fit(self._ensemble(4), bootstrap=20)}
        path = tmp_path / "h.csv"
        write_harmonic_csv(res, str(path))
        header = path.read_text().splitlines()[0].split(",")
        for col in ("N_fit_err", "N_bias_percent_err", "n_boot"):
            assert col in header
