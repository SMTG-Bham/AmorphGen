"""Offline scattering benchmarks with experimental reference provenance.

Silica: Mozzi & Warren, J. Appl. Cryst. 2, 164–172 (1969),
https://doi.org/10.1107/S0021889869006868, report a Si–O distance of 1.62 Å
and an experimental Q range reaching 20 Å^-1. Only the first-shell position
is compared: no measured S(Q)/T(r) intensity curve is bundled. The Lorch
window is our transform choice, not a claim about the experimental reduction.

Silicon calibration: Cline et al., NIST SP 260-245 (2024), Table 2,
https://doi.org/10.6028/NIST.SP.260-245. These tabulated positions are computed
from the experimentally certified powder lattice parameter at 22.5 °C. Using
that parameter as input checks scattering geometry and angular units, not a
potential's ability to predict lattice constants or experimental intensities.
See data/experimental/README.md for fixture provenance and tolerance policy.
"""

from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from amorphgen.analysis import StructureAnalyser
from amorphgen.analysis.xrd import compute_xrd_pattern


DATA = Path(__file__).parent / "data" / "experimental"


@pytest.mark.parametrize("weighting", ["xray", "neutron"])
def test_silica_total_correlation_first_shell_matches_diffraction(weighting):
    analyser = StructureAnalyser(str(DATA / "sio2.xyz"), cutoff=2.0)
    result = analyser.total_correlation(
        weighting=weighting, qmin=0.3, qmax=20, nq=400,
        rmax=4, nr=400, sigma_q=0, window="lorch", n_bootstrap=0,
    )
    r = np.asarray(result["r"])
    tr = np.asarray(result["T_r"])
    # Isolate the Si–O shell, excluding the O–O and Si–Si shells. Comparing
    # positions allows different X-ray/neutron weights without fitting scale.
    shell = (r >= 1.3) & (r <= 2.0)
    assert np.isfinite(tr[shell]).all()
    peak = int(np.argmax(tr[shell]))
    assert 0 < peak < shell.sum() - 1, "No resolved first-shell peak"
    # A 0.05 Å regression allowance covers the model and finite-Q transform;
    # it is not the uncertainty of the experimental bond length.
    assert r[shell][peak] == pytest.approx(1.62, abs=0.05)
    assert result["uncertainty"]["n_structures"] == 1
    assert all(value is None for value in result["uncertainty"]["ci_low"])


@pytest.fixture(scope="module")
def silicon_standard():
    # Powder value 0.5431109 nm, not the distinct single-crystal boule value.
    silicon = bulk("Si", "diamond", a=5.431109, cubic=True)
    return compute_xrd_pattern(
        [silicon], wavelength=1.5405929, qmax=6, nq=3000,
        sigma_q=0, n_bootstrap=0,
    )


@pytest.mark.parametrize("two_theta", [
    pytest.param(28.441, id="111"),
    pytest.param(47.301, id="220"),
    pytest.param(56.120, id="311"),
    pytest.param(69.127, id="400"),
    pytest.param(76.373, id="331"),
    pytest.param(88.026, id="422"),
])
def test_silicon_reflections_match_nist_standard(silicon_standard, two_theta):
    angles = np.asarray(silicon_standard["two_theta"])
    intensity = np.asarray(silicon_standard["intensity"], dtype=float)
    region = (np.abs(angles - two_theta) < 0.3) & np.isfinite(intensity)
    assert region.any(), f"Missing reflection near {two_theta} degrees"
    peak = np.argmax(intensity[region])
    # An extinct shell can still contain roundoff noise: require a real peak.
    assert intensity[region][peak] > 1.0  # electron^2/atom
    # q-bin width is 0.002 Å^-1; 0.03° covers half-bin angular error even
    # at the highest tested angle, plus the table's 0.001° rounding.
    # This numerical tolerance is not the SRM's certified uncertainty.
    assert angles[region][peak] == pytest.approx(two_theta, abs=0.03)
