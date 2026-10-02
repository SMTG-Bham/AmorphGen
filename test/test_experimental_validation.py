"""Offline comparisons of simulated amorphous structures with experiment.

The coordinates in data/experimental are model snapshots from the bundled
validation examples, not experimentally determined atomic coordinates. These
tests exercise the real neighbour analysis without downloading an ML model.

Mozzi & Warren, J. Appl. Cryst. 2, 164–172 (1969), abstract:
https://doi.org/10.1107/S0021889869006868
reports Si–O = 1.62 Å and coordination Si–O = 4, O–Si = 2 for vitreous silica.
The 0.03 Å and 0.1/0.05 coordination allowances below are model regression
tolerances, not experimental error bars or confidence intervals. The paper's
144-degree Si–O–Si value is a distribution *mode*, so we do not compare it
with the analyser's mean angle. Density is also omitted: an input cell volume
is not an independent density prediction.

Laaziri et al., Phys. Rev. B 60, 13520–13533 (1999), abstract:
https://doi.org/10.1103/PhysRevB.60.13520
reports first-shell coordination 3.79 (as implanted) and 3.88 (annealed at
600 C) for pure amorphous Si. The model snapshot disagrees with both. That
discrepancy is retained explicitly, rather than widening an acceptance band.
Single snapshots do not establish agreement of a generated ensemble.
"""

from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from amorphgen.analysis import StructureAnalyser, validate_against_reference


DATA = Path(__file__).parent / "data" / "experimental"

SILICA_REFERENCE = {
    "system": "vitreous silica",
    "references": [
        "Mozzi & Warren (1969), https://doi.org/10.1107/S0021889869006868"
    ],
    "bond_distances": {"Si-O": {"expected": [1.59, 1.65], "units": "A"}},
    "coordination": {
        "Si-O": {"mean_expected": [3.9, 4.1]},
        "O-Si": {"mean_expected": [1.95, 2.05]},
    },
}


@pytest.fixture(scope="module", params=[2.0, 2.2, "auto-rdf"])
def silica_analyser(request):
    """Check the first-shell plateau and the production default cutoff."""
    cutoff = request.param
    if isinstance(cutoff, float):
        cutoff = {"Si-O": cutoff, "Si-Si": 0.0, "O-O": 0.0}
    return StructureAnalyser([read(DATA / "sio2.xyz")], cutoff=cutoff)


def test_silica_bond_length_agrees_with_xray_diffraction(silica_analyser):
    # The analyser stores unordered bond distances alphabetically: O-Si.
    distance = silica_analyser.bond_distances()["O-Si"]["mean"]
    assert np.isfinite(distance)
    assert distance == pytest.approx(1.62, abs=0.03), (
        f"Simulated Si-O distance {distance:.4f} A differs from the measured "
        "1.62 A by more than the 0.03 A model tolerance"
    )


@pytest.mark.parametrize("pair, measured, tolerance", [
    ("Si-O", 4.0, 0.1),
    ("O-Si", 2.0, 0.05),
])
def test_silica_coordination_agrees_with_xray_diffraction(
    silica_analyser, pair, measured, tolerance
):
    # Coordination is directional; checking both also catches reversed keys.
    coordination = silica_analyser.coordination()[pair]["mean"]
    assert coordination == pytest.approx(measured, abs=tolerance), (
        f"Simulated CN {pair}={coordination:.4f}; measured {measured:g}, "
        f"model tolerance {tolerance:g}"
    )


def test_silica_reference_report_preserves_single_snapshot_limit(silica_analyser):
    result = validate_against_reference(silica_analyser, SILICA_REFERENCE)
    assert result["sources"] == SILICA_REFERENCE["references"]
    assert [row[0] for row in result["rows"]] == [
        "Bond Si-O", "CN Si-O", "CN O-Si"
    ]
    for descriptor, value, low, high, _, verdict in result["rows"]:
        assert low <= value <= high, (descriptor, value, low, high)
        # A point estimate inside a model tolerance is not evidence for an
        # ensemble mean. Never replicate the same snapshot to obtain a CI.
        assert verdict == "inconclusive"
        interval = result["intervals"][descriptor]
        assert interval["n_structures"] == 1
        assert interval["ci_low"] is None
        assert interval["ci_high"] is None


@pytest.mark.parametrize("sample, measured", [
    ("as implanted", 3.79),
    ("annealed at 600 C", 3.88),
])
def test_silicon_comparison_keeps_experimental_coordination_discrepancy(
    sample, measured
):
    analyser = StructureAnalyser([read(DATA / "si.xyz")], cutoff=2.8)
    reference = {
        "system": f"pure amorphous Si, {sample}",
        "references": [
            "Laaziri et al. (1999), https://doi.org/10.1103/PhysRevB.60.13520"
        ],
        # A reported scalar, with no invented experimental uncertainty.
        "coordination": {"Si-Si": {"mean_expected": [measured, measured]}},
    }
    result = validate_against_reference(analyser, reference)
    assert result["sources"] == reference["references"]
    descriptor, computed, low, high, _, verdict = result["rows"][0]
    assert descriptor == "CN Si-Si"
    assert (low, high) == (measured, measured)
    assert computed == pytest.approx(4.0)
    assert computed > high, "The bundled Si model overcoordinates experiment"
    assert verdict == "inconclusive"  # one structure cannot estimate a CI


def test_silica_distance_comparison_detects_expanded_structure():
    """The experimental distance must discriminate a distorted model."""
    atoms = read(DATA / "sio2.xyz")
    atoms.set_cell(atoms.cell * 1.1, scale_atoms=True)
    analyser = StructureAnalyser(
        [atoms], cutoff={"Si-O": 2.2, "Si-Si": 0.0, "O-O": 0.0}
    )
    result = validate_against_reference(analyser, SILICA_REFERENCE)
    bond = next(row for row in result["rows"] if row[0] == "Bond Si-O")
    assert np.isfinite(bond[1])
    assert bond[1] > bond[3]
    # Uniform expansion preserves the topology but spoils the bond distance.
    assert analyser.coordination()["Si-O"]["mean"] == pytest.approx(4.0)
