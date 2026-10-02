"""Oxygen connectivity distinguishes network formers from modifiers."""

import numpy as np
import pytest
from ase import Atoms

from amorphgen.analysis import StructureAnalyser, compute_oxygen_speciation


def test_all_species_and_modifier_exclusion():
    symbols, positions = [], []
    directions = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    for count in range(5):
        origin = np.array([5 + 10 * count, 5, 5])
        symbols.extend(["O", "Na"] + ["Si"] * count)
        positions.extend([origin, origin + [0, 0, 1.5]])
        positions.extend(origin + 1.5 * directions[:count])
    atoms = Atoms(symbols, positions=positions, cell=[55, 10, 10], pbc=True)
    result = StructureAnalyser([atoms], cutoff=1.6).oxygen_speciation()
    assert result["network_formers"] == ["Si"]
    assert result["network_formers_inferred"]
    assert result["total_oxygen"] == 5
    assert list(result["counts"].values()) == [1] * 5
    assert list(result["fractions"].values()) == pytest.approx([0.2] * 5)
    assert result["per_structure"][0]["network_former_coordination"] == [0, 1, 2, 3, 4]


def test_periodic_images_and_supercell_invariance():
    atoms = Atoms("OSi", positions=[[0, 0, 0], [1.5, 0, 0]],
                  cell=[3, 8, 8], pbc=True)
    for sample in (atoms, atoms.repeat((3, 1, 1))):
        result = compute_oxygen_speciation([sample], cutoff=1.6)
        assert result["fractions"]["bridging"] == 1
        assert result["per_structure"][0]["network_former_coordination"] == [2] * (len(sample) // 2)


def test_pair_cutoffs_and_explicit_network_formers():
    atoms = Atoms("OSiAl", positions=[[0, 0, 0], [1.5, 0, 0], [0, 1.8, 0]],
                  cell=[10, 10, 10], pbc=True)
    sa = StructureAnalyser([atoms], cutoff={"default": 1.0, "O-Si": 1.6, "Al-O": 1.9})
    assert sa.oxygen_speciation()["fractions"]["bridging"] == 1
    assert sa.oxygen_speciation(["Si"])["fractions"]["non_bridging"] == 1
    sa = StructureAnalyser([atoms], cutoff={"default": 1.0, "O-Si": 1.6, "Al-O": 1.7})
    assert sa.oxygen_speciation("Si, Al")["fractions"]["non_bridging"] == 1


def test_ensemble_is_oxygen_weighted():
    a = Atoms("OSi", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[10] * 3, pbc=True)
    b = Atoms("O2Si", positions=[[0, 0, 0], [0, 4, 0], [4, 4, 4]],
              cell=[10] * 3, pbc=True)
    result = compute_oxygen_speciation([a, b], cutoff=1.6)
    assert result["fractions"]["free"] == pytest.approx(2 / 3)
    assert result["fractions"]["non_bridging"] == pytest.approx(1 / 3)


def test_unusual_oxide_requires_selection_and_oxygen_free_is_empty():
    atoms = Atoms("GaO", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[10] * 3, pbc=True)
    with pytest.raises(ValueError, match="explicitly"):
        compute_oxygen_speciation([atoms], cutoff=1.6)
    assert compute_oxygen_speciation([atoms], "Ga", cutoff=1.6)["counts"]["non_bridging"] == 1
    result = compute_oxygen_speciation([Atoms("Cu", cell=[4] * 3, pbc=True)], cutoff=2)
    assert result["total_oxygen"] == 0
    assert sum(result["fractions"].values()) == 0


@pytest.mark.parametrize("formers", [[], ["O"], ["H"], ["Unobtainium"], ["Ge"]])
def test_invalid_formers(formers):
    with pytest.raises(ValueError):
        compute_oxygen_speciation([Atoms("SiO", cell=[10] * 3)], formers, cutoff=1.6)


def test_empty_and_nonfinite_structures():
    with pytest.raises(ValueError, match="nonempty"):
        compute_oxygen_speciation([])
    atoms = Atoms("SiO", positions=[[0, 0, 0], [np.nan, 0, 0]], cell=[10] * 3)
    with pytest.raises(ValueError, match="finite"):
        compute_oxygen_speciation([atoms], cutoff=1.6)


def test_different_element_sets_do_not_silently_lose_cutoffs():
    with pytest.raises(ValueError, match="same element set"):
        compute_oxygen_speciation([Atoms("SiO"), Atoms("AlO")], cutoff=2.0)


def test_invalid_periodic_cell_and_nonperiodic_molecule():
    atoms = Atoms("SiO", positions=[[0, 0, 0], [1.5, 0, 0]], pbc=True)
    with pytest.raises(ValueError, match="cell vectors"):
        compute_oxygen_speciation([atoms], cutoff=2.0)
    atoms.pbc = False
    assert compute_oxygen_speciation([atoms], cutoff=2.0)["counts"]["non_bridging"] == 1
