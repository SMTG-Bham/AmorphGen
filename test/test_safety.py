"""MLIP tripwires on real ASE states, without ML packages or model downloads."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from amorphgen.utils.common import DivergenceError
from amorphgen.utils.safety import SafetyMonitor, validate_safety_config


class ConstantCalculator(Calculator):
    implemented_properties = ("energy", "forces", "stress")

    def __init__(self, energy=0.0, force=0.0, stress=0.0):
        super().__init__()
        self.energy = energy
        self.force = force
        self.stress = stress
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        self.results = {
            "energy": self.energy,
            "forces": np.full((len(atoms), 3), self.force),
            "stress": np.full(6, self.stress),
        }


@pytest.fixture
def atoms():
    result = Atoms("Cu2", positions=[[0, 0, 0], [2, 0, 0]], cell=[8, 8, 8], pbc=True)
    result.calc = ConstantCalculator()
    return result


@pytest.mark.parametrize("property,value,label", [
    ("energy", np.nan, "potential energy"),
    ("energy", np.inf, "potential energy"),
    ("force", np.nan, "forces"),
    ("force", np.inf, "forces"),
    ("stress", np.nan, "stress"),
])
def test_nonfinite_model_results(atoms, property, value, label):
    setattr(atoms.calc, property, value)
    with pytest.raises(DivergenceError, match=f"Non-finite {label}.*step 17.*melt"):
        SafetyMonitor(context="melt").check(atoms, step=17)


@pytest.mark.parametrize("state", ["positions", "cell", "momenta"])
def test_nonfinite_geometry_fails_before_calculator(atoms, state):
    if state == "positions":
        atoms.positions[0, 0] = np.nan
    elif state == "cell":
        atoms.cell[0, 0] = np.inf
    else:
        atoms.set_momenta(np.full((len(atoms), 3), np.nan))
    with pytest.raises(DivergenceError, match=f"Non-finite {state}"):
        SafetyMonitor().check(atoms)
    assert atoms.calc.calls == 0


def test_constraints_cannot_hide_nonfinite_forces(atoms):
    atoms.set_constraint(FixAtoms(indices=[0, 1]))
    atoms.calc.force = np.nan
    assert np.isfinite(atoms.get_forces()).all()
    with pytest.raises(DivergenceError, match="Non-finite forces"):
        SafetyMonitor().check(atoms)


def test_all_fixed_stationary_atoms_are_valid(atoms):
    atoms.set_constraint(FixAtoms(indices=[0, 1]))
    SafetyMonitor().check(atoms)


def test_unavailable_stress_does_not_get_requested(atoms):
    class NoStressCalculator(ConstantCalculator):
        implemented_properties = ("energy", "forces")

        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            del self.results["stress"]

    atoms.calc = NoStressCalculator()
    SafetyMonitor().check(atoms)
    assert atoms.calc.calls == 1


def test_nonfinite_cached_force_consistent_energy_is_rejected(atoms):
    atoms.calc = SinglePointCalculator(
        atoms, energy=0, free_energy=np.nan, forces=np.zeros((len(atoms), 3))
    )
    with pytest.raises(DivergenceError, match="Non-finite free energy"):
        SafetyMonitor().check(atoms)


def test_calculator_errors_propagate(atoms):
    class BrokenCalculator(ConstantCalculator):
        def calculate(self, *args, **kwargs):
            raise RuntimeError("model evaluation failed")

    atoms.calc = BrokenCalculator()
    with pytest.raises(RuntimeError, match="^model evaluation failed$"):
        SafetyMonitor().check(atoms)


@pytest.mark.parametrize("periodic", [False, True])
def test_close_contacts_across_periodic_boundary(atoms, periodic):
    atoms.pbc = periodic
    atoms.positions[1, 0] = 7.8 if periodic else 0.2
    with pytest.raises(DivergenceError, match="Close contact between atoms"):
        SafetyMonitor().check(atoms)
    assert atoms.calc.calls == 0


@pytest.mark.parametrize("cell", [
    [[0.2, 0, 0], [0, 8, 0], [0, 0, 8]],
    [[8, 0, 0], [8, 0.2, 0], [0, 0, 8]],
])
def test_periodic_self_image_is_detected_including_skew_cells(cell):
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=cell, pbc=True)
    with pytest.raises(DivergenceError, match="periodic self-image"):
        SafetyMonitor().check_geometry(atoms)


def test_degenerate_periodic_cell_fails_before_neighbour_search(atoms):
    atoms.cell[1] = atoms.cell[0]
    with pytest.raises(DivergenceError, match="Degenerate periodic cell"):
        SafetyMonitor().check_geometry(atoms)


def test_nonperiodic_molecule_needs_no_cell(atoms):
    atoms.pbc = False
    atoms.cell = np.zeros((3, 3))
    SafetyMonitor().check(atoms)


def test_energy_jumps_compare_consecutive_per_atom_values(atoms):
    monitor = SafetyMonitor({"max_energy_jump_per_atom": 2})
    monitor.check(atoms, step=0)
    atoms.calc.energy = 3
    atoms.calc.reset()
    monitor.check(atoms, step=1)  # 1.5 eV/atom
    atoms.calc.energy = 6
    atoms.calc.reset()
    monitor.check(atoms, step=2)  # another 1.5 eV/atom, not a jump of 3
    atoms.calc.energy = -1
    atoms.calc.reset()
    with pytest.raises(DivergenceError, match="Energy jump.*3.5 eV/atom"):
        monitor.check(atoms, step=3)


def test_temperature_runaway(atoms):
    atoms.set_momenta(np.ones((len(atoms), 3)) * 100)
    assert atoms.get_temperature() > 100000
    with pytest.raises(DivergenceError, match="Temperature runaway"):
        SafetyMonitor().check(atoms)
    assert atoms.calc.calls == 0


@pytest.mark.parametrize("scale", [0.5, 2])
def test_volume_runaway_uses_initial_cell(atoms, scale):
    monitor = SafetyMonitor()
    monitor.check(atoms)
    atoms.set_cell(atoms.cell * scale, scale_atoms=True)
    with pytest.raises(DivergenceError, match="Volume runaway"):
        monitor.check(atoms)


def test_geometry_checks_do_not_advance_energy_baseline(atoms):
    monitor = SafetyMonitor({"max_energy_jump_per_atom": 1})
    monitor.check(atoms)
    atoms.calc.energy = 10
    atoms.calc.reset()
    monitor.check_geometry(atoms)
    with pytest.raises(DivergenceError, match="Energy jump"):
        monitor.check(atoms)


def test_disabled_physical_thresholds_keep_finite_guard(atoms):
    monitor = SafetyMonitor({
        "min_distance": None, "max_energy_jump_per_atom": None,
        "max_temperature": None, "min_volume_ratio": None, "max_volume_ratio": None,
    })
    monitor.check(atoms)
    atoms.positions[1] = [0.1, 0, 0]
    atoms.set_cell(atoms.cell * 2)
    atoms.set_momenta(np.ones((len(atoms), 3)) * 100)
    atoms.calc.energy = 100
    monitor.check(atoms)
    atoms.calc.force = np.nan
    atoms.calc.reset()
    with pytest.raises(DivergenceError, match="Non-finite forces"):
        monitor.check(atoms)


@pytest.mark.parametrize("config", [
    False, {"min_distance": -1}, {"max_temperature": np.nan},
    {"max_energy_jump_per_atom": True}, {"max_temperature": "hot"},
    {"max_energy_jump": 1}, {"min_volume_ratio": 5, "max_volume_ratio": 2},
    {"reference": {"model": "mace", "interval": 0}},
    {"reference": {"model": "mace", "interval": 2.5}},
    {"reference": {"interval": 1}},
    {"reference": {"model": "mace", "max_force_rmse": -1}},
    {"reference": {"model": "mace", "unknown": 1}},
])
def test_bad_config_rejected(config):
    with pytest.raises(ValueError):
        validate_safety_config(config)


def test_config_is_copied_and_empty_reference_disabled():
    config = {"reference": {"model": "mace", "interval": 2}}
    monitor = SafetyMonitor(config)
    config["reference"]["interval"] = 17
    assert monitor.config["reference"]["interval"] == 2
    assert validate_safety_config({"reference": {}})["reference"] is None


@pytest.mark.parametrize("steps", [[0, 1, 2, 3, 4], [None] * 5])
def test_reference_cadence_and_no_primary_mutation(atoms, steps):
    reference = ConstantCalculator()
    monitor = SafetyMonitor({"reference": {"interval": 2}}, reference_calc=reference)
    original_calc = atoms.calc
    atoms.set_constraint(FixAtoms(indices=[0]))
    for i, step in enumerate(steps):
        atoms.positions[1, 1] = i * 0.01  # invalidate reference caching
        before = atoms.copy()
        monitor.check(atoms, step)
        assert atoms.calc is original_calc
        assert len(atoms.constraints) == 1
        np.testing.assert_array_equal(atoms.positions, before.positions)
        np.testing.assert_array_equal(atoms.cell, before.cell)
    assert reference.calls == 3
    assert len(reference.atoms.constraints) == 0


def test_reference_calculator_is_lazy_and_reused(atoms, monkeypatch):
    from amorphgen.utils import calculators

    loaded = []

    def load(**kwargs):
        loaded.append(kwargs)
        return ConstantCalculator()

    monkeypatch.setattr(calculators, "get_calculator", load)
    monitor = SafetyMonitor({"reference": {
        "model": "second-model", "device": "cpu", "default_dtype": "float64",
        "interval": 2,
    }})
    assert loaded == []
    monitor.check(atoms, 0)
    monitor.check(atoms, 1)
    monitor.check(atoms, 2)
    assert loaded == [{"model": "second-model", "device": "cpu", "default_dtype": "float64"}]


def test_reference_force_disagreement_ignores_constraints(atoms):
    atoms.set_constraint(FixAtoms(indices=[0, 1]))
    reference = ConstantCalculator(force=2)
    monitor = SafetyMonitor(reference_calc=reference, context="relax")
    with pytest.raises(DivergenceError, match="Reference model disagreement.*force RMSE.*relax"):
        monitor.check(atoms, step=5)


def test_reference_energy_offsets_are_opt_in(atoms):
    reference = ConstantCalculator(energy=100)
    SafetyMonitor(reference_calc=reference).check(atoms)
    monitor = SafetyMonitor(
        {"reference": {"max_energy_difference_per_atom": 10}}, reference_calc=reference
    )
    with pytest.raises(DivergenceError, match="energy difference 50 eV/atom"):
        monitor.check(atoms)


@pytest.mark.parametrize("property", ["energy", "force", "stress"])
def test_nonfinite_reference_is_rejected(atoms, property):
    reference = ConstantCalculator()
    setattr(reference, property, np.nan)
    with pytest.raises(DivergenceError, match="Non-finite reference"):
        SafetyMonitor(reference_calc=reference).check(atoms)


def test_reference_cannot_reuse_primary_calculator(atoms):
    with pytest.raises(ValueError, match="independent calculator"):
        SafetyMonitor(reference_calc=atoms.calc).check(atoms)


def test_reference_errors_propagate(atoms):
    class BrokenReference(ConstantCalculator):
        def calculate(self, *args, **kwargs):
            raise LookupError("reference load failed")

    with pytest.raises(LookupError, match="reference load failed"):
        SafetyMonitor(reference_calc=BrokenReference()).check(atoms)


def test_reference_compares_raw_model_under_repulsive_wrapper(atoms):
    class FakeCore(ConstantCalculator):
        def __init__(self):
            super().__init__(energy=10, force=10)
            self.base_calculator = ConstantCalculator()

    atoms.calc = FakeCore()
    monitor = SafetyMonitor(
        {"reference": {"max_energy_difference_per_atom": 1}},
        reference_calc=ConstantCalculator(),
    )
    monitor.check(atoms)  # matching raw models despite the added stabilizing core


def test_duplicate_steps_still_check_results_without_rechecking_reference(atoms):
    reference = ConstantCalculator()
    monitor = SafetyMonitor(reference_calc=reference)
    monitor.check(atoms, step=0)
    monitor.check(atoms, step=0)
    assert reference.calls == 1
    atoms.calc.results["stress"] = np.full(6, np.nan)
    with pytest.raises(DivergenceError, match="Non-finite stress"):
        monitor.check(atoms, step=0)


def test_reference_cannot_reuse_wrapped_base_calculator(atoms):
    primary = ConstantCalculator()
    atoms.calc.base_calculator = primary
    with pytest.raises(ValueError, match="independent calculator"):
        SafetyMonitor(reference_calc=primary).check(atoms)
    assert primary.calls == 0


def test_repeated_geometry_reuses_contact_search_but_checks_temperature(atoms, monkeypatch):
    from amorphgen.utils import safety

    calls = []
    real_neighbours = safety.neighbor_list

    def neighbours(*args, **kwargs):
        calls.append(1)
        return real_neighbours(*args, **kwargs)

    monkeypatch.setattr(safety, "neighbor_list", neighbours)
    monitor = SafetyMonitor()
    monitor.check_geometry(atoms)
    monitor.check(atoms)
    monitor.check(atoms)
    assert len(calls) == 1
    atoms.set_momenta(np.ones((len(atoms), 3)) * 100)
    with pytest.raises(DivergenceError, match="Temperature runaway"):
        monitor.check_geometry(atoms)


@pytest.mark.parametrize("change", ["positions", "cell", "pbc", "threshold"])
def test_contact_cache_invalidated_by_in_place_geometry_changes(atoms, change):
    monitor = SafetyMonitor()
    if change == "pbc":
        atoms.pbc = False
        atoms.positions[1, 0] = 7.8
    monitor.check_geometry(atoms)
    if change == "positions":
        atoms.positions[1, 0] = 0.1
    elif change == "cell":
        atoms.cell[0, 0] = 2.2
    elif change == "pbc":
        atoms.pbc[:] = True
    else:
        monitor.config["min_distance"] = 2.1
    with pytest.raises(DivergenceError, match="Close contact"):
        monitor.check_geometry(atoms)
