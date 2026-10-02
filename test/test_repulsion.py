"""Core derivatives, periodic image accounting, and optional backend parity."""
import importlib.util

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes

from amorphgen.utils.repulsion import (
    validate_repulsive_core_config,
    with_repulsive_core,
    wrap_torch_model,
)


CONFIG = {"enabled": True, "cutoff": 1.0, "strength": 2.0}


class ZeroCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress", "energies", "stresses"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "free_energy": -0.1,
                        "forces": np.zeros((len(atoms), 3)), "stress": np.zeros(6),
                        "energies": np.zeros(len(atoms)),
                        "stresses": np.zeros((len(atoms), 6))}


def pair(distance=0.8):
    return Atoms("He2", positions=[[0, 0, 0], [distance, 0, 0]], cell=[4, 4, 4])


def attach(atoms, config=CONFIG):
    atoms.calc = with_repulsive_core(ZeroCalculator(), config)
    return atoms


def test_disabled_and_idempotent_wrappers():
    base = ZeroCalculator()
    assert with_repulsive_core(base) is base
    assert with_repulsive_core(base, {"enabled": False}) is base
    wrapped = with_repulsive_core(base, CONFIG)
    assert with_repulsive_core(wrapped, CONFIG) is wrapped
    assert with_repulsive_core(wrapped) is wrapped
    assert with_repulsive_core(wrapped, {"enabled": False}) is wrapped
    changed = with_repulsive_core(wrapped, {**CONFIG, "strength": 3.0})
    assert changed.base_calculator is base
    assert wrap_torch_model(base) is base  # No torch import required.


@pytest.mark.parametrize("config", [True, {"enabled": "false"}, {"cutoff": 0},
                                    {"strength": -1}, {"strength": float("nan")},
                                    {"cutoff": float("inf")}, {"cutoff": True},
                                    {"cutoff": "1.0"}, {"strength": [1.0]},
                                    {"cuttof": 1}])
def test_invalid_configuration(config):
    with pytest.raises(ValueError, match="repulsive_core"):
        validate_repulsive_core_config(config)


def test_pair_energy_free_energy_and_force_derivative():
    atoms = attach(pair())
    energy = CONFIG["strength"] * (CONFIG["cutoff"] / 0.8 - 1)**2
    assert atoms.get_potential_energy() == pytest.approx(energy)
    assert atoms.get_potential_energy(force_consistent=True) == pytest.approx(energy - 0.1)
    assert atoms.get_potential_energies().sum() == pytest.approx(energy)
    force = atoms.get_forces()
    h = 1e-6
    atoms.positions[1, 0] += h
    plus = atoms.get_potential_energy()
    atoms.positions[1, 0] -= 2 * h
    minus = atoms.get_potential_energy()
    assert force[1, 0] == pytest.approx(-(plus - minus) / (2 * h), rel=1e-8)
    assert force[1, 0] > 0
    np.testing.assert_allclose(force.sum(axis=0), 0.0)


def test_core_is_continuous_and_zero_beyond_cutoff():
    for distance in [1.0, 1.1, 4.0]:
        atoms = attach(pair(distance))
        assert atoms.get_potential_energy() == 0.0
        np.testing.assert_array_equal(atoms.get_forces(), 0.0)
    almost = attach(pair(1.0 - 1e-8))
    assert almost.get_potential_energy() < 1e-14
    assert abs(almost.get_forces()).max() < 1e-6


def test_exact_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlapping atoms"):
        attach(pair(0.0)).get_potential_energy()


def test_stress_matches_strain_derivative_and_atomic_sum():
    atoms = attach(Atoms("He2", positions=[[0, 0, 0], [0.6, 0.4, 0.1]],
                        cell=[[3.0, 0, 0], [0.2, 3.5, 0], [0.1, 0.3, 4.0]], pbc=True))
    stress = atoms.get_stress(voigt=False)
    np.testing.assert_allclose(atoms.get_stresses().sum(axis=0), atoms.get_stress())
    cell, positions, volume = atoms.cell.copy(), atoms.positions.copy(), atoms.get_volume()
    h = 1e-6
    for i in range(3):
        for j in range(3):
            strain = np.zeros((3, 3))
            strain[i, j] = h
            atoms.set_cell(cell @ (np.eye(3) + strain))
            atoms.positions = positions @ (np.eye(3) + strain)
            plus = atoms.get_potential_energy()
            atoms.set_cell(cell @ (np.eye(3) - strain))
            atoms.positions = positions @ (np.eye(3) - strain)
            minus = atoms.get_potential_energy()
            assert stress[i, j] == pytest.approx((plus - minus) / (2 * h * volume), rel=1e-7)


def test_periodic_self_images_have_energy_and_stress_without_net_force():
    atoms = attach(Atoms("He", positions=[[0, 0, 0]], cell=[0.8, 4, 4], pbc=True))
    # Two directed neighbors (+a and -a) count as one pair per cell.
    expected = 2.0 * (1.0 / 0.8 - 1)**2
    assert atoms.get_potential_energy() == pytest.approx(expected)
    np.testing.assert_allclose(atoms.get_forces(), 0, atol=1e-12)
    assert atoms.get_stress()[0] < 0
    volume = atoms.get_volume()
    h = 1e-6
    atoms.set_cell([0.8 * (1 + h), 4, 4])
    plus = atoms.get_potential_energy()
    atoms.set_cell([0.8 * (1 - h), 4, 4])
    minus = atoms.get_potential_energy()
    atoms.set_cell([0.8, 4, 4])
    assert atoms.get_stress()[0] == pytest.approx((plus - minus) / (2 * h * volume), rel=1e-8)


def test_wrapper_does_not_advertise_missing_base_stress_or_free_energy():
    class EnergyForces(ZeroCalculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            self.results = {k: self.results[k] for k in self.implemented_properties}

    atoms = pair()
    atoms.calc = with_repulsive_core(EnergyForces(), CONFIG)
    assert atoms.get_potential_energy() > 0
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_stress()
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_potential_energy(force_consistent=True)


@pytest.mark.skipif(importlib.util.find_spec("torch_sim") is None,
                    reason="torch-sim optional dependency not installed")
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_torch_batched_parity_and_property_preservation(dtype_name):
    import torch
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface

    class ZeroModel(ModelInterface):
        def __init__(self):
            super().__init__()
            self._device = torch.device("cpu")
            self._dtype = getattr(torch, dtype_name)
            self._compute_forces = True
            self._compute_stress = True

        def forward(self, state, **kwargs):
            return {"energy": state.positions.new_zeros(state.n_systems),
                    "free_energy": state.positions.new_full((state.n_systems,), -0.1),
                    "forces": torch.zeros_like(state.positions),
                    "stress": torch.zeros_like(state.cell),
                    "energies": state.positions.new_zeros(state.n_atoms),
                    "stresses": state.positions.new_zeros((state.n_atoms, 3, 3)),
                    "extra": state.positions.new_full((state.n_systems,), 42)}

    structures = [Atoms("He", positions=[[0, 0, 0]], cell=[0.8, 4, 4], pbc=True),
                  Atoms("He2", positions=[[0, 0, 0], [0.6, 0.4, 0.1]],
                        cell=[[3, 0, 0], [0.2, 3.5, 0], [0.1, 0.3, 4]], pbc=True)]
    model = ZeroModel()
    wrapped = wrap_torch_model(model, CONFIG)
    assert wrapped.device == model.device and wrapped.dtype == model.dtype
    assert wrapped.compute_stress == model.compute_stress
    assert wrap_torch_model(wrapped, CONFIG) is wrapped
    assert wrap_torch_model(wrapped, {**CONFIG, "strength": 4.0}).base_model is model
    state = ts.io.atoms_to_state(structures, device=model.device, dtype=model.dtype)
    result = wrapped(state)
    assert set(result) == set(model(state))
    offset = 0
    for k, atoms in enumerate(structures):
        attach(atoms)
        assert result["energy"][k].item() == pytest.approx(atoms.get_potential_energy(), rel=1e-5)
        np.testing.assert_allclose(result["forces"][offset:offset + len(atoms)].numpy(),
                                   atoms.get_forces(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(result["stress"][k].numpy(), atoms.get_stress(voigt=False),
                                   rtol=1e-5, atol=1e-6)
        assert result["free_energy"][k].item() == pytest.approx(
            atoms.get_potential_energy(force_consistent=True), rel=1e-5)
        offset += len(atoms)
    assert torch.all(result["extra"] == 42)
    assert all(value.dtype == model.dtype and value.device == model.device for value in result.values())
    torch.testing.assert_close(result["energies"].sum(), result["energy"].sum())
    torch.testing.assert_close(result["stresses"].sum(0), result["stress"].sum(0))

    model._compute_stress = False
    original_forward = model.forward
    model.forward = lambda state: {k: v for k, v in original_forward(state).items()
                                  if k not in ("stress", "stresses")}
    assert not wrapped.compute_stress
    assert "stress" not in wrapped(state)

    overlap = ts.io.atoms_to_state([pair(0.0)], device=model.device, dtype=model.dtype)
    with pytest.raises(ValueError, match="overlapping atoms"):
        wrapped(overlap)
