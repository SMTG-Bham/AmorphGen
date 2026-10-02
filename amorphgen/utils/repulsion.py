"""Optional, species-independent repulsive pair core for ASE and torch-sim.

For each unique pair, including periodic images, the added energy is
``U(r) = strength * (cutoff / r - 1)**2`` for ``0 < r < cutoff`` and zero
otherwise. ``strength`` is in eV and ``cutoff`` in Angstrom. Both energy and
force vanish continuously at the cutoff; the energy diverges at overlap.
Exact overlaps raise an error because their force direction is undefined.
This changes the potential energy surface and is disabled by default.

Usage::

    atoms.calc = with_repulsive_core(mlip, {
        "enabled": True, "cutoff": 1.0, "strength": 1.0,
    })

The torch-sim wrapper uses the same potential and a native batched neighbor
list. Importing this module or using the ASE wrapper never imports torch.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from ase.stress import full_3x3_to_voigt_6_stress


def validate_repulsive_core_config(config=None):
    """Return validated defaults for the ``repulsive_core`` config block."""
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("repulsive_core must be a mapping")
    unknown = set(config) - {"enabled", "cutoff", "strength"}
    if unknown:
        raise ValueError(f"Unknown repulsive_core setting(s): {sorted(unknown)}")
    result = {"enabled": False, "cutoff": 1.0, "strength": 1.0, **config}
    if not isinstance(result["enabled"], bool):
        raise ValueError("repulsive_core.enabled must be true or false")
    for key in ("cutoff", "strength"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"repulsive_core.{key} must be a finite positive number")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"repulsive_core.{key} must be finite and positive")
        result[key] = number
    return result


def _ase_core(atoms, cutoff, strength, need_stress=False):
    """Evaluate the core using a directed pair list, including self images."""
    ii, distance, displacement = neighbor_list("idD", atoms, cutoff)
    if np.any(distance == 0):
        raise ValueError("Repulsive core encountered overlapping atoms (zero distance); "
                         "separate the atoms before running the model")
    u = strength * (cutoff / distance - 1.0)**2
    derivative = -2.0 * strength * cutoff * (cutoff / distance - 1.0) / distance**2
    pair_force = derivative[:, None] * displacement / distance[:, None]
    forces = np.zeros((len(atoms), 3))
    energies = np.zeros(len(atoms))
    np.add.at(forces, ii, pair_force)
    np.add.at(energies, ii, 0.5 * u)
    result = {"energy": float(energies.sum()), "free_energy": float(energies.sum()),
              "forces": forces, "energies": energies}
    if need_stress:
        if atoms.cell.rank < 3 or atoms.get_volume() <= 0:
            raise ValueError("Repulsive-core stress requires a nonzero three-dimensional cell")
        virial = 0.5 * np.einsum("ni,nj->nij", displacement, pair_force)
        stresses = np.zeros((len(atoms), 3, 3))
        np.add.at(stresses, ii, virial / atoms.get_volume())
        result["stress"] = stresses.sum(axis=0)
        result["stresses"] = stresses
    return result


class RepulsiveCoreCalculator(Calculator):
    """Add the core to properties supported by ``base_calculator`` only.

    ``free_energy`` receives the same conservative correction as ``energy``.
    Stress is available only if the underlying calculator implements it;
    the wrapper never substitutes core-only stress for missing MLIP stress.
    """

    def __init__(self, base_calculator, config):
        super().__init__()
        self.base_calculator = base_calculator
        self.repulsive_core_config = validate_repulsive_core_config(config)
        self.implemented_properties = list(base_calculator.implemented_properties)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        for prop in properties:
            self.base_calculator.get_property(prop, self.atoms)
        base = self.base_calculator.results
        correction = _ase_core(
            self.atoms, self.repulsive_core_config["cutoff"],
            self.repulsive_core_config["strength"],
            need_stress=bool({"stress", "stresses"} & base.keys()),
        )
        self.results = {}
        for prop, value in base.items():
            if prop not in self.implemented_properties:
                continue
            if prop in correction:
                extra = correction[prop]
                if prop in ("stress", "stresses") and np.shape(value)[-1:] == (6,):
                    extra = full_3x3_to_voigt_6_stress(extra)
                value = value + extra
            self.results[prop] = deepcopy(value)


def with_repulsive_core(calc, config=None):
    """Wrap an ASE calculator if enabled; repeated wrapping never doubles it."""
    config = validate_repulsive_core_config(config)
    if not config["enabled"]:
        return calc
    if isinstance(calc, RepulsiveCoreCalculator):
        if calc.repulsive_core_config == config:
            return calc
        calc = calc.base_calculator
    return RepulsiveCoreCalculator(calc, config)


def _torch_core(state, config, properties):
    """Batched tensor implementation, with the same directed-pair weights."""
    import torch
    from torch_sim.neighbors import torchsim_nl

    positions = state.positions
    cells = state.cell.transpose(-1, -2)  # torch-sim stores lattice columns.
    system_idx = state.system_idx
    mapping, systems, shifts = torchsim_nl(
        positions=positions, cell=cells, pbc=state.pbc,
        cutoff=config["cutoff"], system_idx=system_idx,
    )
    displacement = (positions[mapping[1]] - positions[mapping[0]]
                    + torch.einsum("ni,nij->nj", shifts.to(positions), cells[systems]))
    distance = displacement.norm(dim=-1)
    keep = distance < config["cutoff"]
    ii, systems = mapping[0, keep], systems[keep]
    distance, displacement = distance[keep], displacement[keep]
    if torch.any(distance == 0):
        raise ValueError("Repulsive core encountered overlapping atoms (zero distance); "
                         "separate the atoms before running the model")
    u = config["strength"] * (config["cutoff"] / distance - 1.0)**2
    derivative = (-2.0 * config["strength"] * config["cutoff"]
                  * (config["cutoff"] / distance - 1.0) / distance**2)
    pair_force = derivative[:, None] * displacement / distance[:, None]
    energies = positions.new_zeros(len(positions)).index_add(0, ii, 0.5 * u)
    energy = positions.new_zeros(len(cells)).index_add(0, system_idx, energies)
    result = {"energy": energy, "free_energy": energy, "energies": energies,
              "forces": torch.zeros_like(positions).index_add(0, ii, pair_force)}
    if {"stress", "stresses"} & properties:
        volumes = torch.linalg.det(cells).abs()
        if torch.any(volumes <= 0):
            raise ValueError("Repulsive-core stress requires nonzero three-dimensional cells")
        virial = 0.5 * torch.einsum("ni,nj->nij", displacement, pair_force)
        stresses = positions.new_zeros((len(positions), 3, 3)).index_add(0, ii, virial)
        stresses = stresses / volumes[system_idx, None, None]
        result["stresses"] = stresses
        result["stress"] = positions.new_zeros((len(cells), 3, 3)).index_add(
            0, system_idx, stresses)
    return result


def wrap_torch_model(model, config=None):
    """Add the optional core to a torch-sim model, preserving its properties.

    The wrapper exposes ``base_model`` for independent-model spot checks.
    No torch dependency is imported when the feature is disabled.
    """
    config = validate_repulsive_core_config(config)
    if not config["enabled"]:
        return model
    if getattr(model, "_amorphgen_repulsive_core", False):
        if model.repulsive_core_config == config:
            return model
        model = model.base_model
    from torch_sim.models.interface import ModelInterface

    class RepulsiveCoreModel(ModelInterface):
        _amorphgen_repulsive_core = True

        def __init__(self):
            super().__init__()
            self.base_model = model
            self.repulsive_core_config = config

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(super().__getattr__("base_model"), name)

        @property
        def device(self):
            return self.base_model.device

        @property
        def dtype(self):
            return self.base_model.dtype

        @property
        def compute_forces(self):
            return self.base_model.compute_forces

        @compute_forces.setter
        def compute_forces(self, value):
            self.base_model.compute_forces = value

        @property
        def compute_stress(self):
            return self.base_model.compute_stress

        @compute_stress.setter
        def compute_stress(self, value):
            self.base_model.compute_stress = value

        @property
        def memory_scales_with(self):
            scaling = self.base_model.memory_scales_with
            return "n_atoms_x_density" if scaling == "n_atoms" else scaling

        def forward(self, state, **kwargs):
            self.last_base_results = self.base_model(state, **kwargs)
            result = dict(self.last_base_results)
            correction = _torch_core(state, self.repulsive_core_config, result.keys())
            retain_graph = getattr(self.base_model, "retain_graph", False)
            for prop in result.keys() & correction.keys():
                extra = correction[prop].to(result[prop])
                if not retain_graph:
                    extra = extra.detach()
                result[prop] = result[prop] + extra
            return result

    return RepulsiveCoreModel()
