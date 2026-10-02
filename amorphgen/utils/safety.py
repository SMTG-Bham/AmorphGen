"""Stateful checks for invalid or physically unstable MLIP trajectories.

The limits are conservative tripwires, not a test of model accuracy.  Set an
individual physical limit to ``None`` to disable it; finite-state checks always
remain active.  A monitor belongs to one continuous MD or relaxation segment.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral, Real

import numpy as np
from ase import units
from ase.geometry import minkowski_reduce
from ase.neighborlist import neighbor_list

from .common import DivergenceError

DEFAULT_SAFETY_CONFIG = {
    "min_distance": 0.5,
    "max_energy_jump_per_atom": 10.0,
    "max_temperature": 100000.0,
    "min_volume_ratio": 0.2,
    "max_volume_ratio": 5.0,
    "reference": None,
}

_REFERENCE_DEFAULTS = {
    "interval": 100,
    "max_force_rmse": 1.0,
    "max_energy_difference_per_atom": None,
}
_REFERENCE_MODEL_KEYS = {"model", "model_path", "device", "default_dtype"}


def _positive_or_none(value, name):
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number or null")


def validate_safety_config(config=None, *, reference_calc=None):
    """Return a validated copy of the safety mapping, including defaults.

    ``reference`` is optional and needs a model name or model path unless a
    calculator is injected. An empty reference mapping disables spot checks.
    """
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("safety must be a mapping")  # noqa: TRY004 - uniform configuration errors
    unknown = set(config) - set(DEFAULT_SAFETY_CONFIG)
    if unknown:
        raise ValueError(f"Unknown safety option(s): {', '.join(sorted(unknown))}")
    result = deepcopy(DEFAULT_SAFETY_CONFIG)
    result.update(deepcopy(dict(config)))
    for name in DEFAULT_SAFETY_CONFIG.keys() - {"reference"}:
        _positive_or_none(result[name], f"safety.{name}")
    lower, upper = result["min_volume_ratio"], result["max_volume_ratio"]
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("safety.min_volume_ratio must not exceed max_volume_ratio")

    reference = result["reference"]
    if reference is not None and not isinstance(reference, Mapping):
        raise ValueError("safety.reference must be a mapping or null")
    if not reference and reference_calc is None:
        result["reference"] = None
        return result
    reference = dict(reference or {})
    unknown = set(reference) - set(_REFERENCE_DEFAULTS) - _REFERENCE_MODEL_KEYS
    if unknown:
        raise ValueError(
            f"Unknown safety.reference option(s): {', '.join(sorted(unknown))}"
        )
    for name in _REFERENCE_MODEL_KEYS:
        value = reference.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"safety.reference.{name} must be a nonempty string")
    if reference_calc is None and not (
        reference.get("model") or reference.get("model_path")
    ):
        raise ValueError("safety.reference requires model or model_path")
    merged = {**_REFERENCE_DEFAULTS, **reference}
    interval = merged["interval"]
    if isinstance(interval, bool) or not isinstance(interval, Integral) or interval < 1:
        raise ValueError("safety.reference.interval must be a positive integer")
    for name in ("max_force_rmse", "max_energy_difference_per_atom"):
        _positive_or_none(merged[name], f"safety.reference.{name}")
    result["reference"] = merged
    return result


class SafetyMonitor:
    """Check finite state, physical limits, and optional model disagreement.

    Energies are in eV, distances in angstrom, temperatures in K, and force
    RMSE in eV/angstrom. Energy jumps compare consecutive successful checks;
    volume ratios use the first valid cell. Reference checks happen on the
    first call and then at least ``interval`` steps apart. With no step number,
    the interval counts calls. Calculator exceptions propagate unchanged.
    """

    def __init__(self, config=None, context="", reference_calc=None):
        self.config = validate_safety_config(config, reference_calc=reference_calc)
        self.context = context
        self.reference_calc = reference_calc
        self._initial_volume = None
        self._previous_energy_per_atom = None
        self._checks = 0
        self._last_reference_step = None
        self._contact_state = None

    def _fail(self, message, step):
        where = f" at step {step}" if step is not None else ""
        during = f" during {self.context}" if self.context else ""
        raise DivergenceError(
            f"{message}{where}{during} — the calculation has diverged or exceeded "
            "its safety limits. Check the structure and model validity, reduce "
            "the timestep or temperature, or adjust the safety limits for this system."
        )

    def check_geometry(self, atoms, step=None):
        """Check geometry and momenta before calling a calculator.

        Also checks temperature, close contacts and volume. Does not advance
        the energy baseline or reference cadence, so it is safe to use before
        an integrator or optimizer evaluates the first forces.
        """
        for name, values in (
            ("positions", atoms.positions),
            ("cell", atoms.cell.array),
            ("momenta", atoms.get_momenta()),
        ):
            if not np.isfinite(values).all():
                self._fail(f"Non-finite {name}", step)
        if not len(atoms):
            self._fail("Empty structure", step)

        periodic_cell = atoms.cell.array[atoms.pbc]
        if len(periodic_cell) and np.linalg.matrix_rank(periodic_cell) < len(periodic_cell):
            self._fail("Degenerate periodic cell", step)
        volume = abs(float(np.linalg.det(atoms.cell.array)))
        if not np.isfinite(volume):
            self._fail("Non-finite cell volume", step)
        if self._initial_volume is not None:
            ratio = volume / self._initial_volume
            lower, upper = self.config["min_volume_ratio"], self.config["max_volume_ratio"]
            if (lower is not None and ratio < lower) or (upper is not None and ratio > upper):
                self._fail(
                    f"Volume runaway: volume ratio {ratio:.6g} relative to the initial "
                    f"cell is outside limits [{lower}, {upper}]", step
                )

        # ASE defines temperature using unconstrained degrees of freedom. A
        # completely fixed, stationary structure has no thermal degrees of freedom.
        dof = atoms.get_number_of_degrees_of_freedom()
        kinetic_energy = atoms.get_kinetic_energy()
        temperature = (2 * kinetic_energy / (dof * units.kB) if dof > 0
                       else (0.0 if kinetic_energy == 0 else np.inf))
        if not np.isfinite(temperature):
            self._fail("Non-finite temperature", step)
        maximum = self.config["max_temperature"]
        if maximum is not None and temperature > maximum:
            self._fail(
                f"Temperature runaway: {temperature:.6g} K exceeds {maximum:g} K", step
            )

        minimum = self.config["min_distance"]
        if minimum is not None:
            self._check_contacts(atoms, minimum, step)
        if self._initial_volume is None and volume > 0:
            self._initial_volume = volume

    def _check_contacts(self, atoms, minimum, step):
        # The pre-evaluation and post-evaluation guards often see identical
        # geometry. Reuse only a successful contact search; finite state,
        # temperature and volume remain checked on every call. Copies matter:
        # ASE integrators commonly update positions or the cell in place.
        previous = self._contact_state
        if previous is not None and previous[3] == minimum and all(
            np.array_equal(old, current)
            for old, current in zip(previous[:3], (atoms.positions, atoms.cell.array, atoms.pbc))
        ):
            return
        # Check self-images first, including short lattice combinations in
        # skew cells. This also avoids huge neighbour lists in collapsed cells.
        if atoms.pbc.any():
            reduced, _ = minkowski_reduce(atoms.cell.array, pbc=atoms.pbc)
            shortest = np.linalg.norm(reduced[atoms.pbc], axis=1).min()
            if shortest < minimum:
                self._fail(
                    f"Close contact with a periodic self-image: {shortest:.6g} "
                    f"angstrom < min_distance {minimum:g} angstrom", step
                )
        indices_i, indices_j, distances = neighbor_list("ijd", atoms, minimum)
        if len(distances):
            closest = int(np.argmin(distances))
            self._fail(
                f"Close contact between atoms {indices_i[closest]} and "
                f"{indices_j[closest]}: {distances[closest]:.6g} angstrom "
                f"< min_distance {minimum:g} angstrom", step
            )
        self._contact_state = (
            atoms.positions.copy(), atoms.cell.array.copy(), atoms.pbc.copy(), minimum
        )

    def check(self, atoms, step=None):
        """Raise :class:`DivergenceError` for an invalid current state."""
        self.check_geometry(atoms, step)
        energy = atoms.get_potential_energy(apply_constraint=False)
        forces = atoms.get_forces(apply_constraint=False)
        self._check_results(energy, forces, atoms.calc, step)
        energy_per_atom = float(energy) / len(atoms)
        maximum = self.config["max_energy_jump_per_atom"]
        if maximum is not None and self._previous_energy_per_atom is not None:
            jump = abs(energy_per_atom - self._previous_energy_per_atom)
            if jump > maximum:
                self._fail(
                    f"Energy jump: {jump:.6g} eV/atom exceeds "
                    f"max_energy_jump_per_atom {maximum:g} eV/atom", step
                )

        reference = self.config["reference"]
        current = step if isinstance(step, Real) else self._checks
        if reference is not None and (
            self._last_reference_step is None
            or current - self._last_reference_step >= reference["interval"]
            or current < self._last_reference_step
        ):
            self._check_reference(atoms, energy, forces, step)
            self._last_reference_step = current
        self._previous_energy_per_atom = energy_per_atom
        self._checks += 1

    def _check_results(self, energy, forces, calc, step, prefix=""):
        bad = []
        if not np.isfinite(energy).all():
            bad.append("potential energy")
        if not np.isfinite(forces).all():
            bad.append("forces")
        # Do not request optional properties from models which cannot supply
        # them or trigger extra evaluations. Force-consistent optimizers use
        # free_energy, so a finite ordinary energy alone is insufficient.
        cached = getattr(calc, "results", {})
        for name, label in (("free_energy", "free energy"), ("stress", "stress")):
            values = cached.get(name)
            if values is not None and not np.isfinite(values).all():
                bad.append(label)
        if bad:
            self._fail(f"Non-finite {prefix}{' and '.join(bad)}", step)

    def _check_reference(self, atoms, energy, forces, step):
        reference = self.config["reference"]
        if self.reference_calc is None:
            from .calculators import get_calculator

            kwargs = {name: reference[name] for name in _REFERENCE_MODEL_KEYS
                      if reference.get(name) is not None}
            self.reference_calc = get_calculator(**kwargs)
        base = getattr(atoms.calc, "base_calculator", None)
        if self.reference_calc is atoms.calc or self.reference_calc is base:
            raise ValueError("safety.reference must use an independent calculator")
        comparison = atoms.copy()
        comparison.set_constraint()
        comparison.calc = self.reference_calc
        reference_energy = comparison.get_potential_energy(apply_constraint=False)
        reference_forces = comparison.get_forces(apply_constraint=False)
        self._check_results(reference_energy, reference_forces, comparison.calc,
                            step, prefix="reference ")

        # A stabilizing core is not model disagreement. Compare the underlying
        # MLIP while still requiring the total wrapped results to be finite.
        if base is not None:
            energy = base.get_potential_energy(atoms)
            forces = base.get_forces(atoms)
            self._check_results(energy, forces, base, step, prefix="primary model ")
        force_rmse = float(np.sqrt(np.mean((forces - reference_forces) ** 2)))
        maximum = reference["max_force_rmse"]
        if maximum is not None and force_rmse > maximum:
            self._fail(
                f"Reference model disagreement: force RMSE {force_rmse:.6g} "
                f"eV/angstrom exceeds {maximum:g} eV/angstrom", step
            )
        energy_difference = abs(float(energy) - float(reference_energy)) / len(atoms)
        maximum = reference["max_energy_difference_per_atom"]
        if maximum is not None and energy_difference > maximum:
            self._fail(
                f"Reference model disagreement: energy difference "
                f"{energy_difference:.6g} eV/atom exceeds {maximum:g} eV/atom", step
            )
