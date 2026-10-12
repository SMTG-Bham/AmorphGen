"""Small shared test setup helpers, independent of production validators."""

import sys

import numpy as np
from ase.build import bulk
from ase.data import atomic_masses, atomic_numbers
from ase.neighborlist import neighbor_list


def run_cli(monkeypatch, arguments):
    """Run the real CLI in-process, preserving its return value and exceptions."""
    from amorphgen.cli import main

    monkeypatch.setattr(sys, "argv", ["amorphgen", *map(str, arguments)])
    return main()


def rattled_cu(seed, scale=1.05):
    """Fresh 32-atom copper specimen used by the CPU and GPU engine tests."""
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    atoms.rattle(0.15, seed=seed)
    atoms.set_cell(atoms.cell * scale, scale_atoms=True)
    return atoms


def composition_symbols(composition):
    return [symbol for symbol, count in composition.items() for _ in range(count)]


def minsep_floors(composition):
    """The minsep table generate_random builds for this composition."""
    from amorphgen.utils.radii import auto_target_cn, default_minsep

    target_cn, _ = auto_target_cn(composition)
    return default_minsep(composition_symbols(composition), target_cn=target_cn)


def shortest_pair_distances(atoms, cutoff):
    """Independent crystal-contact measurement with an explicit search radius."""
    symbols = np.array(atoms.get_chemical_symbols())
    i, j, distances = neighbor_list("ijd", atoms, cutoff)
    shortest = {}
    for a, b, distance in zip(symbols[i], symbols[j], distances):
        key = "-".join(sorted((a, b)))
        shortest[key] = min(shortest.get(key, np.inf), distance)
    return shortest


def estimated_density(composition):
    """Independent mass/volume oracle for the estimated cubic cell (g/cm³)."""
    from amorphgen.utils.radii import estimate_cell_length

    length = estimate_cell_length(composition)
    mass = sum(atomic_masses[atomic_numbers[e]] * n for e, n in composition.items())
    return mass * 1.66054 / length**3
