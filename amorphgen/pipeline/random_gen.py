"""
amorphgen.pipeline.random_gen
------------------------------
Generate amorphous structures by random atom placement with
minimum-separation constraints (AIRSS-style), then optionally
relax with a foundation model.

This provides an alternative to the melt-and-quench route: instead of
melting a crystal and cooling it, we place atoms randomly inside a box
subject to pairwise distance constraints, then optimise the structure.
"""

from __future__ import annotations

import os
import numpy as np
from ase import Atoms
from ase.io import write
from ase.data import covalent_radii, atomic_numbers


# ── Default minimum separations based on covalent radii ───────────────────────

def _default_minsep(symbols: list[str], scale: float = 0.85) -> dict:
    """Build a minsep dict from covalent radii with a safety scale."""
    unique = sorted(set(symbols))
    minsep = {}
    for i, s1 in enumerate(unique):
        for s2 in unique[i:]:
            r1 = covalent_radii[atomic_numbers[s1]]
            r2 = covalent_radii[atomic_numbers[s2]]
            key = f"{s1}-{s2}" if s1 <= s2 else f"{s2}-{s1}"
            minsep[key] = (r1 + r2) * scale
    return minsep


def _get_minsep(s1: str, s2: str, minsep: dict) -> float:
    """Look up the minimum separation for a pair of species."""
    key1 = f"{s1}-{s2}"
    key2 = f"{s2}-{s1}"
    return minsep.get(key1, minsep.get(key2, 1.5))


def _estimate_cell_length(composition: dict, target_density: float | None = None,
                          packing_factor: float = 0.6) -> float:
    """Estimate cubic cell length from composition and target density."""
    from ase.data import atomic_masses as am
    total_mass = sum(am[atomic_numbers[s]] * n for s, n in composition.items())
    if target_density is not None:
        # density in g/cm³ -> Å³
        vol_cm3 = (total_mass / 6.022e23) / target_density
        vol_A3 = vol_cm3 * 1e24
    else:
        # Rough estimate from atomic volumes
        n_atoms = sum(composition.values())
        vol_A3 = n_atoms * 20.0 / packing_factor  # ~20 ų per atom
    return vol_A3 ** (1.0 / 3.0)


def generate_random(
    composition: dict[str, int],
    cell_length_ang: float | None = None,
    target_density: float | None = None,
    minsep: dict[str, float] | None = None,
    minsep_scale: float = 0.85,
    seed: int | None = None,
    max_attempts_per_atom: int = 200000,
    pbc: bool = True,
) -> Atoms:
    """
    Generate a single random structure.

    Parameters
    ----------
    composition : dict
        e.g. {"In": 32, "O": 48}
    cell_length_ang : float, optional
        Cubic cell edge length in Å.  If None, estimated from
        target_density or atomic volumes.
    target_density : float, optional
        Target density in g/cm³ for cell size estimation.
    minsep : dict, optional
        Minimum pair separations, e.g. {"In-In": 2.8, "In-O": 1.9}.
        If None, defaults are computed from covalent radii.
    minsep_scale : float
        Scale factor for default minsep (ignored if minsep is provided).
    seed : int, optional
        Random seed for reproducibility.
    max_attempts_per_atom : int
        Max placement attempts per atom before raising an error.
    pbc : bool
        Periodic boundary conditions.

    Returns
    -------
    ase.Atoms
    """
    rng = np.random.default_rng(seed)

    # Build atom list
    symbols = []
    for species, count in composition.items():
        symbols.extend([species] * count)
    n_atoms = len(symbols)

    # Shuffle for random ordering
    rng.shuffle(symbols)

    # Cell size
    if cell_length_ang is None:
        cell_length_ang = _estimate_cell_length(composition, target_density)
    L = cell_length_ang

    # Minimum separations
    if minsep is None:
        minsep = _default_minsep(symbols, scale=minsep_scale)

    # Place atoms one by one (vectorised distance checks)
    positions = np.empty((n_atoms, 3))
    placed_symbols = []
    n_placed = 0

    for i, sym in enumerate(symbols):
        # Pre-compute minsep thresholds for this species vs all placed
        if n_placed > 0:
            min_dists = np.array([_get_minsep(sym, ps, minsep)
                                  for ps in placed_symbols])

        placed = False
        for attempt in range(max_attempts_per_atom):
            pos = rng.random(3) * L

            if n_placed == 0:
                # First atom — always accept
                positions[0] = pos
                placed_symbols.append(sym)
                n_placed = 1
                placed = True
                break

            # Vectorised distance check: all placed atoms at once
            d = pos - positions[:n_placed]
            if pbc:
                d -= L * np.round(d / L)
            dists = np.sqrt(np.sum(d * d, axis=1))

            if np.all(dists >= min_dists):
                positions[n_placed] = pos
                placed_symbols.append(sym)
                n_placed += 1
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Could not place atom {i} ({sym}) after "
                f"{max_attempts_per_atom} attempts. "
                f"Try increasing cell_length_ang or reducing minsep."
            )

    atoms = Atoms(
        symbols=placed_symbols,
        positions=positions[:n_placed],
        cell=[L, L, L],
        pbc=pbc,
    )
    atoms.wrap()
    return atoms


def batch_random(
    composition: dict[str, int],
    n_structures: int = 10,
    output_dir: str = "random_structures",
    relax: bool = False,
    calc=None,
    fmax: float = 0.05,
    max_relax_steps: int = 200,
    **kwargs,
) -> list[str]:
    """
    Generate multiple random structures, optionally relaxing each.

    Parameters
    ----------
    composition : dict
    n_structures : int
    output_dir : str
    relax : bool
        If True and calc is provided, optimise each structure.
    calc : ASE calculator, optional
    fmax : float
    max_relax_steps : int
    **kwargs
        Forwarded to generate_random().

    Returns
    -------
    list of str — paths to output files
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    # Extract seed once — each structure gets seed+i for reproducibility
    base_seed = kwargs.pop("seed", None)

    for i in range(n_structures):
        seed_i = base_seed + i if base_seed is not None else None
        atoms = generate_random(composition, seed=seed_i, **kwargs)

        if relax and calc is not None:
            from ase.optimize import LBFGS
            from ase.filters import UnitCellFilter
            atoms.calc = calc
            ucf = UnitCellFilter(atoms)
            opt = LBFGS(ucf, logfile=None)
            opt.run(fmax=fmax, steps=max_relax_steps)

        fname = os.path.join(output_dir, f"random_{i:04d}.extxyz")
        write(fname, atoms, format="extxyz")
        paths.append(fname)
        formula = atoms.get_chemical_formula(mode="hill")
        print(f"  [{i+1}/{n_structures}] {formula} -> {fname}")

    print(f"\n  Generated {len(paths)} structures in {output_dir}/")
    return paths
