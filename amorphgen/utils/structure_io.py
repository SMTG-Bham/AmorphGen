"""Structure format names and the ensemble CLI's input discovery policy."""

import glob
import os


# Format key -> (ASE format, file extension). Callers own fallback/error policy.
STRUCTURE_FORMATS = {
    "xyz": ("extxyz", ".xyz"),
    "extxyz": ("extxyz", ".xyz"),
    "vasp": ("vasp", ".vasp"),
    "cif": ("cif", ".cif"),
}
SNAPSHOT_FORMATS = {**STRUCTURE_FORMATS, "traj": ("traj", ".traj")}
STRUCTURE_PATTERNS = ("*.xyz", "*.extxyz", "*.vasp", "*.cif", "POSCAR*")


def write_sorted_vasp(path, atoms):
    """Write the shared Cartesian POSCAR convention without changing atoms.

    Retain the historical atomic-number presort followed by ASE's symbol
    sorting, including its within-species ordering and constraint permutation.
    Torch-sim's direct-coordinate writer has a different convention.
    """
    from ase.io import write

    ordered = atoms[atoms.numbers.argsort()]
    write(path, ordered, format="vasp", sort=True)


def first_structure_files(directory):
    """Return the sorted first nonempty format group used by ensemble modes.

    Conversion intentionally gathers every matching format instead: mixing
    those policies would silently change ensemble membership.
    """
    for pattern in STRUCTURE_PATTERNS:
        files = sorted(glob.glob(os.path.join(directory, pattern)))
        if files:
            return files
    return []
