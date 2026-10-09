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
