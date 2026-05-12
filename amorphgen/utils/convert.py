"""
amorphgen.utils.convert
------------------------
Public file-format conversion utility.

Convert one structure file or every ASE-readable file in a directory
to a target format (xyz / vasp / cif).  VASP outputs are sorted by
species so the resulting POSCAR is clean.

Usable from CLI (``amorphgen --convert PATH --format vasp``), Python
API (``from amorphgen import convert``), or via a ``convert:`` block
in a YAML config file (``amorphgen --config convert.yaml``).
"""

from __future__ import annotations

import glob as _glob
import os
from typing import Iterable

from ase.io import read, write


# Format key → (ASE format string, file extension).  Mirrors the
# random_gen module's table; kept locally to avoid a circular import.
_FORMAT_MAP: dict[str, tuple[str, str]] = {
    "xyz":    ("extxyz", ".xyz"),
    "extxyz": ("extxyz", ".xyz"),
    "vasp":   ("vasp",   ".vasp"),
    "cif":    ("cif",    ".cif"),
}


def _gather_inputs(input_path: str) -> list[str]:
    """Return a sorted list of structure files implied by ``input_path``."""
    if os.path.isdir(input_path):
        files: list[str] = []
        for pattern in ("*.xyz", "*.extxyz", "*.vasp", "*.cif", "POSCAR*"):
            files += _glob.glob(os.path.join(input_path, pattern))
        return sorted(set(files))
    if os.path.isfile(input_path):
        return [input_path]
    raise FileNotFoundError(
        f"convert: input path '{input_path}' does not exist")


def convert(input_path: str,
            output_format: str = "vasp",
            output_dir: str | None = None,
            sort: bool = True,
            verbose: bool = True) -> list[str]:
    """Convert a structure file or directory of files to ``output_format``.

    Parameters
    ----------
    input_path : str
        Path to a single ASE-readable structure file, or to a directory
        containing one or more such files.  Directory globs match
        ``*.xyz``, ``*.extxyz``, ``*.vasp``, ``*.cif`` and ``POSCAR*``.
    output_format : str
        Target format key.  Allowed values: ``"xyz"`` / ``"extxyz"`` (both
        write ASE extended XYZ to ``.xyz``), ``"vasp"`` (POSCAR to
        ``.vasp``), ``"cif"``.
    output_dir : str, optional
        Directory to write converted files into.  If ``None``, defaults
        to ``"<input>_<format>"`` for directory inputs, or the parent
        directory of ``input_path`` for single-file inputs.
    sort : bool, default True
        For VASP outputs, sort atoms by species so the POSCAR is clean.
        Ignored for other formats.
    verbose : bool, default True
        Print one progress line per file plus a summary footer.

    Returns
    -------
    list of str
        Paths to the converted output files, in input order.

    Examples
    --------
    >>> from amorphgen import convert
    >>> convert("snapshots/", output_format="vasp",
    ...         output_dir="snapshots_vasp/")
    ['snapshots_vasp/snapshot_0000_frame00000.vasp', ...]
    """
    if output_format not in _FORMAT_MAP:
        raise ValueError(
            f"convert: unknown format '{output_format}'. "
            f"Choices: {sorted(_FORMAT_MAP)}")
    ase_format, ext = _FORMAT_MAP[output_format]

    files = _gather_inputs(input_path)
    if not files:
        raise FileNotFoundError(
            f"convert: no structure files found in '{input_path}/' "
            f"(looked for *.xyz, *.extxyz, *.vasp, *.cif, POSCAR*)")

    # Resolve default output directory.
    if output_dir is None:
        if os.path.isdir(input_path):
            output_dir = f"{input_path.rstrip('/')}_{output_format}"
        else:
            output_dir = os.path.dirname(input_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n[Convert] {len(files)} file(s) -> "
              f"{output_dir}/  (format: {output_format})")

    written: list[str] = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        atoms = read(f)
        dest = os.path.join(output_dir, base + ext)
        if ase_format == "vasp" and sort:
            atoms = atoms[atoms.numbers.argsort()]
            write(dest, atoms, format=ase_format, sort=True)
        else:
            write(dest, atoms, format=ase_format)
        written.append(dest)
        if verbose:
            print(f"  {os.path.basename(f)}  ->  {os.path.basename(dest)}")

    if verbose:
        print(f"[Convert] Done — wrote {len(written)} file(s) to {output_dir}/")
    return written
