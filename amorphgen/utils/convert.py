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

from ase.io import read, write

from .structure_io import (STRUCTURE_FORMATS as _FORMAT_MAP, STRUCTURE_PATTERNS,
                           write_sorted_vasp)


def _gather_inputs(input_path: str) -> list[str]:
    """Return a sorted list of structure files implied by ``input_path``."""
    if os.path.isdir(input_path):
        files: list[str] = []
        for pattern in STRUCTURE_PATTERNS:
            files += _glob.glob(os.path.join(input_path, pattern))
        return sorted(set(files))
    if os.path.isfile(input_path):
        return [input_path]
    raise FileNotFoundError(
        f"convert: input path '{input_path}' does not exist")


def _plan_outputs(files: list[str], output_dir: str, ext: str) -> list[str]:
    """Name and validate all outputs before writing any of them."""
    source_paths = {os.path.realpath(path) for path in files}
    source_ids = {(stat.st_dev, stat.st_ino) for stat in map(os.stat, files)}
    used_bases: set[str] = set()
    destinations: list[str] = []
    output_paths: set[str] = set()
    output_ids: set[tuple[int, int]] = set()
    for path in files:
        base, in_ext = os.path.splitext(os.path.basename(path))
        # s.xyz, s.cif and s_xyz.cif must all receive distinct names.
        candidate = base if base not in used_bases else f"{base}_{in_ext.lstrip('.')}"
        suffix = 2
        unique_base = candidate
        while unique_base in used_bases:
            unique_base = f"{candidate}_{suffix}"
            suffix += 1
        used_bases.add(unique_base)
        dest = os.path.join(output_dir, unique_base + ext)
        resolved = os.path.realpath(dest)
        try:
            stat = os.stat(dest)
            dest_id = (stat.st_dev, stat.st_ino)
        except FileNotFoundError:
            dest_id = None
        if resolved in source_paths or dest_id in source_ids:
            raise ValueError(
                f"convert: output '{dest}' would overwrite an input file. "
                "Choose a different output directory with --work-dir or output_dir.")
        if resolved in output_paths or dest_id in output_ids:
            raise ValueError(
                f"convert: output '{dest}' aliases another output file. "
                "Choose a different output directory with --work-dir or output_dir.")
        output_paths.add(resolved)
        if dest_id is not None:
            output_ids.add(dest_id)
        destinations.append(dest)
    return destinations


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

    Raises
    ------
    ValueError
        If an output would overwrite any input, including through a symbolic
        or hard link, or if two outputs alias the same file. No files are
        written when such a collision is found.

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
    destinations = _plan_outputs(files, output_dir, ext)
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n[Convert] {len(files)} file(s) -> "
              f"{output_dir}/  (format: {output_format})")

    written: list[str] = []
    for f, dest in zip(files, destinations):
        atoms = read(f)
        if ase_format == "vasp" and sort:
            write_sorted_vasp(dest, atoms)
        else:
            write(dest, atoms, format=ase_format)
        written.append(dest)
        if verbose:
            print(f"  {os.path.basename(f)}  ->  {os.path.basename(dest)}")

    if verbose:
        print(f"[Convert] Done — wrote {len(written)} file(s) to {output_dir}/")
    return written
