"""
amorphgen.utils.lammps_potential
--------------------------------
Any LAMMPS pair style as an ASE calculator, through ASE's ``LAMMPSlib``.

``lammps_setup`` turns the ``lammps_params`` block (YAML, or the CLI's
``--pair-style`` / ``--pair-coeff`` / ``--lammps-elements``) into LAMMPSlib
arguments without importing LAMMPS, so the CLI can check it before any work
starts. ``LAMMPSCalculator`` is LAMMPSlib with three fixes for AmorphGen's
stages:

* the element → LAMMPS type map is always explicit. LAMMPSlib otherwise
  numbers types in order of first appearance in the first structure it sees,
  which silently swaps the parameters of ``pair_coeff * * file Si O`` when a
  structure starts with O;
* potential files are made absolute when the calculator is built, because
  LAMMPSlib only runs the pair commands on the first calculation, after the
  pipeline has changed into the run directory;
* a deep copy returns the calculator itself (see ``__deepcopy__``).
"""

from __future__ import annotations

import ctypes
import os
import sys

from ase.calculators.calculator import all_changes
from ase.calculators.lammpslib import LAMMPSlib
from ase.data import atomic_masses, atomic_numbers, chemical_symbols

_SYMBOLS = frozenset(chemical_symbols[1:])

LAMMPS_PARAM_KEYS = frozenset({
    "pair_style", "pair_coeff", "elements", "commands", "masses",
    "lammps_header", "amendments", "log_file",
})


def _as_list(value, key: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ValueError(f"lammps_params.{key} must be a string or a list of strings")


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _absolutise(line: str, files: list[str], skip: int) -> str:
    """Make every token of *line* that names an existing file absolute.

    The first *skip* tokens (style or command name, type ranges), element
    symbols, ``NULL``, ``*`` and numbers are never treated as files, even if
    a file of that name exists in the current directory.
    """
    tokens = line.split()
    for i, token in enumerate(tokens):
        if (i < skip or token in _SYMBOLS or token in ("NULL", "*") or _is_number(token)
                or not os.path.isfile(token)):
            continue
        tokens[i] = os.path.abspath(token)
        if tokens[i] not in files:
            files.append(tokens[i])
    return " ".join(tokens)


def _infer_elements(coeffs: list[str]) -> list[str] | None:
    """Element of each LAMMPS type from the ``* *`` pair_coeff lines.

    The trailing element symbols of ``pair_coeff * * Si.sw Si`` name the
    types in order; hybrid styles give one line per sub-style, with ``NULL``
    for the types that sub-style does not handle. Returns None when no line
    names its elements this way (e.g. ``pair_coeff 1 1 0.01 2.3``).
    """
    merged = None
    for line in coeffs:
        tokens = line.split()
        if tokens[:2] != ["*", "*"]:
            continue
        tail = []
        for token in reversed(tokens[2:]):
            if token != "NULL" and token not in _SYMBOLS:
                break
            tail.append(token)
        tail.reverse()
        if not tail or all(t == "NULL" for t in tail):
            continue
        if merged is None:
            merged = tail
            continue
        if len(tail) != len(merged):
            raise ValueError(
                f"pair_coeff lines name different numbers of LAMMPS types: "
                f"{' '.join(merged)} vs {' '.join(tail)}")
        for i, (old, new) in enumerate(zip(merged, tail)):
            if new == "NULL":
                continue
            if old not in ("NULL", new):
                raise ValueError(
                    f"pair_coeff lines disagree on LAMMPS type {i + 1}: "
                    f"{old} vs {new}")
            merged[i] = new
    if merged is not None and "NULL" in merged:
        i = merged.index("NULL")
        raise ValueError(
            f"LAMMPS type {i + 1} is NULL in every pair_coeff line; give the "
            f"type order with lammps_params.elements (--lammps-elements)")
    return merged


def lammps_setup(params: dict | None) -> dict:
    """Validate a ``lammps_params`` block and build the LAMMPSlib arguments.

    Pure: LAMMPS itself is not imported, so this also serves as the CLI's
    fail-fast check.

    Parameters
    ----------
    params : dict
        ``pair_style`` (str) and ``pair_coeff`` (str or list) are required.
        Optional: ``elements`` (element of each LAMMPS type, in type order;
        inferred from ``pair_coeff * * <file> El1 El2 ...`` when omitted),
        ``commands`` (more LAMMPS commands run after the pair_coeff lines),
        ``masses`` (element → amu), ``lammps_header`` and ``amendments``
        (passed to LAMMPSlib unchanged) and ``log_file``.

    Returns
    -------
    dict
        ``lmpcmds``, ``elements``, ``atom_types``, ``files`` (absolute paths
        of the potential files found in the commands), ``masses``,
        ``lammps_header``, ``amendments`` and ``log_file``.
    """
    if not params:
        raise ValueError(
            "Model 'lammps' needs a pair style: --pair-style and --pair-coeff, "
            "or a lammps_params block in the YAML config.")
    unknown = sorted(set(params) - LAMMPS_PARAM_KEYS)
    if unknown:
        raise ValueError(f"Unknown lammps_params key(s): {', '.join(unknown)}")
    style = params.get("pair_style")
    if not isinstance(style, str) or not style.strip():
        raise ValueError("lammps_params.pair_style must be a non-empty string")
    coeffs = _as_list(params.get("pair_coeff"), "pair_coeff")
    if not coeffs:
        raise ValueError("lammps_params.pair_coeff is required")
    commands = _as_list(params.get("commands"), "commands")

    # Accept the lines with or without their leading LAMMPS command name.
    style = style.strip().removeprefix("pair_style").strip()
    coeffs = [c.strip().removeprefix("pair_coeff").strip() for c in coeffs]

    files: list[str] = []
    style = _absolutise(style, files, skip=1)
    coeffs = [_absolutise(c, files, skip=2) for c in coeffs]
    commands = [_absolutise(c, files, skip=1) for c in commands]

    inferred = _infer_elements(coeffs)
    given = params.get("elements")
    if given is not None:
        if isinstance(given, str):
            given = [e.strip() for e in given.split(",") if e.strip()]
        if (not isinstance(given, (list, tuple)) or not given
                or not all(isinstance(e, str) and e in _SYMBOLS for e in given)):
            raise ValueError(
                f"lammps_params.elements must list element symbols, got {given!r}")
        if len(set(given)) != len(given):
            raise ValueError(f"lammps_params.elements repeats an element: {list(given)}")
        if inferred is not None and list(given) != inferred:
            raise ValueError(
                f"lammps_params.elements {list(given)} disagrees with the type "
                f"order of the pair_coeff lines {inferred}")
        elements = list(given)
    elif inferred is not None:
        elements = inferred
    else:
        raise ValueError(
            "Cannot tell which element each LAMMPS type is from the pair_coeff "
            "lines; give the type order with lammps_params.elements "
            "(--lammps-elements Si,O means type 1 = Si, type 2 = O)")

    masses = params.get("masses") or None
    if masses is not None:
        if not isinstance(masses, dict) or set(masses) - set(elements):
            raise ValueError(
                f"lammps_params.masses must map elements of {elements} to amu")
        masses = {el: float(m) for el, m in masses.items()}

    log_file = params.get("log_file")
    if log_file is not None:
        log_file = os.path.abspath(log_file)

    return {
        "lmpcmds": [f"pair_style {style}"]
                   + [f"pair_coeff {c}" for c in coeffs] + commands,
        "elements": elements,
        "atom_types": {el: i + 1 for i, el in enumerate(elements)},
        "files": files,
        "masses": masses,
        "lammps_header": (_as_list(params["lammps_header"], "lammps_header")
                          if params.get("lammps_header") else None),
        "amendments": (_as_list(params["amendments"], "amendments")
                       if params.get("amendments") else None),
        "log_file": log_file,
    }


def _preload_pip_mpi() -> bool:
    """Load the pip ``mpich`` wheel's libmpi so the PyPI LAMMPS wheel starts.

    The PyPI ``lammps`` wheel links ``libmpi.so.12`` from the ``mpich`` wheel,
    which installs it to ``<prefix>/lib``, outside liblammps's search path.
    Only called after LAMMPS failed to load, so a LAMMPS built against another
    MPI never gets MPICH loaded next to it.
    """
    path = os.path.join(sys.prefix, "lib", "libmpi.so.12")
    if not os.path.isfile(path):
        return False
    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    return True


class LAMMPSCalculator(LAMMPSlib):
    """LAMMPSlib with an explicit type map and absolute potential paths.

    Build it with :func:`make_lammps_calculator`. ``elements`` lists the
    element of each LAMMPS type and ``potential_files`` the potential files
    its commands read (hashed into the run manifest).
    """

    def __init__(self, *, potential_files=(), **kwargs):
        super().__init__(**kwargs)
        self.elements = list(self.parameters.atom_types)
        self.potential_files = list(potential_files)

    def start_lammps(self):
        try:
            super().start_lammps()
        except OSError as exc:
            if "libmpi" not in str(exc) or not _preload_pip_mpi():
                raise
            super().start_lammps()

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        if atoms is not None:
            missing = sorted(set(atoms.get_chemical_symbols()) - set(self.elements))
            if missing:
                raise ValueError(
                    f"The LAMMPS potential covers {', '.join(self.elements)}, "
                    f"not {', '.join(missing)}")
        super().calculate(atoms, properties, system_changes)

    def __deepcopy__(self, memo):
        # The stages deep-copy their input Atoms, and the attached calculator
        # with them; a started LAMMPS instance holds a C pointer that cannot
        # be copied. The stages then attach the shared calculator anyway.
        return self


def make_lammps_calculator(params: dict) -> LAMMPSCalculator:
    """Build a :class:`LAMMPSCalculator` from a ``lammps_params`` block."""
    setup = lammps_setup(params)
    kwargs = {"lmpcmds": setup["lmpcmds"], "atom_types": setup["atom_types"],
              "keep_alive": True}
    if setup["masses"]:
        kwargs["atom_type_masses"] = {
            el: setup["masses"].get(el, atomic_masses[atomic_numbers[el]])
            for el in setup["elements"]}
    for key in ("lammps_header", "amendments", "log_file"):
        if setup[key] is not None:
            kwargs[key] = setup[key]
    return LAMMPSCalculator(potential_files=setup["files"], **kwargs)
