"""
amorphgen.utils.accelerator
---------------------------
Run MACE models through Symmetrix-XL without the user handling converted
model files.

    amorphgen POSCAR --model mace-mpa-0 --accelerator symmetrix-xl

The same model (a foundation name or a local ``.model``) is converted once
into a persistent *model store* and evaluated by the accelerated engine.

Accelerators
~~~~~~~~~~~~
* ``none``          plain MACE (default)
* ``symmetrix-xl``  bonan-group/symmetrix-xl
* ``auto``          Symmetrix-XL when it is installed, else plain MACE

The original Symmetrix (wcwitt/symmetrix) is not supported in this version.
It installs under the same import name (``symmetrix``), so the package is
identified by its distribution (pip) name.

Model store
~~~~~~~~~~~
``$AMORPHGEN_MODEL_STORE`` or ``~/.amorphgen/models``::

    symmetrix-xl-0.2.0/mace-mpa-0__<sha12>__1024.json
    index.json        # source hash, package, spline points, check results

A stored file holds every element of the model. It is reused while the source
model's contents (SHA-256), the package version, the spline points and the
head are unchanged. Conversion runs in a subprocess under a file lock, so
parallel jobs convert a model once.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ACCELERATORS = ("none", "auto", "symmetrix-xl")
FLAVOURS = ("symmetrix-xl",)
# The original Symmetrix (wcwitt/symmetrix): same import name, not supported.
ORIGINAL_SYMMETRIX = "symmetrix"
DEFAULT_SPLINE_POINTS = 1024
DEFAULT_TOLERANCE = 1.0        # meV/Angstrom, max force difference vs MACE

# Constructor options Symmetrix-XL accepts (checked before any work).
FLAVOUR_OPTIONS = {
    "symmetrix-xl": {"metal", "execution_profile", "streamed_edges",
                     "allow_fixed_workspace", "neighbor_skin"},
}

ACCELERATOR_KEYS = {"engine", "spline_points", "elements", "validate",
                    "tolerance", "convert", "store", "options",
                    "batch_max_atoms", "batch_skin"}

# Release series the joined (torch-sim) evaluation has been tested with. It
# uses private functions of the package, so a newer series is used with a
# warning; the result stays checked against MACE and against single calls.
TESTED_SERIES = {"symmetrix-xl": ("0.2",)}
_INSTALL = {
    "symmetrix-xl": 'pip install "symmetrix-xl[mace]>=0.2,<0.3"   (Linux x86-64 '
                    "wheels; build from source elsewhere)",
}


def untested_version(flavour: str, version: str) -> str | None:
    """A one-line warning when *version* is outside the tested series, else None."""
    tested = TESTED_SERIES.get(flavour, ())
    series = ".".join(version.split(".")[:2])
    if not tested or series in tested:
        return None
    return (f"[Accelerator] {flavour} {version} is outside the series tested with "
            f"this AmorphGen ({', '.join(t + '.x' for t in tested)}). Results are "
            f"still checked against MACE; with --engine torchsim the batch may fall "
            f"back to a slower route. Tested install: {_INSTALL[flavour].split('   ')[0]}")


class AcceleratorError(ValueError):
    """The requested accelerator cannot be used as configured."""


# ═════════════════════════════════════════════════════════════════════════════
# Settings
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class AcceleratorSettings:
    engine: str = "none"
    spline_points: int = DEFAULT_SPLINE_POINTS
    elements: list[str] | None = None
    validate: bool = True
    tolerance: float = DEFAULT_TOLERANCE
    convert: bool = True
    store: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    # --engine torchsim evaluates each batch in one joined Symmetrix call
    # (amorphgen.utils.torchsim_symmetrix): an upper bound on the atoms in one
    # call (GPU memory), and the skin (A) of the neighbour lists kept between
    # steps (amorphgen.utils.neighbor_cache; 0 builds a fresh list every step).
    batch_max_atoms: int | None = None
    batch_skin: float = 0.5

    @classmethod
    def from_config(cls, value) -> "AcceleratorSettings":
        """Accept ``None``, an engine name, or the YAML ``accelerator`` block."""
        if value is None:
            return cls()
        if isinstance(value, AcceleratorSettings):
            return value
        if isinstance(value, str):
            value = {"engine": value}
        if not isinstance(value, dict):
            raise AcceleratorError(
                f"accelerator must be a name or a mapping, got {type(value).__name__}")
        unknown = sorted(set(value) - ACCELERATOR_KEYS)
        if unknown:
            raise AcceleratorError(f"Unknown accelerator key(s): {', '.join(unknown)}")
        settings = cls(**{k: v for k, v in value.items() if v is not None})
        settings.engine = str(settings.engine).lower()
        if settings.engine not in ACCELERATORS:
            raise AcceleratorError(
                f"accelerator must be one of {', '.join(ACCELERATORS)}; "
                f"got '{settings.engine}'")
        if settings.elements is not None:
            settings.elements = sorted(set(settings.elements), key=_atomic_number)
        settings.options = dict(settings.options or {})
        if isinstance(settings.batch_skin, bool) or \
                not isinstance(settings.batch_skin, (int, float)) or settings.batch_skin < 0:
            raise AcceleratorError(
                "accelerator.batch_skin must be a length in Angstrom (0 or more); "
                f"got {settings.batch_skin!r}")
        return settings

    @property
    def enabled(self) -> bool:
        return self.engine != "none"


def _atomic_number(symbol: str) -> int:
    from ase.data import atomic_numbers
    try:
        return atomic_numbers[symbol]
    except KeyError:
        raise AcceleratorError(f"Unknown element '{symbol}' in accelerator.elements")


# ═════════════════════════════════════════════════════════════════════════════
# Which package is installed
# ═════════════════════════════════════════════════════════════════════════════

def installed_flavours() -> dict[str, str]:
    """``{distribution: version}`` of the Symmetrix packages installed
    (``symmetrix-xl``, and the unsupported original ``symmetrix``)."""
    import importlib.metadata as metadata
    found = {}
    for name in (*FLAVOURS, ORIGINAL_SYMMETRIX):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def resolve_flavour(engine: str) -> tuple[str, str] | None:
    """The (flavour, version) to use for *engine*, or None for plain MACE.

    ``auto`` uses Symmetrix-XL when it is installed and plain MACE otherwise;
    ``symmetrix-xl`` by name must be installed.
    """
    if engine == "none":
        return None
    installed = installed_flavours()
    original = installed.get(ORIGINAL_SYMMETRIX)
    if "symmetrix-xl" in installed:
        if original:
            raise AcceleratorError(
                "Both symmetrix and symmetrix-xl are installed in this environment; "
                "they share the 'symmetrix' import and cannot work together. "
                "Uninstall the original symmetrix.")
        return "symmetrix-xl", installed["symmetrix-xl"]
    have = (f"this environment has the original Symmetrix ({original}), which "
            f"this version does not support" if original
            else "symmetrix-xl is not installed")
    if engine == "auto":
        print(f"[Accelerator] auto: {have}; using plain MACE")
        return None
    raise AcceleratorError(
        f"--accelerator {engine} was requested, but {have}.\n"
        f"  Install it:   {_INSTALL[engine]}"
        + ("   (in a separate environment: the two cannot be installed together)"
           if original else ""))


def check_options(flavour: str, options: dict, dtype: str) -> None:
    """Refuse options that Symmetrix-XL does not accept."""
    for key in options:
        if key in FLAVOUR_OPTIONS[flavour]:
            continue
        if key == "use_kokkos":
            raise AcceleratorError(
                "accelerator option 'use_kokkos' is for the original Symmetrix, "
                "which this version does not support; Symmetrix-XL selects its "
                "backend with the SYMMETRIX_BACKEND environment variable")
        raise AcceleratorError(
            f"Unknown accelerator option '{key}' for {flavour}; accepted: "
            f"{', '.join(sorted(FLAVOUR_OPTIONS[flavour]))}")
    if options.get("metal"):
        if sys.platform != "darwin":
            raise AcceleratorError("accelerator option 'metal' needs macOS on Apple silicon")
        if dtype != "float32":
            raise AcceleratorError(
                "accelerator option 'metal' runs in float32 only; add "
                "--dtype float32 (default_dtype: float32)")


# ═════════════════════════════════════════════════════════════════════════════
# Source model and store
# ═════════════════════════════════════════════════════════════════════════════

def model_store(path: str | os.PathLike | None = None) -> Path:
    root = path or os.environ.get("AMORPHGEN_MODEL_STORE") or "~/.amorphgen/models"
    return Path(root).expanduser()


def split_head(model: str | None) -> tuple[str | None, str | None]:
    """``'mace-mh-1:omat_pbe'`` → (``'mace-mh-1'``, ``'omat_pbe'``); URLs are left whole."""
    if model and ":" in model and "://" not in model:
        name, head = model.split(":", 1)
        return name, head or None
    return model, None


def source_model(model: str | None, model_path: str | None) -> tuple[Path, str]:
    """The MACE ``.model`` file to convert and a readable label for it."""
    if model_path is not None:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MACE model file not found: {model_path}")
        return path.resolve(), _label(path.name)
    from .calculators import MACE_FOUNDATION_MODELS, _ci_get
    resolved = _ci_get(MACE_FOUNDATION_MODELS, model)
    if os.path.isfile(resolved):
        return Path(resolved).resolve(), _label(Path(resolved).name)
    try:
        from mace.calculators.foundations_models import download_mace_mp_checkpoint
    except ImportError:
        raise AcceleratorError(
            "Converting a model for the accelerator needs mace-torch "
            "(pip install \"amorphgen[mace]\").")
    return Path(download_mace_mp_checkpoint(resolved)).resolve(), _label(model)


def _label(name: str) -> str:
    stem = re.sub(r"\.(model|pt|ckpt)$", "", name)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-") or "model"


def file_sha256(path: Path) -> str:
    """SHA-256 of *path* (a retrained model with the same name gets a new hash)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stored_name(label: str, sha: str, spline_points: int, head: str | None) -> str:
    parts = [label, sha[:12], str(spline_points)]
    if head:
        parts.append(_label(head))
    return "__".join(parts) + ".json"


@contextlib.contextmanager
def _locked(path: Path):
    """Exclusive lock on *path* (POSIX flock); other processes wait."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as handle:
        try:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        except ImportError:          # pragma: no cover - non-POSIX
            pass
        try:
            yield
        finally:
            with contextlib.suppress(ImportError):
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)


def _read_index(store: Path) -> dict:
    try:
        return json.loads((store / "index.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _update_index(store: Path, key: str, entry: dict) -> None:
    with _locked(store / ".index.lock"):
        index = _read_index(store)
        index[key] = {**index.get(key, {}), **entry}
        tmp = store / f".index.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(index, indent=2, sort_keys=True))
        tmp.replace(store / "index.json")


# ═════════════════════════════════════════════════════════════════════════════
# Conversion
# ═════════════════════════════════════════════════════════════════════════════

def _converter_command(source: Path, output: Path, spline_points: int,
                       head: str | None) -> list[str]:
    # Run as a script, in a subprocess: it keeps the torch model out of this
    # process's memory. No element list: the export holds every element.
    script = ("import sys; from symmetrix.cli.extract_mace import main; "
              "sys.argv[0] = 'symmetrix_extract_mace'; main()")
    command = [sys.executable, "-c", script, "--model", str(source),
               "--num-spline-points", str(spline_points), "--output", str(output)]
    if head:
        command += ["--head", head]
    return command


def prepare(model: str | None = None, model_path: str | None = None, *,
            flavour: str, version: str, spline_points: int = DEFAULT_SPLINE_POINTS,
            store: str | None = None, convert: bool = True,
            force: bool = False) -> Path:
    """Return the stored, converted file for this model, converting if needed."""
    name, head = split_head(model)
    source, label = source_model(name, model_path)
    sha = file_sha256(source)
    root = model_store(store)
    folder = root / f"{flavour}-{version}"
    target = folder / stored_name(label, sha, spline_points, head)
    key = f"{folder.name}/{target.name}"

    with _locked(folder / f".{target.name}.lock"):
        if target.is_file() and not force:
            print(f"[Accelerator] Using stored model {target}")
            return target
        if not convert:
            raise AcceleratorError(
                f"No converted model for '{model_path or model}' ({flavour} "
                f"{version}) in {root}, and conversion is disabled "
                f"(accelerator.convert: false). Prepare it where conversion "
                f"is possible, with the same accelerator settings.")
        folder.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.{os.getpid()}.partial")
        command = _converter_command(source, partial, spline_points, head)
        print(f"[Accelerator] Converting {label} with {flavour} {version} "
              f"({spline_points} spline points) ...")
        start = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not partial.is_file():
            partial.unlink(missing_ok=True)
            tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-15:])
            raise AcceleratorError(
                f"Converting {source} with {flavour} {version} failed "
                f"(exit {result.returncode}):\n{tail}")
        partial.replace(target)
        seconds = time.perf_counter() - start
        size_mb = target.stat().st_size / 2**20
        print(f"[Accelerator] Saved {target} ({size_mb:.0f} MB, {seconds:.0f} s)")
        _update_index(root, key, {
            "source": str(source), "source_sha256": sha, "label": label,
            "flavour": flavour, "version": version, "spline_points": spline_points,
            "head": head, "size_bytes": target.stat().st_size,
            "converted": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
    return target


# ═════════════════════════════════════════════════════════════════════════════
# Check against MACE
# ═════════════════════════════════════════════════════════════════════════════

def test_structure(elements: list[str], n_atoms: int = 32, seed: int = 0):
    """A small periodic cell of *elements* on a jittered cubic lattice.

    The cell is the nearest cube, so the default gives 27 atoms.
    """
    import numpy as np
    from ase import Atoms
    rng = np.random.default_rng(seed)
    side = round(n_atoms ** (1 / 3))
    n = side ** 3
    spacing = 2.4
    grid = np.array([(i, j, k) for i in range(side) for j in range(side)
                     for k in range(side)], dtype=float) * spacing
    positions = grid + rng.normal(0.0, 0.05, grid.shape)
    symbols = [elements[i % len(elements)] for i in range(n)]
    return Atoms(symbols, positions=positions, cell=[side * spacing] * 3, pbc=True)


def check_against_mace(accelerated, mace, atoms) -> dict:
    """Energy, force and stress differences between two calculators on *atoms*."""
    import numpy as np
    a = atoms.copy(); a.calc = accelerated
    b = atoms.copy(); b.calc = mace
    n = len(atoms)
    return {
        "n_atoms": n,
        "dE_meV_per_atom": abs(a.get_potential_energy() - b.get_potential_energy()) / n * 1e3,
        "max_dF_meV_per_A": float(np.abs(a.get_forces() - b.get_forces()).max() * 1e3),
        "max_dS_GPa": float(np.abs(a.get_stress() - b.get_stress()).max() * 160.21766),
    }


def _check_key(dtype: str, elements: list[str], options: dict | None) -> str:
    """Index key of one check: the dtype, the elements and the options used.

    The options select the code path (``metal``, the execution
    profile), so a check made with other options does not cover this one.
    """
    key = f"check_{dtype}_{'-'.join(elements)}"
    if options:
        text = json.dumps(options, sort_keys=True, default=str)
        key += "_" + hashlib.sha256(text.encode()).hexdigest()[:8]
    return key


def _require_passed(result: dict, tolerance: float) -> None:
    """Raise unless the recorded force difference is within *tolerance*."""
    if result["max_dF_meV_per_A"] <= tolerance:
        return
    raise AcceleratorError(
        f"The accelerated model differs from MACE by "
        f"{result['max_dF_meV_per_A']:.3f} meV/A (tolerance {tolerance}). "
        f"Raise accelerator.spline_points, or set accelerator.tolerance "
        f"if this is expected (e.g. float32).")


def validate(json_path: Path, calc, *, model: str | None, model_path: str | None,
             elements: list[str], dtype: str, tolerance: float,
             store: str | None = None, options: dict | None = None) -> dict | None:
    """Compare *calc* with MACE (float64, CPU) once per stored file, dtype,
    element set and option set.

    A recorded check is not repeated, but its force difference is held to the
    current *tolerance* every time, so a failed check keeps refusing the run.
    """
    root = model_store(store)
    key = f"{json_path.parent.name}/{json_path.name}"
    check_key = _check_key(dtype, elements, options)
    previous = _read_index(root).get(key, {}).get(check_key)
    if previous is not None:
        _require_passed(previous, tolerance)
        return previous
    from .calculators import _load_mace
    name, head = split_head(model)
    mace_kwargs = {"default_dtype": "float64"}
    if head:
        mace_kwargs["head"] = head
    mace = _load_mace(name or "mace-mpa-0", device="cpu", model_path=model_path,
                      **mace_kwargs)
    result = check_against_mace(calc, mace, test_structure(elements))
    result["tolerance_meV_per_A"] = tolerance
    result["passed"] = result["max_dF_meV_per_A"] <= tolerance
    _update_index(root, key, {check_key: result})
    status = "OK" if result["passed"] else "FAILED"
    print(f"[Accelerator] Check vs MACE ({result['n_atoms']} atoms, {dtype}): "
          f"dE {result['dE_meV_per_atom']:.4f} meV/atom, "
          f"max dF {result['max_dF_meV_per_A']:.4f} meV/A, "
          f"max dS {result['max_dS_GPa']:.5f} GPa  [{status}]")
    _require_passed(result, tolerance)
    return result


def _validate_on_first_use(calc, check) -> None:
    """Run ``check(elements)`` before the first calculation on *calc*.

    Used when the elements are not known when the calculator is built (the
    Python API, or a mode whose input the command line cannot read ahead):
    they are taken from the first structure evaluated, so the check against
    MACE still runs instead of being skipped.
    """
    original = calc.calculate

    def calculate(atoms=None, *args, **kwargs):
        target = atoms if atoms is not None else getattr(calc, "atoms", None)
        if target is None:
            return original(atoms, *args, **kwargs)
        del calc.calculate                 # the check runs once, and uses calc
        check(sorted(set(target.get_chemical_symbols()), key=_atomic_number))
        return original(atoms, *args, **kwargs)

    calc.calculate = calculate


# ═════════════════════════════════════════════════════════════════════════════
# Fail-fast check (CLI gate) and entry point used by get_calculator
# ═════════════════════════════════════════════════════════════════════════════

def require_accelerator(accelerator, *, backend: str, model: str | None,
                        model_path: str | None, default_dtype: str | None,
                        engine: str | None = None) -> None:
    """Check the accelerator settings before any work starts.

    Raises :class:`AcceleratorError` for an unknown engine or key, an explicit
    accelerator with a non-MACE model, Symmetrix-XL not installed, or an
    option it does not accept.
    """
    settings = AcceleratorSettings.from_config(accelerator)
    if not settings.enabled:
        return
    if engine == "torchsim" and backend == "mace":
        # torch-sim drives the batch; Symmetrix-XL can be its model (see
        # torchsim_symmetrix). Anything else runs plain MACE, or is refused
        # when the accelerator was asked for by name.
        if torchsim_flavour(settings) is not None:
            flavour, _ = resolve_flavour(settings.engine)
            dtype = "float64" if default_dtype in (None, "auto") else default_dtype
            check_options(flavour, settings.options, dtype)
        return
    if backend != "mace":
        if settings.engine != "auto":
            raise AcceleratorError(
                f"--accelerator {settings.engine} accelerates MACE models; "
                f"'{model_path or model}' uses the {backend} backend. Use "
                f"--accelerator auto to run other backends unchanged.")
        return
    resolved = resolve_flavour(settings.engine)
    if resolved is None:
        return
    flavour, _ = resolved
    dtype = "float64" if default_dtype in (None, "auto") else default_dtype
    check_options(flavour, settings.options, dtype)


# ═════════════════════════════════════════════════════════════════════════════
# --engine torchsim
# ═════════════════════════════════════════════════════════════════════════════

def torchsim_flavour(settings: AcceleratorSettings) -> tuple[str, str] | None:
    """The package that evaluates the model under ``--engine torchsim``, or
    None for plain (PyTorch) MACE (no accelerator, or ``auto`` without
    Symmetrix-XL). A batch is evaluated in one joined Symmetrix-XL call
    (:mod:`amorphgen.utils.torchsim_symmetrix`)."""
    if not settings.enabled:
        return None
    resolved = resolve_flavour(settings.engine)
    if resolved is None:
        print("[Accelerator] --engine torchsim runs plain MACE")
    return resolved


def torchsim_model(accelerator, *, model: str | None, model_path: str | None,
                   dtype: str, device, torch_dtype):
    """A torch-sim model that evaluates this MACE model with Symmetrix-XL, or
    None when torch-sim should build its usual PyTorch MACE model."""
    settings = AcceleratorSettings.from_config(accelerator)
    if torchsim_flavour(settings) is None:
        return None
    calc = load_accelerated(settings, model=model, model_path=model_path,
                            dtype=str(dtype), device=str(device).split(":")[0])
    from .torchsim_symmetrix import symmetrix_torchsim_model
    wrapped = symmetrix_torchsim_model(calc, device, torch_dtype,
                                       batch_atoms=settings.batch_max_atoms)
    wrapped.amorphgen_accelerator = calc.amorphgen_accelerator
    print(f"  [torch-sim] model evaluated by {calc.amorphgen_accelerator['flavour']} "
          f"({wrapped.evaluation}), chunks of {wrapped.amorphgen_batch} structures")
    return wrapped


# ═════════════════════════════════════════════════════════════════════════════
# Entry point used by get_calculator
# ═════════════════════════════════════════════════════════════════════════════

def load_accelerated(settings: AcceleratorSettings, *, model: str | None,
                     model_path: str | None, dtype: str, device: str):
    """An accelerated calculator for this MACE model, or None for plain MACE."""
    resolved = resolve_flavour(settings.engine)
    if resolved is None:
        return None
    flavour, version = resolved
    check_options(flavour, settings.options, dtype)
    print(f"[Accelerator] Using {flavour} {version}; store {model_store(settings.store)}")
    warning = untested_version(flavour, version)
    if warning:
        print(warning)
    json_path = prepare(model, model_path, flavour=flavour, version=version,
                        spline_points=settings.spline_points,
                        store=settings.store, convert=settings.convert)
    from .calculators import _load_symmetrix
    calc = _load_symmetrix(str(json_path), device=device, default_dtype=dtype,
                           symmetrix_params=settings.options)
    if settings.validate:
        def check(elements):
            validate(json_path, calc, model=model, model_path=model_path,
                     elements=elements, dtype=dtype, tolerance=settings.tolerance,
                     store=settings.store, options=settings.options)
        if settings.elements:
            check(settings.elements)
        else:
            print("[Accelerator] Elements not given; the check against MACE "
                  "runs on the first structure evaluated")
            _validate_on_first_use(calc, check)
    entry = _read_index(model_store(settings.store)).get(
        f"{json_path.parent.name}/{json_path.name}", {})
    calc.amorphgen_accelerator = {
        "engine": settings.engine, "flavour": flavour, "version": version,
        "model_file": str(json_path), "spline_points": settings.spline_points,
        "source": entry.get("source"), "source_sha256": entry.get("source_sha256"),
        "options": dict(settings.options),
        "batch_max_atoms": settings.batch_max_atoms,
        "batch_skin": float(settings.batch_skin),
    }
    return calc
