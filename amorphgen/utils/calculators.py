"""
amorphgen.utils.calculators
----------------------------
Calculator factory for universal machine learning interatomic
potentials (MLIPs).

All backends return a standard ASE calculator. The user picks a
model by short name
(``--model mace-mpa-0``, ``--model chgnet``, …)
and the factory handles the import, initialisation, and device
placement transparently.

Supported backends
~~~~~~~~~~~~~~~~~~
* **MACE**      — ``mace-mp-*``, ``mace-mpa-*``, ``mace-omat-*``,
                   ``mace-mh-*``, ``mace-matpes-*``, ``mace-omol``
* **CHGNet**    — ``chgnet``  (latest pretrained CHGNet)
* **SevenNet**  — ``sevennet``, ``7net-mf-ompa``, ``7net-l3i5``, ...
* **ACE**       — ``ace`` with a pacemaker ``.yaml`` / ``.yace`` / ``.ace``
                   file via ``--model-path`` (the suffix alone implies ACE)
* **LAMMPS**    — ``lammps`` with any pair style (``lammps_params``)
* **Custom**    — any local ``.model`` file via ``--model-path``
* **External**  — pass your own ASE calculator object directly
"""

from __future__ import annotations

import os
import warnings
from typing import Any

from .common import resolve_device


# ═════════════════════════════════════════════════════════════════════════════
# MACE model registry
# Source: https://github.com/ACEsuit/mace-foundations
# ═════════════════════════════════════════════════════════════════════════════

MACE_FOUNDATION_MODELS: dict[str, str] = {
    # ── MACE-MP-0a (initial release, MPTrj, PBE+U) ──────────────────────────
    "mace-mp-0a-small":   "small",
    "mace-mp-0a-medium":  "medium",
    "mace-mp-0a-large":   "large",
    # ── MACE-MP-0b (improved pair repulsion) ─────────────────────────────────
    "mace-mp-0b-small":   "small-0b",
    "mace-mp-0b-medium":  "medium-0b",
    "mace-mp-0b-large":   "large-0b",
    # ── MACE-MP-0b2 (high-pressure stability) ────────────────────────────────
    "mace-mp-0b2-small":  "small-0b2",
    "mace-mp-0b2-medium": "medium-0b2",
    "mace-mp-0b2-large":  "large-0b2",
    # ── MACE-MP-0b3 (fixed phonons vs 0b2) ───────────────────────────────────
    "mace-mp-0b3-small":  "small-0b3",
    "mace-mp-0b3-medium": "medium-0b3",
    "mace-mp-0b3-large":  "large-0b3",
    # ── MACE-MPA-0 (MPTrj + sAlex — recommended default) ────────────────────
    "mace-mpa-0":         "medium-mpa-0",
    "mace-mpa-0-medium":  "medium-mpa-0",
    # ── MACE-OMAT-0 (Open Materials — excellent phonons, ASL license) ────────
    # Use mace_mp's own short names (not raw URLs): mace_mp downloads them,
    # whereas MACECalculator(model_paths=URL) cannot fetch a URL.
    "mace-omat-0-small":  "small-omat-0",
    "mace-omat-0-medium": "medium-omat-0",
    "mace-omat-0":        "medium-omat-0",
    # ── MACE-MATPES (PBE / r2SCAN, ASL license) ─────────────────────────────
    "mace-matpes-pbe":    "mace-matpes-pbe-0",
    "mace-matpes-r2scan": "mace-matpes-r2scan-0",
    # ── MACE-MH (multi-domain: bulk + surface + molecule) ────────────────────
    "mace-mh-0":          "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-0.model",
    "mace-mh-1":          "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
    # ── MACE-OMOL (molecules) ────────────────────────────────────────────────
    "mace-omol":          "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/mace-omol-0-medium.model",
}

# ── CHGNet identifiers ───────────────────────────────────────────────────────
CHGNET_MODELS: set[str] = {"chgnet"}

# Raised by _load_chgnet and, before any work starts, by require_dtype.
_CHGNET_FLOAT64_MSG = (
    "CHGNet does not support default_dtype='float64': its "
    "composition_model submodule constructs input tensors at "
    "float32 regardless of torch.get_default_dtype(), so the "
    "forward pass crashes with a dtype mismatch.  CHGNet is "
    "trained and benchmarked at float32 — keep default_dtype "
    "as 'float32' (or omit it) for CHGNet, or switch to MACE "
    "(model='mace-mpa-0', default_dtype='float64') if you need "
    "float64 precision."
)


def _ci_get(registry: dict, name: str):
    """Case-insensitive registry lookup (``MACE-MPA-0`` == ``mace-mpa-0``)."""
    if name in registry:
        return registry[name]
    low = name.lower()
    for k, v in registry.items():
        if k.lower() == low:
            return v
    return name

# ── SevenNet identifiers ────────────────────────────────────────────────────
# Maps short name → SevenNet checkpoint name passed to SevenNetCalculator.
SEVENNET_MODELS: dict[str, str] = {
    "sevennet":       "7net-mf-ompa",        # alias → recommended default
    "sevennet-mf":    "7net-mf-ompa",
    "7net-mf-ompa":   "7net-mf-ompa",
    "7net-mf-0":      "7net-mf-0",
    "7net-omat":      "7net-omat",
    "7net-l3i5":      "7net-l3i5",
    "7net-0":         "7net-0",
    "7net-omni":      "7net-omni",
}

# ── Classical potential identifiers ──────────────────────────────────────────
CLASSICAL_MODELS: set[str] = {"lennard-jones", "lj", "buckingham", "buck"}

# ── Potential-file backends ──────────────────────────────────────────────────
# ACE reads its potential from --model-path; LAMMPS from lammps_params.
ACE_MODELS: set[str] = {"ace"}
LAMMPS_MODELS: set[str] = {"lammps"}
# The file formats PyACECalculator reads (B-basis .yaml, C-tilde .yace/.ace).
ACE_SUFFIXES = (".yaml", ".yace", ".ace")

# Parameter blocks a model reads from the config, and the backend each is for.
POTENTIAL_PARAM_KEYS = {
    "classical_params": "classical",
    "lammps_params": "lammps",
    "ace_params": "ace",
}


def is_ace_file(path) -> bool:
    """True if *path* names an ACE potential file by its suffix."""
    return path is not None and os.fspath(path).endswith(ACE_SUFFIXES)


def potential_kwargs(cfg) -> dict:
    """The parameter blocks set in *cfg*, as :func:`get_calculator` kwargs."""
    return {key: cfg[key] for key in POTENTIAL_PARAM_KEYS if cfg and cfg.get(key)}


def calculator_kwargs(cfg, *, defaults=None) -> dict:
    """Translate resolved settings into calculator factory arguments.

    A present value (including None) overrides a default. Backend loading,
    device resolution, caching and repulsive-core wrapping remain with their
    existing owners. CLI callers can supply argument-parser defaults.
    """
    fallback = {
        "model": "mace-mpa-0", "device": "cuda",
        "model_path": None, "default_dtype": "auto",
    }
    if defaults is not None:
        fallback.update(defaults)
    return {**{key: cfg.get(key, value) for key, value in fallback.items()},
            **potential_kwargs(cfg)}


# ═════════════════════════════════════════════════════════════════════════════
# Human-readable model descriptions (for --list-models)
# ═════════════════════════════════════════════════════════════════════════════

MODEL_DESCRIPTIONS: dict[str, str] = {
    # MACE
    "mace-mp-0a-small":   "MACE-MP-0a  small   | MPTrj | DFT PBE+U | initial release",
    "mace-mp-0a-medium":  "MACE-MP-0a  medium  | MPTrj | DFT PBE+U | initial release",
    "mace-mp-0a-large":   "MACE-MP-0a  large   | MPTrj | DFT PBE+U | initial release",
    "mace-mp-0b-small":   "MACE-MP-0b  small   | MPTrj | improved pair repulsion",
    "mace-mp-0b-medium":  "MACE-MP-0b  medium  | MPTrj | improved pair repulsion",
    "mace-mp-0b-large":   "MACE-MP-0b  large   | MPTrj | improved pair repulsion",
    "mace-mp-0b2-small":  "MACE-MP-0b2 small   | MPTrj | improved high-pressure stability",
    "mace-mp-0b2-medium": "MACE-MP-0b2 medium  | MPTrj | improved high-pressure stability",
    "mace-mp-0b2-large":  "MACE-MP-0b2 large   | MPTrj | improved high-pressure stability",
    "mace-mp-0b3-small":  "MACE-MP-0b3 small   | MPTrj | fixed phonons vs 0b2",
    "mace-mp-0b3-medium": "MACE-MP-0b3 medium  | MPTrj | fixed phonons vs 0b2",
    "mace-mp-0b3-large":  "MACE-MP-0b3 large   | MPTrj | fixed phonons vs 0b2",
    "mace-mpa-0-medium":  "MACE-MPA-0  medium  | MPTrj+sAlex | ★ recommended default",
    "mace-omat-0-small":  "MACE-OMAT-0 small   | OMAT | excellent phonons | ASL license",
    "mace-omat-0-medium": "MACE-OMAT-0 medium  | OMAT | excellent phonons | ASL license",
    "mace-matpes-pbe":    "MACE-MATPES-PBE     | MATPES-PBE | DFT PBE, no +U | ASL",
    "mace-matpes-r2scan": "MACE-MATPES-r2SCAN  | MATPES-r2SCAN | better functional | ASL",
    "mace-mh-0":          "MACE-MH-0           | multi-domain bulk/surface/molecule",
    "mace-mh-1":          "MACE-MH-1           | multi-domain | ★ best cross-domain",
    "mace-omol":          "MACE-OMOL-0         | OMOL | optimised for molecules",
    # CHGNet
    "chgnet":             "CHGNet              | MPTrj | charge-informed | magnetic moments",
    # SevenNet
    "sevennet":           "SevenNet-MF-OMPA    | OMat+MPtrj+Alexandria | ★ recommended SevenNet",
    "7net-mf-ompa":       "SevenNet-MF-OMPA    | OMat+MPtrj+Alexandria | foundation",
    "7net-mf-0":          "SevenNet-MF-0       | multi-fidelity baseline",
    "7net-omat":          "SevenNet-OMat       | OMat-only training",
    "7net-l3i5":          "SevenNet-l3i5       | improved equivariant features",
    "7net-0":             "SevenNet-0          | original release (Jul 2024)",
    "7net-omni":          "SevenNet-omni       | omni model (multi-task)",
    # Classical
    "lennard-jones":      "Lennard-Jones       | pair potential | no GPU needed",
    "buckingham":         "Buckingham+Coulomb  | rigid-ion | Wolf summation | no GPU needed",
    # Potential files
    "ace":                "ACE (pacemaker)     | --model-path pot.yace (.yaml/.ace) | CPU",
    "lammps":             "LAMMPS pair style   | --pair-style ... --pair-coeff ... | CPU",
}


# ═════════════════════════════════════════════════════════════════════════════
# Backend loaders (lazy imports — each backend is only imported when needed)
# ═════════════════════════════════════════════════════════════════════════════

def _load_mace(model: str, device: str, model_path: str | None = None,
               **kwargs) -> Any:
    """Load a MACE calculator."""
    try:
        from mace.calculators import mace_mp, MACECalculator
    except ImportError:
        raise ImportError(
            "MACE is not installed. Install it with:\n"
            "  pip install amorphgen[mace]\n"
            "Or: pip install mace-torch\n"
            "See: https://github.com/ACEsuit/mace"
        )

    # Multi-head models (mace-mh-*) bundle several functionals in one file and
    # require a `head` to pick which one. Accept a `model:head` syntax, e.g.
    # 'mace-mh-1:omat_pbe'. Guard on '://' so raw URLs aren't split on their
    # scheme colon. An explicit head=... kwarg takes precedence.
    head = kwargs.pop("head", None)
    if head is None and ":" in model and "://" not in model:
        model, head = model.split(":", 1)
    if head is not None:
        kwargs["head"] = head

    # Custom / local model file
    if model_path is not None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Custom MACE model file not found: {model_path}\n"
                "Please provide a valid path to a .model file."
            )
        print(f"[MACE] Loading custom model: {model_path}")
        return MACECalculator(model_paths=model_path, device=device, **kwargs)

    # Resolve short-name → internal string / URL
    resolved = _ci_get(MACE_FOUNDATION_MODELS, model)

    if os.path.isfile(resolved):
        # Local .model file
        print(f"[MACE] Loading local model file: {resolved}")
        return MACECalculator(model_paths=resolved, device=device, **kwargs)
    # Short name (e.g. 'medium-mpa-0') OR a URL — mace_mp resolves names and
    # downloads URLs to its cache. (MACECalculator cannot fetch a URL, which is
    # why passing URLs to it previously broke the omat/matpes/mh/omol models.)
    _head = f", head='{head}'" if head else ""
    print(f"[MACE] Loading foundation model '{model}' → mace_mp(model='{resolved[:70]}'{_head})")
    return mace_mp(model=resolved, device=device, **kwargs)

def _load_chgnet(device: str, default_dtype: str | None = None,
                 **kwargs) -> Any:
    """Load the pretrained CHGNet calculator.

    CHGNet's published MD benchmarks all use float32 and its checkpoint is
    stored at float32 natively.  Pre-2026-05-11 this loader accepted a
    ``default_dtype`` kwarg, silently forwarded it to ``CHGNetCalculator``,
    and the latter ignored it — so YAML configs with ``default_dtype: float64``
    were a no-op.  This loader makes the YAML field meaningful by setting
    torch's default dtype before importing chgnet (so its module-level
    ``TORCH_DTYPE`` constant captures the right precision) and overwriting
    ``chgnet.model.model.TORCH_DTYPE`` explicitly in case chgnet was already
    imported by an earlier call.

    **float64 is currently not supported** — CHGNet's ``composition_model``
    submodule builds its input feature vectors via a code path that bypasses
    ``TORCH_DTYPE``, so even after casting the main model to float64 the
    forward pass crashes with a dtype mismatch.  Use MACE (e.g. ``mace-mpa-0``)
    if you need a float64 backend.

    Parameters
    ----------
    device : str
        ``"cpu"``, ``"cuda"``, or ``"mps"``.
    default_dtype : {"float32", None}, optional
        ``None`` (default) and ``"float32"`` both give native float32.
        ``"float64"`` raises ``NotImplementedError``.

    Raises
    ------
    NotImplementedError
        If ``default_dtype="float64"`` — CHGNet's composition_model
        sub-module cannot be cleanly cast to float64.  Use MACE if you
        need float64 precision.
    ValueError
        If ``default_dtype`` is some other unrecognised string.
    """
    import torch

    if default_dtype is None:
        default_dtype = "float32"
    if default_dtype == "float64":
        raise NotImplementedError(_CHGNET_FLOAT64_MSG)
    if default_dtype != "float32":
        raise ValueError(
            f"default_dtype must be 'float32' or None for CHGNet; "
            f"got {default_dtype!r}"
        )

    # Set torch's default dtype *before* importing chgnet so the module-level
    # TORCH_DTYPE constant is captured at float32.  This is the standard
    # state for PyTorch anyway; we set it explicitly to override any earlier
    # caller that set float64 (e.g. a MACE loader earlier in the process).
    torch.set_default_dtype(torch.float32)

    try:
        from chgnet.model.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        import chgnet.model.model as _chgnet_model_mod
    except ImportError:
        raise ImportError(
            "CHGNet is not installed. Install it with:\n"
            "  pip install chgnet\n"
            "See: https://chgnet.lbl.gov/"
        )

    # If chgnet was already imported, its TORCH_DTYPE is frozen — overwrite.
    if getattr(_chgnet_model_mod, "TORCH_DTYPE", None) is not torch.float32:
        _chgnet_model_mod.TORCH_DTYPE = torch.float32

    print(f"[CHGNet] Loading pretrained model on {device} (dtype=float32)")
    # Load on CPU first to avoid MPS float64 crash, then cast and move.
    model = CHGNet.load(use_device="cpu")

    if device == "mps":
        model = model.float()   # MPS requires float32
        model = model.to("mps")
        print("[CHGNet] Moved to MPS (float32)")

    # CHGNetCalculator silently ignores unknown kwargs; strip default_dtype
    # so it doesn't sit in **kwargs and confuse future signature checks.
    kwargs.pop("default_dtype", None)
    return CHGNetCalculator(model=model, use_device=device, **kwargs)




def _load_sevennet(model: str, device: str, **kwargs) -> Any:
    """Load a SevenNet calculator (KAIST equivariant MLIP).

    Multi-fidelity models (``7net-mf-*``) require a ``modal`` argument
    selecting the DFT functional/dataset; we default to ``'mpa'``
    (MPtrj+Alexandria, PBE) which is the closest match to most PBE-trained
    benchmarks. Override with ``modal='omat24'`` for OMat-style PBE+U.
    """
    try:
        from sevenn.calculator import SevenNetCalculator
    except ImportError:
        raise ImportError(
            "SevenNet is not installed. Install it with:\n"
            "  pip install sevenn\n"
            "See: https://github.com/MDIL-SNU/SevenNet"
        )

    checkpoint = _ci_get(SEVENNET_MODELS, model)
    # Multi-fidelity (mf) models need a modal selection
    if "mf" in checkpoint and "modal" not in kwargs:
        kwargs["modal"] = "mpa"
        print(f"[SevenNet] '{checkpoint}' is multi-fidelity; defaulting "
              f"modal='mpa' (MPtrj+Alexandria, PBE)")
    print(f"[SevenNet] Loading pretrained '{checkpoint}' on {device}")
    return SevenNetCalculator(model=checkpoint, device=device, **kwargs)


def _load_classical(model: str, device: str = "cpu", **kwargs) -> Any:
    """
    Load a classical pair potential calculator.

    Parameters are passed via ``classical_params`` in kwargs (from YAML
    config or Python API).

    Parameters
    ----------
    model : str
        "lennard-jones" / "lj" or "buckingham" / "buck".
    **kwargs
        Must contain ``classical_params`` dict with:

        For LJ::

            {"params": {("Ar","Ar"): {"epsilon": 0.0104, "sigma": 3.40}},
             "cutoff": 10.0}

        For Buckingham::

            {"params": {("Si","O"): {"A": 18003.76, "rho": 0.2052, "C": 133.54}},
             "charges": {"Si": 2.4, "O": -1.2},
             "cutoff": 10.0}
    """
    from .classical import LennardJonesCalculator, BuckinghamCalculator

    cp = kwargs.pop("classical_params", None)
    if cp is None:
        raise ValueError(
            f"Model '{model}' requires 'classical_params' with potential "
            f"parameters. Provide via YAML config or Python API.\n"
            f"See: amorphgen.utils.classical for parameter format."
        )

    # Convert string-keyed pair dicts to tuple keys
    # YAML gives {"Si-O": {...}} but calculators expect {("Si","O"): {...}}
    import copy as _copy
    cp = _copy.deepcopy(cp)          # never mutate the caller's config (YAML/JSON-safe)
    params = cp.get("params", {})
    converted = {}
    for key, val in params.items():
        if isinstance(key, str) and "-" in key:
            s1, s2 = key.split("-", 1)
            converted[(s1, s2)] = val
        else:
            converted[key] = val
    cp["params"] = converted

    lower = model.lower()
    dev_str = f", device={device}" if device != "cpu" else ""
    if lower in ("lennard-jones", "lj"):
        print(f"[Classical] Lennard-Jones, {len(converted)} pair(s), "
              f"cutoff={cp.get('cutoff', 10.0)} A{dev_str}")
        return LennardJonesCalculator(
            params=converted,
            cutoff=cp.get("cutoff", 10.0),
            device=device,
        )
    else:
        charges = cp.get("charges", {})
        coulomb = cp.get("coulomb", True)
        alpha = cp.get("alpha", None)
        coulomb_method = cp.get("coulomb_method", "ewald")
        print(f"[Classical] Buckingham+Coulomb, {len(converted)} pair(s), "
              f"cutoff={cp.get('cutoff', 10.0)} A, "
              f"coulomb={coulomb_method if coulomb else 'off'}{dev_str}")
        return BuckinghamCalculator(
            params=converted,
            charges=charges,
            cutoff=cp.get("cutoff", 10.0),
            alpha=alpha,
            coulomb=coulomb,
            coulomb_method=coulomb_method,
            device=device,
        )


def _cpu_float64_note(tag: str, device: str, default_dtype: str | None) -> None:
    """ACE and LAMMPS evaluate on the CPU in float64 whatever was requested."""
    if device != "cpu":
        print(f"[{tag}] Runs on the CPU (device '{device}' is not used)")
    if default_dtype == "float32":
        print(f"[{tag}] Evaluates in float64 (default_dtype 'float32' is not used)")


def _load_ace(model_path: str, device: str = "cpu",
              default_dtype: str | None = None,
              ace_params: dict | None = None) -> Any:
    """Load an ACE potential file (pacemaker .yaml / .yace / .ace) with pyace.

    ``ace_params`` holds PyACECalculator evaluator options
    (``recursive_evaluator``, ``recursive``, ``fast_nl``).
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"ACE potential file not found: {model_path}\n"
            f"Give a pacemaker {' / '.join(ACE_SUFFIXES)} file with --model-path.")
    try:
        from .ace_potential import ACE_PARAM_KEYS, ACECalculator
    except ImportError:
        raise BackendNotInstalledError(_install_message("ace", "ace"))
    options = dict(ace_params or {})
    unknown = sorted(set(options) - ACE_PARAM_KEYS)
    if unknown:
        raise ValueError(f"Unknown ace_params key(s): {', '.join(unknown)}")
    _cpu_float64_note("ACE", device, default_dtype)
    path = os.path.abspath(model_path)
    print(f"[ACE] Loading potential: {path}")
    return ACECalculator(path, **options)


def _load_lammps(device: str = "cpu", default_dtype: str | None = None,
                 lammps_params: dict | None = None) -> Any:
    """Build a LAMMPS calculator for the pair style in ``lammps_params``.

    See :func:`amorphgen.utils.lammps_potential.lammps_setup` for the keys.
    LAMMPS itself starts on the first calculation.
    """
    from .lammps_potential import make_lammps_calculator

    calc = make_lammps_calculator(lammps_params)
    _cpu_float64_note("LAMMPS", device, default_dtype)
    print(f"[LAMMPS] {calc.parameters.lmpcmds[0]}, types "
          f"{' '.join(f'{t}={el}' for el, t in calc.parameters.atom_types.items())}")
    return calc


# ═════════════════════════════════════════════════════════════════════════════
# Backend detection
# ═════════════════════════════════════════════════════════════════════════════

def _detect_backend(model: str) -> str:
    """
    Determine which backend a model name belongs to.

    Returns one of: "mace", "chgnet", "sevennet", "classical", "ace",
    "lammps". Raises ValueError if the model is not recognised. A model
    path changes the answer; use :func:`backend_for` when there may be one.
    """
    lower = model.lower()

    # Strip a 'model:head' suffix (multi-head MACE, e.g. 'mace-mh-1:omat_pbe')
    # for classification only — guard on '://' so URLs aren't split.
    if ":" in lower and "://" not in lower:
        lower = lower.split(":", 1)[0]

    # Classical pair potentials
    if lower in CLASSICAL_MODELS:
        return "classical"

    # Potential-file backends
    if lower in ACE_MODELS:
        return "ace"
    if lower in LAMMPS_MODELS:
        return "lammps"

    # MACE — explicit registry match only
    if lower in MACE_FOUNDATION_MODELS:
        return "mace"

    # MACE prefix (allow custom mace-* models, but warn if not in registry)
    if lower.startswith("mace-"):
        import warnings
        warnings.warn(
            f"Model '{model}' not in MACE registry. "
            f"Will attempt to load it. Use --list-models to see known models.",
            stacklevel=3,
        )
        return "mace"

    # CHGNet
    if lower in CHGNET_MODELS:
        return "chgnet"

    # SevenNet
    if lower in SEVENNET_MODELS or lower.startswith("7net") or lower.startswith("sevennet"):
        return "sevennet"

    raise ValueError(
        f"Unrecognised model '{model}'. Use --list-models to see available "
        f"options, or pass --model-path for a custom model file."
    )


def backend_for(model: str, model_path: str | None = None) -> str:
    """The backend that :func:`get_calculator` uses for *model* / *model_path*.

    A model path means MACE (it takes priority over *model*), except for an
    ACE potential: ``model='ace'`` or a ``.yaml`` / ``.yace`` / ``.ace`` path.
    """
    name = (model or "").lower()
    if model_path is not None:
        if name in ACE_MODELS or is_ace_file(model_path):
            return "ace"
        if name in LAMMPS_MODELS:
            raise ValueError(
                "Model 'lammps' takes its potential files in the pair_coeff "
                "lines (--pair-coeff or lammps_params), not --model-path.")
        return "mace"
    if name in ACE_MODELS:
        raise ValueError(
            "Model 'ace' needs its potential file: --model-path pot.yace "
            "(or a pacemaker .yaml / .ace file).")
    return _detect_backend(model)


# ═════════════════════════════════════════════════════════════════════════════
# Backend availability for fail-fast checks and model-list install hints
# ═════════════════════════════════════════════════════════════════════════════

# Import name and pip-extra per MLIP backend. Single source of backend
# knowledge: the CLI fail-fast, --list-models markers, and error messages all
# derive from these two dicts.
_BACKEND_IMPORT = {"mace": "mace", "chgnet": "chgnet", "sevennet": "sevenn",
                   "ace": "pyace", "lammps": "lammps"}
_BACKEND_EXTRA = {"mace": "mace", "chgnet": "chgnet", "sevennet": "sevennet",
                  "ace": "ace", "lammps": "lammps"}
# Where the pip extra is not the whole story.
_BACKEND_NOTE = {
    "ace": ("python-ace has wheels for Linux x86_64, Python 3.10-3.13; "
            "elsewhere build it from\n"
            "                          https://github.com/ICAMS/python-ace"),
    "lammps": ("the PyPI lammps wheel has no pair_style pace; for that use "
               "conda-forge's\n"
               "                          lammps (conda install -c conda-forge lammps)"),
}


class BackendNotInstalledError(ImportError):
    """The requested model's MLIP backend package is not installed."""


def backend_available(backend: str) -> bool:
    """True if *backend*'s python package is importable.

    ``"classical"`` is always available (NumPy implementation in the base
    install). Does not import the package — checks the module spec only, so
    it is cheap and safe on torch-free installs.
    """
    if backend == "classical":
        return True
    mod = _BACKEND_IMPORT.get(backend)
    if mod is None:
        return False
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def available_backends() -> dict[str, bool]:
    """Installed-state of every backend, e.g. ``{"mace": False, "chgnet": True,
    "sevennet": False, "classical": True}``."""
    return {b: backend_available(b) for b in (*_BACKEND_IMPORT, "classical")}


def _install_message(model: str, backend: str) -> str:
    extra = _BACKEND_EXTRA[backend]
    installed = [b for b, ok in available_backends().items() if ok]
    note = (f"  Note:                   {_BACKEND_NOTE[backend]}\n"
            if backend in _BACKEND_NOTE else "")
    return (
        f"Model '{model}' needs the {backend.upper()} backend, which is not "
        f"installed.\n"
        f"  Install it:             pip install \"amorphgen[{extra}]\"\n"
        f"{note}"
        f"  Installed backends:     {', '.join(installed)}\n"
        f"  Torch-free alternative: classical potentials (--model lj or "
        f"buckingham,\n"
        f"                          with classical_params in a YAML config)\n"
        f"  See all models:         amorphgen --list-models"
    )


def require_backend(model: str, model_path: str | None = None) -> str:
    """Fail-fast check that *model*'s backend is importable.

    Returns the backend name when available. Raises
    :class:`BackendNotInstalledError` with a curated, copy-pasteable message
    otherwise (and propagates :class:`ValueError` for unrecognised models).
    The CLI calls this BEFORE any setup work; :func:`get_calculator` keeps its
    own lazy import errors as the API-level backstop.
    """
    backend = backend_for(model, model_path)
    if backend_available(backend):
        return backend
    raise BackendNotInstalledError(_install_message(model, backend))


def require_potential(model: str, model_path: str | None = None, *,
                      engine: str | None = None,
                      lammps_params: dict | None = None, **_params) -> None:
    """Fail-fast check of the ACE / LAMMPS potential before any work starts.

    ACE needs an existing potential file and LAMMPS a usable
    ``lammps_params`` block (the same check its loader runs); neither runs
    on the torch-sim engine. Other backends pass.
    """
    backend = backend_for(model, model_path)
    if backend not in ("ace", "lammps"):
        return
    if engine == "torchsim":
        from .torchsim_engine import check_model
        check_model(model, model_path)
    if backend == "ace":
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ACE potential file not found: {model_path}")
    else:
        from .lammps_potential import lammps_setup
        lammps_setup(lammps_params)


def require_dtype(model: str, default_dtype: str | None = None,
                  model_path: str | None = None) -> None:
    """Fail-fast check that *model* can run at *default_dtype*.

    Raises the same :class:`NotImplementedError` as the CHGNet loader when
    CHGNet is asked for float64. The CLI calls this next to
    :func:`require_backend`, BEFORE any setup work: the melt-quench stages
    build their calculator without ``default_dtype``, so ``--mq-ensemble``
    would otherwise run stages 1-4 and only fail when phase 3 builds one.
    """
    # dtype first: _detect_backend warns about unregistered mace-* names
    if (default_dtype == "float64" and model_path is None
            and _detect_backend(model) == "chgnet"):
        raise NotImplementedError(_CHGNET_FLOAT64_MSG)


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def get_calculator(
    model: str = "mace-mpa-0",
    device: str = "auto",
    model_path: str | None = None,
    **kwargs,
) -> Any:
    """
    Build and return an ASE calculator for the given foundation model.

    This is the **unified entry point** for all supported MLFF backends.
    The returned object is always a standard ASE calculator that can be
    attached to any ``ase.Atoms`` object.

    Parameters
    ----------
    model : str
        Short name identifying the model. Examples:

        * MACE:      ``"mace-mpa-0"``, ``"mace-mh-1"``, ``"mace-omat-0"``
        * CHGNet:    ``"chgnet"``
        * SevenNet:  ``"sevennet"``, ``"7net-mf-ompa"``, ``"7net-l3i5"``
        * Classical: ``"lennard-jones"``, ``"buckingham"``
        * ACE:       ``"ace"`` (with *model_path*)
        * LAMMPS:    ``"lammps"`` (with ``lammps_params``)

        Use :func:`list_models` or ``--list-models`` to see all options.
        Ignored if *model_path* is provided (defaults to MACE backend),
        unless it is ``"ace"``.

    device : str
        ``"auto"`` (default), ``"cuda"``, ``"mps"``, or ``"cpu"``.
        Auto selects CUDA, then MPS, then CPU, as in the pipeline stages.
        Without PyTorch installed, auto selects CPU.

    model_path : str, optional
        Path to a local ``.model`` file (e.g. a fine-tuned MACE model).
        Takes priority over *model*. A pacemaker ``.yaml`` / ``.yace`` /
        ``.ace`` file loads an ACE potential instead.

    **kwargs
        Extra keyword arguments forwarded to the backend-specific
        calculator constructor. The parameter blocks ``classical_params``,
        ``lammps_params`` and ``ace_params`` are only accepted by their own
        backend.

    Returns
    -------
    ase.calculators.calculator.Calculator
        A ready-to-use ASE calculator.

    Raises
    ------
    ValueError
        If the model name is not recognised by any backend.
    ImportError
        If the required backend package is not installed.
    FileNotFoundError
        If *model_path* points to a non-existent file.

    Examples
    --------
    >>> calc = get_calculator("mace-mpa-0", device="cuda")
    >>> calc = get_calculator("chgnet", device="cpu")
    >>> calc = get_calculator("7net-mf-ompa", device="cuda")
    >>> calc = get_calculator(model_path="/data/my_finetuned.model")
    >>> calc = get_calculator("buckingham", classical_params={...})
    >>> calc = get_calculator(model_path="potential.yace")
    >>> calc = get_calculator("lammps", lammps_params={
    ...     "pair_style": "sw", "pair_coeff": "* * Si.sw Si"})
    """
    device = resolve_device(device)
    backend = backend_for(model, model_path)
    for key, owner in POTENTIAL_PARAM_KEYS.items():
        if key in kwargs and not kwargs[key]:
            del kwargs[key]
        elif key in kwargs and owner != backend:
            raise ValueError(
                f"'{key}' is for {owner} models, but model '{model}' "
                f"uses the {backend} backend.")

    # ── Potential files: ACE (model_path) and LAMMPS (lammps_params) ──────
    # Both always evaluate in float64; an explicit dtype only gets a note.
    if backend in ("ace", "lammps"):
        if kwargs.get("default_dtype") == "auto":
            del kwargs["default_dtype"]
        if backend == "ace":
            return _load_ace(model_path, device=device, **kwargs)
        return _load_lammps(device=device, **kwargs)

    # ── Resolve "auto" default_dtype per backend ──────────────────────────
    # CHGNet only supports float32; passing float64 raises NotImplementedError
    # in _load_chgnet. MACE / SevenNet default to float64 (the established
    # precision in their training and published benchmarks). Classical
    # potentials use float32 (faster, accuracy doesn't matter for LJ /
    # Buckingham force evaluations).
    requested_dtype = kwargs.get("default_dtype", "auto")
    if requested_dtype == "auto":
        kwargs["default_dtype"] = (
            "float32" if backend in ("chgnet", "classical") else "float64"
        )

    # ── Custom model path → MACE backend ──────────────────────────────────
    if model_path is not None:
        return _load_mace(model, device=device, model_path=model_path, **kwargs)

    # ── Route by backend ──────────────────────────────────────────────────
    if backend == "classical":
        # Classical loaders don't take default_dtype yet — drop it
        kwargs.pop("default_dtype", None)
        return _load_classical(model, device=device, **kwargs)
    elif backend == "mace":
        return _load_mace(model, device=device, **kwargs)
    elif backend == "chgnet":
        return _load_chgnet(device=device, **kwargs)
    elif backend == "sevennet":
        return _load_sevennet(model, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown backend '{backend}' for model '{model}'")


# ── Deprecated alias for backward compatibility ───────────────────────────

def get_mace_calculator(
    model: str = "mace-mpa-0",
    device: str = "cuda",
    model_path: str | None = None,
    **kwargs,
) -> Any:
    """
    Build and return a MACE calculator.

    .. deprecated:: 2.0.0
        Use :func:`get_calculator` instead, which supports MACE and
        all other backends (CHGNet, SevenNet).

    Parameters
    ----------
    model : str
        MACE model short name (e.g. ``"mace-mpa-0"``).
    device : str
        ``"cuda"`` or ``"cpu"``.
    model_path : str, optional
        Path to a local ``.model`` file.
    **kwargs
        Forwarded to ``MACECalculator`` or ``mace_mp()``.

    Returns
    -------
    ASE calculator
    """
    warnings.warn(
        "get_mace_calculator() is deprecated. Use get_calculator() instead, "
        "which supports MACE and all other MLFF backends.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _load_mace(model, device=device, model_path=model_path, **kwargs)


def list_models() -> None:
    """Print the full model registry with per-backend installed markers.

    Shows every known model regardless of what is installed (discovery),
    with a marker per backend section saying whether it is usable right now
    and, if not, the exact install command (diagnosis).
    """
    bar = "-" * 72
    print(f"\n{bar}")
    print("  Available foundation models  (pass as --model NAME)")
    print(bar)

    avail = available_backends()

    def _marker(backend: str) -> str:
        if backend == "classical":
            return "[built-in]"
        if avail.get(backend):
            return "[installed]"
        return f'[not installed -> pip install "amorphgen[{_BACKEND_EXTRA[backend]}]"]'

    # Group by backend
    sections = [
        ("MACE", "mace", {k: v for k, v in MODEL_DESCRIPTIONS.items()
                          if k.startswith("mace-")}),
        ("CHGNet", "chgnet", {k: v for k, v in MODEL_DESCRIPTIONS.items()
                              if k == "chgnet"}),
        ("SevenNet", "sevennet", {k: v for k, v in MODEL_DESCRIPTIONS.items()
                                  if k == "sevennet" or k.startswith("7net")}),
        ("Classical (requires classical_params in YAML)", "classical", {
            k: v for k, v in MODEL_DESCRIPTIONS.items()
            if k in ("lennard-jones", "buckingham")}),
        ("ACE (pacemaker potential file)", "ace", {"ace": MODEL_DESCRIPTIONS["ace"]}),
        ("LAMMPS (any pair style)", "lammps", {"lammps": MODEL_DESCRIPTIONS["lammps"]}),
    ]

    for backend_name, backend_key, models in sections:
        head = f"  -- {backend_name} "
        print(f"\n{head}{'-' * max(60 - len(backend_name), 2)} {_marker(backend_key)}")
        for name, desc in models.items():
            print(f"  {name:<25s}  {desc}")

    print(f"\n{bar}")
    print("  Custom model:  --model-path /path/to/my_finetuned.model  (MACE)")
    print("                 --model-path /path/to/potential.yace    "
          "(ACE: .yaml / .yace / .ace)")
    print(f"{bar}\n")
