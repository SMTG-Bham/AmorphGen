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
* **Custom**    — any local ``.model`` file via ``--model-path``
* **External**  — pass your own ASE calculator object directly
"""

from __future__ import annotations

import os
import warnings
from typing import Any


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
    "mace-omat-0-small":  "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-small.model",
    "mace-omat-0-medium": "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model",
    "mace-omat-0":        "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model",
    # ── MACE-MATPES (PBE / r2SCAN, ASL license) ─────────────────────────────
    "mace-matpes-pbe":    "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
    "mace-matpes-r2scan": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model",
    # ── MACE-MH (multi-domain: bulk + surface + molecule) ────────────────────
    "mace-mh-0":          "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-0.model",
    "mace-mh-1":          "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
    # ── MACE-OMOL (molecules) ────────────────────────────────────────────────
    "mace-omol":          "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/mace-omol-0-medium.model",
}

# ── CHGNet identifiers ───────────────────────────────────────────────────────
CHGNET_MODELS: set[str] = {"chgnet"}

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
    resolved = MACE_FOUNDATION_MODELS.get(model, model)

    if resolved.startswith("https://") or os.path.isfile(resolved):
        print(f"[MACE] Loading from URL/path: {resolved[:80]}")
        return MACECalculator(model_paths=resolved, device=device, **kwargs)
    else:
        print(f"[MACE] Loading foundation model '{model}' → mace_mp(model='{resolved}')")
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
        raise NotImplementedError(
            "CHGNet does not support default_dtype='float64': its "
            "composition_model submodule constructs input tensors at "
            "float32 regardless of torch.get_default_dtype(), so the "
            "forward pass crashes with a dtype mismatch.  CHGNet is "
            "trained and benchmarked at float32 — keep default_dtype "
            "as 'float32' (or omit it) for CHGNet, or switch to MACE "
            "(model='mace-mpa-0', default_dtype='float64') if you need "
            "float64 precision."
        )
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

    checkpoint = SEVENNET_MODELS.get(model, model)
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
        alpha = cp.get("alpha", 0.2)
        print(f"[Classical] Buckingham+Coulomb, {len(converted)} pair(s), "
              f"cutoff={cp.get('cutoff', 10.0)} A, "
              f"coulomb={'on' if coulomb else 'off'}{dev_str}")
        return BuckinghamCalculator(
            params=converted,
            charges=charges,
            cutoff=cp.get("cutoff", 10.0),
            alpha=alpha,
            coulomb=coulomb,
            device=device,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Backend detection
# ═════════════════════════════════════════════════════════════════════════════

def _detect_backend(model: str) -> str:
    """
    Determine which backend a model name belongs to.

    Returns one of: "mace", "chgnet", "sevennet", "classical".
    Raises ValueError if the model is not recognised.
    """
    lower = model.lower()

    # Classical pair potentials
    if lower in CLASSICAL_MODELS:
        return "classical"

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

        Use :func:`list_models` or ``--list-models`` to see all options.
        Ignored if *model_path* is provided (defaults to MACE backend).

    device : str
        ``"cuda"`` or ``"cpu"``.

    model_path : str, optional
        Path to a local ``.model`` file (e.g. a fine-tuned MACE model).
        Takes priority over *model*.  Currently only MACE ``.model``
        files are supported for custom paths.

    **kwargs
        Extra keyword arguments forwarded to the backend-specific
        calculator constructor.

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
    """
    # ── Resolve "auto" device once, here, so backends always see a real
    #    device string.  Order: explicit > CUDA > MPS > CPU.
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif (hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()):
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

    # ── Custom model path → MACE backend ──────────────────────────────────
    if model_path is not None:
        return _load_mace(model, device=device, model_path=model_path, **kwargs)

    # ── Route by backend ──────────────────────────────────────────────────
    backend = _detect_backend(model)

    if backend == "classical":
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
    """Print all available foundation models to stdout."""
    bar = "-" * 72
    print(f"\n{bar}")
    print("  Available foundation models  (pass as --model NAME)")
    print(bar)

    # Group by backend
    sections = [
        ("MACE", {k: v for k, v in MODEL_DESCRIPTIONS.items()
                  if k.startswith("mace-")}),
        ("CHGNet", {k: v for k, v in MODEL_DESCRIPTIONS.items()
                    if k == "chgnet"}),
        ("SevenNet", {k: v for k, v in MODEL_DESCRIPTIONS.items()
                      if k == "sevennet" or k.startswith("7net")}),
        ("Classical (requires classical_params in YAML)", {
            k: v for k, v in MODEL_DESCRIPTIONS.items()
            if k in ("lennard-jones", "buckingham")}),
    ]

    for backend_name, models in sections:
        print(f"\n  -- {backend_name} {'-' * (60 - len(backend_name))}")
        for name, desc in models.items():
            print(f"  {name:<25s}  {desc}")

    print(f"\n{bar}")
    print("  Custom model:  --model-path /path/to/my_finetuned.model")
    print(f"{bar}\n")
