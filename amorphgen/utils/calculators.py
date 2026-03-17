"""
amorphgen.utils.calculators
----------------------------
Model-agnostic calculator factory for universal machine learning
force fields (MLFFs).

All backends return a standard ASE calculator, making the pipeline
completely model-agnostic.  The user picks a model by short name
(``--model mace-mpa-0``, ``--model chgnet``, …)
and the factory handles the import, initialisation, and device
placement transparently.

Supported backends
~~~~~~~~~~~~~~~~~~
* **MACE**      — ``mace-mp-*``, ``mace-mpa-*``, ``mace-omat-*``,
                   ``mace-mh-*``, ``mace-matpes-*``, ``mace-omol``
* **CHGNet**    — ``chgnet``  (latest pretrained CHGNet)
* **M3GNet**    — ``m3gnet``  (M3GNet-MP-2021.2.8-PES via MatGL)
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

# ── M3GNet / MatGL identifiers ───────────────────────────────────────────────
M3GNET_MODELS: dict[str, str] = {
    "m3gnet":     "M3GNet-MP-2021.2.8-PES",
    "m3gnet-pes": "M3GNet-MP-2021.2.8-PES",
    "matgl":      "M3GNet-MP-2021.2.8-PES",
}


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
    # M3GNet / MatGL
    "m3gnet":             "M3GNet-MP-2021.2.8  | MPTrj | via MatGL | 3-body interactions",
}


# ═════════════════════════════════════════════════════════════════════════════
# Backend loaders (lazy imports — each backend is only imported when needed)
# ═════════════════════════════════════════════════════════════════════════════

def _load_mace(model: str, device: str, model_path: str | None = None,
               **kwargs) -> Any:
    """Load a MACE calculator."""
    from mace.calculators import mace_mp, MACECalculator

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


def _load_chgnet(device: str, **kwargs) -> Any:
    """Load the pretrained CHGNet calculator."""
    try:
        from chgnet.model.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
    except ImportError:
        raise ImportError(
            "CHGNet is not installed. Install it with:\n"
            "  pip install chgnet\n"
            "See: https://chgnet.lbl.gov/"
        )

    print(f"[CHGNet] Loading pretrained model on {device}")
    model = CHGNet.load()
    return CHGNetCalculator(model=model, use_device=device, **kwargs)


def _load_m3gnet(model: str, device: str, **kwargs) -> Any:
    """Load an M3GNet calculator via MatGL."""
    try:
        import matgl
        from matgl.ext.ase import PESCalculator
    except ImportError:
        raise ImportError(
            "MatGL is not installed. Install it with:\n"
            "  pip install matgl\n"
            "See: https://github.com/materialsvirtuallab/matgl"
        )

    model_name = M3GNET_MODELS.get(model, model)
    print(f"[M3GNet/MatGL] Loading model '{model_name}' on {device}")
    pot = matgl.load_model(model_name)
    # MatGL PESCalculator doesn't take a device arg directly;
    # the model itself handles device placement via torch.
    return PESCalculator(pot, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# Backend detection
# ═════════════════════════════════════════════════════════════════════════════

def _detect_backend(model: str) -> str:
    """
    Determine which backend a model name belongs to.

    Returns one of: "mace", "chgnet", "m3gnet".
    Raises ValueError if the model is not recognised.
    """
    lower = model.lower()

    # MACE — explicit registry match or prefix
    if lower in MACE_FOUNDATION_MODELS or lower.startswith("mace-"):
        return "mace"

    # CHGNet
    if lower in CHGNET_MODELS:
        return "chgnet"

    # M3GNet / MatGL
    if lower in M3GNET_MODELS or lower.startswith("m3gnet") or lower == "matgl":
        return "m3gnet"

    raise ValueError(
        f"Unrecognised model '{model}'. Use --list-models to see available "
        f"options, or pass --model-path for a custom model file."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def get_calculator(
    model: str = "mace-mpa-0",
    device: str = "cuda",
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

        * MACE:     ``"mace-mpa-0"``, ``"mace-mh-1"``, ``"mace-omat-0"``
        * CHGNet:   ``"chgnet"``
        * M3GNet:   ``"m3gnet"``

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
    >>> calc = get_calculator("m3gnet")
    >>> calc = get_calculator(model_path="/data/my_finetuned.model")
    """
    # ── Custom model path → MACE backend ──────────────────────────────────
    if model_path is not None:
        return _load_mace(model, device=device, model_path=model_path, **kwargs)

    # ── Route by backend ──────────────────────────────────────────────────
    backend = _detect_backend(model)

    if backend == "mace":
        return _load_mace(model, device=device, **kwargs)
    elif backend == "chgnet":
        return _load_chgnet(device=device, **kwargs)
    elif backend == "m3gnet":
        return _load_m3gnet(model, device=device, **kwargs)
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
        all other backends (CHGNet, M3GNet).

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
    bar = "─" * 72
    print(f"\n{bar}")
    print("  Available foundation models  (pass as --model NAME)")
    print(bar)

    # Group by backend
    sections = [
        ("MACE", {k: v for k, v in MODEL_DESCRIPTIONS.items()
                  if k.startswith("mace-")}),
        ("CHGNet", {k: v for k, v in MODEL_DESCRIPTIONS.items()
                    if k == "chgnet"}),
        ("M3GNet / MatGL", {k: v for k, v in MODEL_DESCRIPTIONS.items()
                            if k.startswith("m3gnet")}),
    ]

    for backend_name, models in sections:
        print(f"\n  ── {backend_name} {'─' * (60 - len(backend_name))}")
        for name, desc in models.items():
            print(f"  {name:<25s}  {desc}")

    print(f"\n{bar}")
    print("  Custom model:  --model-path /path/to/my_finetuned.model")
    print(f"{bar}\n")
