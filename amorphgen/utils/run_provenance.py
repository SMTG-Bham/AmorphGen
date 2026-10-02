"""Best-effort calculator identity for persisted run manifests."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
import sys


def _attribute(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _class_name(obj):
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _text(value):
    return None if value is None else str(value)


def _models(calc):
    """Unwrap AmorphGen's core wrapper and find the loaded backend models."""
    seen = set()
    while calc is not None and id(calc) not in seen:
        seen.add(id(calc))
        base = _attribute(calc, "base_calculator")
        if base is None:
            break
        calc = base
    models = _attribute(calc, "models")
    if not isinstance(models, (list, tuple)):
        model = _attribute(calc, "model")
        models = [model] if model is not None else [calc]
    return calc, [model for model in models if model is not None]


def _settings(calc, models):
    dtype = device = None
    # Parameters reflect loaded weights even if a constructor kept an obsolete
    # dtype/device attribute after moving or casting its model.
    for model in models:
        parameters = _attribute(model, "parameters")
        if callable(parameters):
            try:
                for parameter in parameters():
                    candidate = _text(_attribute(parameter, "dtype"))
                    if candidate and ("float" in candidate or "complex" in candidate):
                        dtype = candidate
                        device = _text(_attribute(parameter, "device"))
                        break
            except Exception:
                pass
        dtype = dtype or _text(_attribute(model, "dtype"))
        device = device or _text(_attribute(model, "device"))
        if dtype is not None and device is not None:
            break
    dtype = dtype or _text(_attribute(calc, "dtype"))
    device = device or _text(_attribute(calc, "device"))
    return dtype.removeprefix("torch.") if dtype else None, device


def _resolved_device(requested):
    if requested != "auto":
        return requested
    # Model construction has already imported torch if it needed it. Merely
    # writing a manifest must not import an optional ML backend.
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return "cuda"
            mps = _attribute(_attribute(torch, "backends"), "mps")
            if mps is not None and mps.is_available():
                return "mps"
        except Exception:
            return None
    return "cpu"


def _state_hash(models):
    digest = hashlib.sha256(b"amorphgen-state-dict-v1\0")
    count = 0
    for index, model in enumerate(models):
        state_dict = _attribute(model, "state_dict")
        if not callable(state_dict):
            raise ValueError("Calculator does not expose loaded model weights")
        state = state_dict()
        if not isinstance(state, Mapping):
            raise ValueError("Model state_dict is not a mapping")
        for name in sorted(state):
            tensor = state[name]
            dtype = str(tensor.dtype)
            shape = list(tensor.shape)
            if callable(_attribute(tensor, "detach")):
                tensor = tensor.detach().cpu().contiguous().numpy()
            metadata = json.dumps([index, name, dtype, shape],
                                  separators=(",", ":")).encode("utf-8")
            data = memoryview(tensor.reshape(-1)).cast("B")
            digest.update(len(metadata).to_bytes(8, "big"))
            digest.update(metadata)
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
            count += 1
    if not count:
        raise ValueError("Calculator does not expose loaded model weights")
    return digest.hexdigest()


def calculator_provenance(cfg, calc=None, *, injected=False):
    """Return JSON-safe identity without making provenance a run dependency.

    File hashes identify checkpoint bytes; ``state_dict-v1`` hashes identify
    the loaded tensors, including their names, shapes and precision. They are
    deliberately different hash sources. Unavailable properties remain null.
    An injected calculator is never attributed to the configured model.
    """
    result = {
        "model": {
            "name": None, "path": None, "sha256": None,
            "hash_source": None, "hash_unavailable_reason": None,
            "calculator_class": _class_name(calc) if calc is not None else None,
        },
        "precision": {"requested": _text(cfg.get("default_dtype", "auto")),
                      "resolved": None},
        "device": {"requested": _text(cfg.get("device", "auto")),
                   "resolved": None},
    }
    model_info = result["model"]
    try:
        base, models = _models(calc)
        dtype, device = _settings(base, models)
        model_name = _text(cfg.get("model", "mace-mpa-0"))
        if injected:
            name = _attribute(base, "name")
            model_info["name"] = name if isinstance(name, str) else _class_name(base)
        else:
            model_info["name"] = model_name
            path = cfg.get("model_path")
            if path is None and model_name and model_name.lower().startswith("mace-"):
                candidate = model_name.split(":", 1)[0]
                if os.path.isfile(candidate):
                    path = candidate
            if path is not None:
                model_info["path"] = os.path.abspath(os.fspath(path))
            if calc is not None:
                device = device or _resolved_device(result["device"]["requested"])
            name = (model_name or "").lower()
            classical = path is None and name in {"lj", "lennard-jones", "buck", "buckingham"}
            if classical and calc is not None:
                # Classical calculators ignore default_dtype; CPU and CUDA
                # evaluate in float64, and MPS evaluates in float32.
                dtype = dtype or ("float32" if device == "mps" else "float64")
            elif dtype is None and calc is not None:
                requested = result["precision"]["requested"]
                if requested == "auto":
                    if path is not None or name.startswith(("mace", "7net", "sevennet")):
                        dtype = "float64"
                    elif name == "chgnet":
                        dtype = "float32"
                else:
                    dtype = requested
        # Injected built-in classical calculators have a known implementation,
        # even when the run's model selection is unrelated to the calculator.
        if (dtype is None and base is not None
                and type(base).__module__ == "amorphgen.utils.classical"):
            dtype = "float32" if device == "mps" else "float64"
        result["precision"]["resolved"] = dtype
        result["device"]["resolved"] = device
        if model_info["path"] is not None:
            digest = hashlib.sha256()
            with open(model_info["path"], "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            model_info["sha256"] = digest.hexdigest()
            model_info["hash_source"] = "file"
        else:
            model_info["sha256"] = _state_hash(models)
            model_info["hash_source"] = "state_dict-v1"
    except Exception as exc:
        model_info["hash_unavailable_reason"] = f"{type(exc).__name__}: {exc}"
    return result
