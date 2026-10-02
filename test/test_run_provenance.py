"""Run provenance remains useful without optional ML packages."""

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from amorphgen.utils.run_provenance import calculator_provenance


def test_local_checkpoint_hash_and_resolved_settings(tmp_path):
    path = tmp_path / "checkpoint.model"
    path.write_bytes(b"checkpoint weights")
    calc = SimpleNamespace(dtype="torch.float32", device="cuda:1")
    result = calculator_provenance({"model_path": path, "default_dtype": "auto"}, calc)
    assert result["model"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["model"]["path"] == str(path)
    assert result["model"]["hash_source"] == "file"
    assert result["precision"] == {"requested": "auto", "resolved": "float32"}
    assert result["device"] == {"requested": "auto", "resolved": "cuda:1"}
    json.dumps(result, allow_nan=False)


def test_injected_calculator_never_inherits_configured_model(tmp_path):
    path = tmp_path / "unrelated.model"
    path.write_bytes(b"not this calculator")
    calc = SimpleNamespace(name="external")
    result = calculator_provenance({"model": "mace-mpa-0", "model_path": path,
                                    "default_dtype": "float64", "device": "cuda"},
                                   calc, injected=True)
    assert result["model"]["name"] == "external"
    assert result["model"]["path"] is None
    assert result["model"]["sha256"] is None
    assert result["model"]["hash_unavailable_reason"]
    assert result["precision"]["resolved"] is None
    assert result["device"]["resolved"] is None


def test_local_mace_file_selected_through_model_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "mace-custom.model"
    path.write_bytes(b"checkpoint")
    result = calculator_provenance({"model": "mace-custom.model:head"})
    assert result["model"]["path"] == str(path)
    assert result["model"]["sha256"] == hashlib.sha256(b"checkpoint").hexdigest()


def test_state_hash_is_deterministic_and_covers_all_weights():
    first = np.array([1, 2], dtype=np.float32)
    second = np.array([[3]], dtype=np.float64)

    def provenance(state):
        model = SimpleNamespace(state_dict=lambda: state)
        return calculator_provenance({}, SimpleNamespace(models=[model]))["model"]

    result = provenance({"a": first, "b": second})
    assert result["hash_source"] == "state_dict-v1"
    assert result["sha256"] == provenance({"b": second, "a": first})["sha256"]
    assert result["sha256"] != provenance({"a": first + 1, "b": second})["sha256"]
    assert result["sha256"] != provenance({"a": first.astype(np.float64), "b": second})["sha256"]
    assert result["sha256"] != provenance({"a": first.reshape(1, 2), "b": second})["sha256"]


def test_torch_style_state_is_detached_and_hashed_without_torch_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    class Tensor:
        dtype = "torch.float32"
        shape = (2,)

        def detach(self):
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

        def numpy(self):
            return np.array([1, 2], dtype=np.float32)

    model = SimpleNamespace(state_dict=lambda: {"weight": Tensor()})
    result = calculator_provenance({}, SimpleNamespace(model=model), injected=True)
    assert result["model"]["sha256"] is not None
    assert result["model"]["hash_source"] == "state_dict-v1"


def test_parameters_determine_effective_settings_and_wrappers_are_unwrapped():
    model = SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(dtype="torch.float32", device="cpu")]),
        dtype="torch.float64", device="cuda",
    )
    calc = SimpleNamespace(base_calculator=SimpleNamespace(model=model))
    result = calculator_provenance({}, calc, injected=True)
    assert result["precision"]["resolved"] == "float32"
    assert result["device"]["resolved"] == "cpu"


@pytest.mark.parametrize("device, expected", [("cpu", "float64"), ("cuda", "float64"),
                                              ("mps", "float32")])
def test_classical_precision_reflects_actual_implementation(device, expected):
    result = calculator_provenance({"model": "lj", "device": device,
                                    "default_dtype": "float32"}, SimpleNamespace())
    assert result["precision"]["resolved"] == expected


def test_missing_checkpoint_records_reason(tmp_path):
    result = calculator_provenance({"model_path": tmp_path / "missing.model"})
    assert result["model"]["sha256"] is None
    assert "FileNotFoundError" in result["model"]["hash_unavailable_reason"]


def test_backend_free_provenance_does_not_import_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    result = calculator_provenance({"model": "chgnet"}, SimpleNamespace())
    assert result["precision"]["resolved"] == "float32"
    assert result["device"]["resolved"] == "cpu"
    assert result["model"]["hash_unavailable_reason"]


def test_unloaded_calculator_has_no_effective_settings():
    result = calculator_provenance({"model": "mace-mpa-0", "device": "cuda",
                                    "default_dtype": "float64"})
    assert result["precision"]["resolved"] is None
    assert result["device"]["resolved"] is None


def test_real_torch_state_hash_covers_empty_buffers_and_scalars():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1, dtype=torch.float64)
    model.register_buffer("empty", torch.empty(0, 3))
    model.register_buffer("scalar", torch.tensor(3))
    calc = SimpleNamespace(models=[model])
    first = calculator_provenance({}, calc, injected=True)
    second = calculator_provenance({}, calc, injected=True)
    assert first["model"]["sha256"] is not None
    assert first["model"]["sha256"] == second["model"]["sha256"]
    assert first["precision"]["resolved"] == "float64"
    assert first["device"]["resolved"] == "cpu"


def test_broken_optional_introspection_does_not_mask_run_failures():
    class BrokenModel:
        @property
        def dtype(self):
            raise RuntimeError("dtype unavailable")

        def state_dict(self):
            raise RuntimeError("weights unavailable")

    result = calculator_provenance({}, SimpleNamespace(model=BrokenModel()))
    assert result["model"]["sha256"] is None
    assert "weights unavailable" in result["model"]["hash_unavailable_reason"]
