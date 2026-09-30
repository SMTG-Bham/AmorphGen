"""Model selection checks run without torch-sim or model downloads."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from amorphgen.utils import torchsim_engine as engine


@pytest.mark.parametrize("path_kind", ["missing", "directory", "empty"])
def test_invalid_explicit_model_path_fails_before_loading_backends(tmp_path, monkeypatch, path_kind):
    path = {"missing": str(tmp_path / "typo.model"),
            "directory": str(tmp_path), "empty": ""}[path_kind]
    require = Mock(side_effect=AssertionError("must validate the requested file first"))
    monkeypatch.setattr(engine, "_require", require)
    with pytest.raises(FileNotFoundError, match="Custom MACE model file not found"):
        engine.build_model("mace-mpa-0", model_path=path)
    require.assert_not_called()


@pytest.fixture
def mace_stubs(monkeypatch):
    model = Mock()
    foundation = Mock(return_value=object())
    monkeypatch.setattr(engine, "_require", lambda: None)
    monkeypatch.setattr(engine, "resolve_torch_device", lambda device: "cpu")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float64="float64", float32="float32"))
    monkeypatch.setitem(sys.modules, "torch_sim.models.mace", SimpleNamespace(MaceModel=model))
    monkeypatch.setitem(sys.modules, "mace.calculators.foundations_models", SimpleNamespace(mace_mp=foundation))
    return model, foundation


def test_existing_explicit_model_path_is_used_without_foundation_fallback(tmp_path, mace_stubs):
    wrapper, foundation = mace_stubs
    path = tmp_path / "custom.model"
    path.write_bytes(b"checkpoint")
    result = engine.build_model("mace-mpa-0", model_path=str(path), device="cpu")
    wrapper.assert_called_once_with(model=str(path), device="cpu", dtype="float64", compute_stress=True)
    foundation.assert_not_called()
    assert result is wrapper.return_value


@pytest.mark.parametrize("name", ["mace-mpa-0", "MACE-MPA-0"])
def test_named_foundation_model_still_resolves_without_model_path(name, mace_stubs):
    wrapper, foundation = mace_stubs
    result = engine.build_model(name, device="cpu", dtype="float32")
    foundation.assert_called_once_with(model="medium-mpa-0", return_raw_model=True,
                                       default_dtype="float32", device="cpu")
    wrapper.assert_called_once_with(model=foundation.return_value, device="cpu", dtype="float32",
                                    compute_stress=True)
    assert result is wrapper.return_value
