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


@pytest.mark.parametrize("name", ["chgnet", "buckingham", "buck"])
def test_unsupported_models_raise_clearly(name, mace_stubs):
    with pytest.raises(ValueError, match="no torch-sim implementation"):
        engine.build_model(name, device="cpu")


def test_mps_is_rejected_without_torchsim(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    with pytest.raises(ValueError, match="MPS"):
        engine.resolve_torch_device("mps")


@pytest.fixture
def batch_inputs(tmp_path, monkeypatch):
    """Keep chunk handling on the real file path, substituting only the backend."""
    from ase.build import bulk
    from ase.io import write

    src = tmp_path / "in"
    src.mkdir()
    for index in range(4):
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
        atoms.info["input_index"] = index
        write(src / f"s{index}.xyz", atoms, format="extxyz")
    monkeypatch.setattr(engine, "build_model", Mock(return_value=object()))
    # GPU cleanup must tolerate a bare installation too.
    monkeypatch.setitem(sys.modules, "torch", None)
    return src, tmp_path / "out"


@pytest.mark.parametrize("message", [
    "CUDA out of memory. Tried to allocate 2.00 GiB",
    "Failed to allocate 180 bytes on device 'cuda:0'",
    "torch.OutOfMemoryError",
])
def test_out_of_memory_splits_chunks_and_preserves_every_output(
    batch_inputs, monkeypatch, message,
):
    import numpy as np
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import read
    from amorphgen.pipeline.opt_cell import batch_optimize

    src, dest = batch_inputs
    attempts = []

    def relax(atoms_list, *args, **kwargs):
        attempts.append([a.info["input_index"] for a in atoms_list])
        if len(atoms_list) > 1:
            raise RuntimeError(message)
        for atoms in atoms_list:
            atoms.calc = SinglePointCalculator(
                atoms, energy=-float(atoms.info["input_index"] + 1),
                forces=np.zeros((len(atoms), 3)),
            )
        return atoms_list

    monkeypatch.setattr(engine, "batch_relax", relax)
    outputs = batch_optimize(str(src), str(dest), engine="torchsim", batch_size=4)
    assert attempts == [[0, 1, 2, 3], [0, 1], [0], [1], [2, 3], [2], [3]]
    assert outputs == [str(dest / f"s{i}_opt.xyz") for i in range(4)]
    for index, path in enumerate(outputs):
        atoms = read(path)
        assert atoms.info["input_index"] == index
        assert atoms.get_potential_energy() == -(index + 1)
        assert (dest / f"s{index}_opt.log").is_file()


@pytest.mark.parametrize("error, batch_size", [
    (RuntimeError("shape mismatch"), 4),
    (ValueError("composition must be a dict"), 4),
    (RuntimeError("CUDA out of memory"), 1),
])
def test_batch_errors_propagate_without_retry(batch_inputs, monkeypatch, error, batch_size):
    from amorphgen.pipeline.opt_cell import batch_optimize

    src, dest = batch_inputs
    relax = Mock(side_effect=error)
    monkeypatch.setattr(engine, "batch_relax", relax)
    with pytest.raises(type(error), match=str(error)):
        batch_optimize(str(src), str(dest), engine="torchsim", batch_size=batch_size)
    relax.assert_called_once()
    assert not list(dest.glob("*_opt.xyz"))
