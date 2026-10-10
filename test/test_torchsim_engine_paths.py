"""torch-sim engine model dispatch, device choice and batch identity guards."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from ase import Atoms

from amorphgen.utils import torchsim_engine as engine


def atoms_pair(distance=1.5, cell=8.0):
    return Atoms("Cu2", positions=[[2, 2, 2], [2 + distance, 2, 2]],
                 cell=[cell] * 3, pbc=True)


@pytest.fixture
def backend_stubs(monkeypatch):
    """Stand-ins for torch and the torch-sim model wrappers (no torch needed)."""
    stubs = SimpleNamespace(lj=Mock(), sevennet=Mock())
    monkeypatch.setattr(engine, "_require", lambda: None)
    monkeypatch.setattr(engine, "resolve_torch_device", lambda device: "cpu")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float64="float64", float32="float32"))
    monkeypatch.setitem(sys.modules, "torch_sim.models.lennard_jones",
                        SimpleNamespace(LennardJonesModel=stubs.lj))
    monkeypatch.setitem(sys.modules, "torch_sim.models.sevennet",
                        SimpleNamespace(SevenNetModel=stubs.sevennet))
    return stubs


@pytest.mark.parametrize("params", [None, {}, {"params": {}, "cutoff": 5.0}])
def test_lennard_jones_without_a_pair_is_rejected(backend_stubs, params):
    with pytest.raises(ValueError, match="Lennard-Jones needs classical_params with one pair"):
        engine.build_model("lj", device="cpu", classical_params=params)
    backend_stubs.lj.assert_not_called()


@pytest.mark.parametrize("name, checkpoint, extra", [
    ("sevennet", "7net-mf-ompa", {"modal": "mpa"}),
    ("7NET-MF-0", "7net-mf-0", {"modal": "mpa"}),
    ("7net-0", "7net-0", {}),
    ("7net-omat", "7net-omat", {}),
])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sevennet_resolves_checkpoint_modal_and_forces_float32(
        backend_stubs, capsys, name, checkpoint, extra, dtype):
    model = engine.build_model(name, device="cpu", dtype=dtype)
    assert model is backend_stubs.sevennet.return_value
    backend_stubs.sevennet.assert_called_once_with(checkpoint, device="cpu", dtype="float32", **extra)
    note = "SevenNet runs in float32" in capsys.readouterr().out
    assert note == (dtype == "float64")


def test_unknown_model_name_is_rejected(backend_stubs):
    with pytest.raises(ValueError, match="Unknown model 'orb-v3' for the torch-sim engine"):
        engine.build_model("orb-v3", device="cpu")
    backend_stubs.sevennet.assert_not_called()
    backend_stubs.lj.assert_not_called()


def test_auto_device_follows_cuda_availability(monkeypatch):
    torch = pytest.importorskip("torch")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert engine.resolve_torch_device("auto") == torch.device(expected)
    assert engine.resolve_torch_device(None) == torch.device(expected)
    assert engine.resolve_torch_device("CPU") == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert engine.resolve_torch_device("auto") == torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert engine.resolve_torch_device("Auto") == torch.device("cpu")


def _zero_model(with_stress=True):
    """A deterministic torch-sim model: zero energy and forces, optional stress."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_sim")
    from torch_sim.models.interface import ModelInterface

    class ZeroModel(ModelInterface):
        def __init__(self):
            super().__init__()
            self._device = torch.device("cpu")
            self._dtype = torch.float64
            self._compute_forces = True
            self._compute_stress = True
            self.with_stress = with_stress
            self.calls = 0

        def forward(self, state, **kwargs):
            self.calls += 1
            result = {"energy": torch.zeros(state.n_systems, dtype=self.dtype),
                      "forces": torch.zeros_like(state.positions)}
            if self.with_stress:
                result["stress"] = torch.zeros_like(state.cell)
            return result

    return ZeroModel()


def test_bridge_identity_tags_must_survive_and_stay_in_range():
    model = _zero_model()
    import torch
    import torch_sim as ts

    bridge = engine._TorchSafetyBridge([atoms_pair(), atoms_pair(cell=9)])
    state = ts.initialize_state([atoms_pair(), atoms_pair(cell=9)], model.device, model.dtype)
    with pytest.raises(RuntimeError, match="lost safety system identifiers"):
        bridge.ids(state)
    bridge.attach(state)
    assert bridge.ids(state) == [0, 1]
    assert bridge.ids(state[[1, 0]]) == [1, 0]
    for bad in ([0, 2], [-1, 0], [0]):
        state.system_extras[bridge._ID] = torch.tensor(bad)
        with pytest.raises(RuntimeError, match="invalid safety system identifiers"):
            bridge.ids(state)


def test_guarded_model_keeps_only_the_current_model_stress():
    model = _zero_model()
    import torch_sim as ts

    bridge = engine._TorchSafetyBridge([atoms_pair()])
    guarded = bridge.wrap_model(model)
    state = bridge.attach(ts.initialize_state([atoms_pair()], model.device, model.dtype))
    guarded(state)
    assert bridge._STRESS in state.system_extras
    # A later evaluation without stress must not leave the stale virial behind.
    model.with_stress = False
    result = guarded(state)
    assert "stress" not in result
    assert bridge._STRESS not in state.system_extras


def test_empty_batch_and_bad_settings_never_call_the_model():
    model = _zero_model()
    assert engine.batch_relax([], model) == []
    assert engine.estimate_batch_size(model, [], fallback=5) == 5
    with pytest.raises(ValueError, match="cell_filter 'triclinic' not supported.*'cubic'"):
        engine.batch_relax([atoms_pair()], model, cell_filter="triclinic", log=lambda *a: None)
    assert model.calls == 0


def test_reordered_optimizer_output_is_rejected(monkeypatch):
    model = _zero_model()
    import torch_sim as ts

    monkeypatch.setattr(ts, "optimize", lambda system, **kwargs: system[[1, 0]])
    with pytest.raises(RuntimeError, match="failed to restore the original structure order"):
        engine.batch_relax([atoms_pair(), atoms_pair(cell=9)], model, cell_filter="none",
                           log=lambda *a: None)


@pytest.mark.filterwarnings("ignore:All systems have reached the maximum number of steps")
def test_cell_convergence_without_state_stress_is_an_error(monkeypatch):
    model = _zero_model(with_stress=False)
    import torch_sim as ts

    # An optimizer that silently dropped the cell filter yields states with
    # forces but no stress; zero forces alone must not pass the cell criterion.
    optimize = ts.optimize
    monkeypatch.setattr(ts, "optimize",
                        lambda **kwargs: optimize(**{**kwargs, "init_kwargs": {}}))
    with pytest.raises(ValueError, match="cell relaxation requested but the state has no stress"):
        engine.batch_relax([atoms_pair()], model, cell_filter="cubic", max_steps=5,
                           log=lambda *a: None)


@pytest.mark.parametrize("n_inputs, peak_gib, expected", [
    (20, 1.0, 8),       # 0.5 x 16 GiB / 1 GiB
    (3, 1.0, 3),        # never more than the inputs
    (20, 9.0, 1),       # always at least one
    (20, 3.0, 2),       # floor of 8/3
])
def test_cuda_memory_probe_chunk_arithmetic(monkeypatch, n_inputs, peak_gib, expected):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_sim")
    from amorphgen.utils import torchsim_md

    gib, base = 2 ** 30, 123
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: base)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: base + int(peak_gib * gib))
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda *a: SimpleNamespace(total_memory=16 * gib))
    probe = Mock()
    monkeypatch.setattr(torchsim_md, "batch_nvt", probe)
    inputs = [atoms_pair(cell=8 + k) for k in range(n_inputs)]
    largest = Atoms("Cu4", positions=[[2, 2, 2], [4, 2, 2], [2, 4, 2], [4, 4, 2]],
                    cell=[9.0] * 3, pbc=True)
    inputs.insert(1, largest)
    lines = []
    model = SimpleNamespace(device=torch.device("cuda"), dtype=torch.float64)
    result = engine.estimate_batch_size(model, inputs[:n_inputs], md=True, log=lines.append)
    assert result == expected
    probe.assert_called_once()
    assert probe.call_args.args[0][0] is largest
    assert probe.call_args.args[2] == 300.0
    assert probe.call_args.kwargs["n_steps"] == 2
    assert f"chunks of {expected}" in lines[0] and "(MD)" in lines[0]
