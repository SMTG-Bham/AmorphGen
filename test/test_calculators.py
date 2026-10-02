"""Calculator dispatch and dtype contracts, without optional ML backends."""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from amorphgen.utils.calculators import get_calculator, list_models, _load_chgnet, _load_mace


class TestListModels:
    def test_runs_and_prints_known_backends(self, capsys):
        list_models()
        out = capsys.readouterr().out.lower()
        for backend in ("mace", "chgnet", "sevennet", "classical"):
            assert backend in out


class TestDeviceAuto:
    @pytest.mark.parametrize("cuda, mps, expected", [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ])
    def test_auto_resolves_device_in_priority_order(self, monkeypatch, cuda, mps, expected):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        )
        monkeypatch.setitem(sys.modules, "torch", torch)
        with patch("amorphgen.utils.calculators._load_classical") as load:
            result = get_calculator("lj", device="auto")
        load.assert_called_once_with("lj", device=expected)
        assert result is load.return_value

    def test_auto_without_torch_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        with patch("amorphgen.utils.calculators._load_classical") as load:
            get_calculator("lj", device="auto")
        load.assert_called_once_with("lj", device="cpu")

    @pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1", "mps"])
    def test_explicit_device_is_preserved(self, monkeypatch, device):
        # Explicit selection must not depend on importing torch.
        monkeypatch.setitem(sys.modules, "torch", None)
        with patch("amorphgen.utils.calculators._load_classical") as load:
            get_calculator("lj", device=device)
        load.assert_called_once_with("lj", device=device)


class TestBackendRouting:
    @pytest.mark.parametrize("model, loader, expected_args, expected_kwargs", [
        ("mace-mpa-0", "_load_mace", ("mace-mpa-0",), {"default_dtype": "float64"}),
        ("chgnet", "_load_chgnet", (), {"default_dtype": "float32"}),
        ("sevennet", "_load_sevennet", ("sevennet",), {"default_dtype": "float64"}),
        ("lennard-jones", "_load_classical", ("lennard-jones",), {}),
        ("lj", "_load_classical", ("lj",), {}),
        ("buckingham", "_load_classical", ("buckingham",), {}),
        ("buck", "_load_classical", ("buck",), {}),
    ])
    def test_routes_to_backend(self, model, loader, expected_args, expected_kwargs):
        with patch(f"amorphgen.utils.calculators.{loader}") as load:
            result = get_calculator(model, device="cpu")
        load.assert_called_once_with(*expected_args, device="cpu", **expected_kwargs)
        assert result is load.return_value

    def test_classical_parameters_are_forwarded(self):
        params = {"params": {}, "charges": {}, "cutoff": 5.0}
        with patch("amorphgen.utils.calculators._load_classical") as load:
            get_calculator("buckingham", device="cpu", classical_params=params)
        load.assert_called_once_with("buckingham", device="cpu", classical_params=params)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unrecognised model"):
            get_calculator("totally-not-a-model-xyz", device="cpu")

    def test_model_path_takes_priority(self):
        with patch("amorphgen.utils.calculators._load_mace") as load:
            result = get_calculator("chgnet", device="cpu", model_path="custom.model")
        load.assert_called_once_with("chgnet", device="cpu", model_path="custom.model",
                                     default_dtype="float64")
        assert result is load.return_value


def test_invalid_mace_model_path_does_not_load_weights(tmp_path, monkeypatch):
    loaders = SimpleNamespace(mace_mp=Mock(), MACECalculator=Mock())
    monkeypatch.setitem(sys.modules, "mace.calculators", loaders)
    with pytest.raises(FileNotFoundError, match="Custom MACE model file not found"):
        _load_mace("mace-mpa-0", device="cpu", model_path=str(tmp_path / "missing.model"))
    loaders.mace_mp.assert_not_called()
    loaders.MACECalculator.assert_not_called()


@pytest.fixture
def chgnet_stubs(monkeypatch):
    """Exercise the loader without loading weights or changing real torch state."""
    torch = SimpleNamespace(float32=object(), float64=object(), set_default_dtype=Mock())
    model_type = Mock()
    calculator = Mock()
    package = ModuleType("chgnet")
    model_package = ModuleType("chgnet.model")
    model_module = ModuleType("chgnet.model.model")
    dynamics = ModuleType("chgnet.model.dynamics")
    model_module.CHGNet = model_type
    model_module.TORCH_DTYPE = torch.float64
    dynamics.CHGNetCalculator = calculator
    package.model = model_package
    model_package.model = model_module
    model_package.dynamics = dynamics
    for name, module in {
        "torch": torch,
        "chgnet": package,
        "chgnet.model": model_package,
        "chgnet.model.model": model_module,
        "chgnet.model.dynamics": dynamics,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(torch=torch, module=model_module,
                           model_type=model_type, calculator=calculator)


class TestChgnetDefaultDtype:
    @pytest.mark.parametrize("dtype, error, message", [
        ("bfloat16", ValueError, "default_dtype"),
        ("float64", NotImplementedError, "composition_model"),
    ])
    def test_unsupported_dtype_fails_before_loading(self, chgnet_stubs, dtype, error, message):
        with pytest.raises(error, match=message):
            _load_chgnet(device="cpu", default_dtype=dtype)
        chgnet_stubs.model_type.load.assert_not_called()
        chgnet_stubs.calculator.assert_not_called()
        chgnet_stubs.torch.set_default_dtype.assert_not_called()

    @pytest.mark.parametrize("dtype", [None, "float32"])
    def test_resets_dtype_and_consumes_default_dtype_kwarg(self, chgnet_stubs, dtype):
        stubs = chgnet_stubs
        result = _load_chgnet(device="cpu", default_dtype=dtype, stress_weight=0.5)
        stubs.torch.set_default_dtype.assert_called_once_with(stubs.torch.float32)
        assert stubs.module.TORCH_DTYPE is stubs.torch.float32
        stubs.model_type.load.assert_called_once_with(use_device="cpu")
        stubs.calculator.assert_called_once_with(
            model=stubs.model_type.load.return_value, use_device="cpu", stress_weight=0.5,
        )
        assert result is stubs.calculator.return_value


# ─── Backend availability / fail-fast checks ─────────────────────────────

class TestBackendAvailability:
    """require_backend / available_backends / list_models markers."""

    def test_classical_always_available(self):
        from amorphgen.utils.calculators import backend_available
        assert backend_available("classical") is True

    def test_available_backends_shape(self):
        from amorphgen.utils.calculators import available_backends
        avail = available_backends()
        assert set(avail) == {"mace", "chgnet", "sevennet", "classical"}
        assert all(isinstance(v, bool) for v in avail.values())
        assert avail["classical"] is True

    def test_require_backend_classical_passes_on_bare_install(self):
        from amorphgen.utils.calculators import require_backend
        assert require_backend("lj") == "classical"
        assert require_backend("buckingham") == "classical"

    def test_require_backend_missing_raises_with_install_hint(self, monkeypatch):
        """The fail-fast message must contain a copy-pasteable install line."""
        import amorphgen.utils.calculators as calc
        monkeypatch.setattr(calc, "backend_available",
                            lambda b: b == "classical")
        with pytest.raises(calc.BackendNotInstalledError) as exc:
            calc.require_backend("mace-mpa-0")
        msg = str(exc.value)
        assert 'pip install "amorphgen[mace]"' in msg
        assert "classical" in msg          # torch-free alternative offered
        assert "--list-models" in msg

    def test_require_backend_model_path_implies_mace(self, monkeypatch):
        import amorphgen.utils.calculators as calc
        monkeypatch.setattr(calc, "backend_available",
                            lambda b: b == "classical")
        with pytest.raises(calc.BackendNotInstalledError):
            calc.require_backend("ignored", model_path="/tmp/custom.model")

    def test_require_backend_unknown_model_valueerror(self):
        from amorphgen.utils.calculators import require_backend
        with pytest.raises(ValueError, match="Unrecognised model"):
            require_backend("not-a-real-model")

    def test_list_models_shows_markers(self, capsys, monkeypatch):
        """Full registry is shown regardless of installs, with per-backend
        installed / not-installed markers (D4)."""
        import amorphgen.utils.calculators as calc
        monkeypatch.setattr(calc, "backend_available",
                            lambda b: b in ("classical", "chgnet"))
        calc.list_models()
        out = capsys.readouterr().out
        assert "mace-mpa-0" in out                       # registry complete
        assert "[installed]" in out                      # chgnet marked
        assert 'pip install "amorphgen[mace]"' in out    # missing marked
        assert "[built-in]" in out                       # classical


class TestRequiresCalculator:
    """CLI gate: which modes trigger the fail-fast (D2)."""

    def _args(self, **kw):
        from amorphgen.cli import _get_parser
        ns = _get_parser().parse_args([])
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    def test_calculator_free_modes(self):
        from amorphgen.cli import _requires_calculator
        assert not _requires_calculator(self._args())                     # nothing
        assert not _requires_calculator(self._args(analyse=True))
        assert not _requires_calculator(self._args(random_gen=True))      # no --relax
        assert not _requires_calculator(self._args(list_models=True))
        assert not _requires_calculator(self._args(convert="x.xyz"))

    def test_calculator_modes(self):
        from amorphgen.cli import _requires_calculator
        assert _requires_calculator(self._args(random_gen=True, relax=True))
        assert _requires_calculator(self._args(batch_opt=True))
        assert _requires_calculator(self._args(batch_quench=True))
        assert _requires_calculator(self._args(mq_ensemble=True))
        assert _requires_calculator(self._args(hybrid_ensemble=True))
        assert _requires_calculator(self._args(input_file="POSCAR"))      # pipeline


class TestDtypeFailFast:
    """require_dtype and its CLI gate: CHGNet + float64 must fail before any
    work starts, not in --mq-ensemble phase 3 after stages 1-4 of MD."""

    def test_chgnet_float64_raises(self):
        from amorphgen.utils.calculators import require_dtype
        with pytest.raises(NotImplementedError, match="composition_model"):
            require_dtype("chgnet", "float64")

    @pytest.mark.parametrize("model, dtype", [
        ("chgnet", None), ("chgnet", "auto"), ("chgnet", "float32"),
        ("mace-mpa-0", "float64"), ("lj", "float64"),
    ])
    def test_supported_combinations_pass(self, model, dtype):
        from amorphgen.utils.calculators import require_dtype
        require_dtype(model, dtype)

    def test_model_path_implies_mace(self):
        from amorphgen.utils.calculators import require_dtype
        require_dtype("chgnet", "float64", model_path="/tmp/custom.model")

    class _ReachedStages(Exception):
        """Raised in place of stages 1-4, i.e. when the gate let the run through."""

    def _mq_ensemble(self, tmp_path, monkeypatch, yaml_text):
        import sys
        import amorphgen.utils.calculators as calc
        import amorphgen.pipeline.run_pipeline as rp
        from amorphgen.cli import main

        def _pipeline(*args, **kwargs):
            raise self._ReachedStages
        monkeypatch.setattr(calc, "backend_available", lambda b: True)
        monkeypatch.setattr(rp, "MeltQuenchPipeline", _pipeline)
        cfg = tmp_path / "mq.yaml"
        cfg.write_text(yaml_text)
        monkeypatch.setattr(sys, "argv", ["amorphgen", "POSCAR", "--mq-ensemble",
                                          "--config", str(cfg), "-o", str(tmp_path / "mq")])
        main()

    def test_cli_refuses_chgnet_float64_before_stages(self, tmp_path, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            self._mq_ensemble(tmp_path, monkeypatch,
                              "model: chgnet\ndefault_dtype: float64\n")
        assert exc.value.code == 1
        assert "default_dtype='float64'" in capsys.readouterr().out
        assert not (tmp_path / "mq").exists()          # no setup work done

    def test_cli_lets_chgnet_auto_through(self, tmp_path, monkeypatch):
        with pytest.raises(self._ReachedStages):
            self._mq_ensemble(tmp_path, monkeypatch, "model: chgnet\n")


def test_real_chgnet_uses_float32_weights_and_computes_finite_results():
    """Exercise the installed backend's bundled checkpoint when available."""
    pytest.importorskip("chgnet")
    import numpy as np
    import torch
    import chgnet.model.model as chgnet_model
    from ase.build import bulk

    original_dtype = torch.get_default_dtype()
    original_chgnet_dtype = chgnet_model.TORCH_DTYPE
    try:
        torch.set_default_dtype(torch.float64)
        chgnet_model.TORCH_DTYPE = torch.float64
        calc = _load_chgnet(device="cpu", default_dtype="float32")
        parameters = tuple(calc.model.parameters())
        assert parameters
        assert all(parameter.dtype is torch.float32 for parameter in parameters)
        atoms = bulk("Si", "diamond", a=5.43, cubic=True)
        atoms.calc = calc
        assert np.isfinite(atoms.get_potential_energy())
        forces = atoms.get_forces()
        assert forces.shape == (len(atoms), 3)
        assert np.isfinite(forces).all()
    finally:
        torch.set_default_dtype(original_dtype)
        chgnet_model.TORCH_DTYPE = original_chgnet_dtype
