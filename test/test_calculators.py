"""Tests for amorphgen.utils.calculators — backend dispatch + helpers.

These tests use mock patching to exercise dispatch logic without needing
the actual MLIP backends (MACE / CHGNet / SevenNet) installed.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from amorphgen.utils.calculators import (
    get_calculator,
    list_models,
    CLASSICAL_MODELS,
)


# ─── list_models ───────────────────────────────────────────────────────────

class TestListModels:
    def test_runs_and_prints_known_backends(self, capsys):
        list_models()
        out = capsys.readouterr().out
        # Should at minimum mention the four backends.
        assert "MACE" in out
        assert "CHGNet" in out.upper() or "chgnet" in out
        assert any(token in out for token in ("SevenNet", "sevennet", "7net"))
        assert any(token in out.lower() for token in ("classical", "buckingham", "lennard"))


# ─── device='auto' resolution ──────────────────────────────────────────────

class TestDeviceAuto:
    def test_auto_resolves_to_real_device(self):
        """device='auto' should pick CUDA / MPS / CPU and forward a real string."""
        with patch("amorphgen.utils.calculators._load_classical") as mock_load:
            mock_load.return_value = MagicMock()
            get_calculator("buckingham", device="auto",
                            classical_params={
                                "params": {("Si", "O"): {"A": 0, "rho": 1, "C": 0}},
                                "charges": {"Si": 0, "O": 0},
                                "cutoff": 5.0,
                            })
            assert mock_load.called
            forwarded_device = mock_load.call_args.kwargs.get("device")
            assert forwarded_device in ("cuda", "mps", "cpu")
            assert forwarded_device != "auto"

    def test_explicit_cpu_is_preserved(self):
        with patch("amorphgen.utils.calculators._load_classical") as mock_load:
            mock_load.return_value = MagicMock()
            get_calculator("buckingham", device="cpu",
                            classical_params={
                                "params": {}, "charges": {}, "cutoff": 5.0,
                            })
            assert mock_load.call_args.kwargs.get("device") == "cpu"


# ─── Backend routing ───────────────────────────────────────────────────────

class TestBackendRouting:
    def test_mace_routes_to_load_mace(self):
        with patch("amorphgen.utils.calculators._load_mace") as mock_load:
            mock_load.return_value = MagicMock()
            get_calculator("mace-mpa-0", device="cpu")
            assert mock_load.called

    def test_chgnet_routes_to_load_chgnet(self):
        with patch("amorphgen.utils.calculators._load_chgnet") as mock_load:
            mock_load.return_value = MagicMock()
            get_calculator("chgnet", device="cpu")
            assert mock_load.called

    def test_sevennet_routes_to_load_sevennet(self):
        with patch("amorphgen.utils.calculators._load_sevennet") as mock_load:
            mock_load.return_value = MagicMock()
            get_calculator("sevennet", device="cpu")
            assert mock_load.called

    def test_classical_routes_to_load_classical(self):
        with patch("amorphgen.utils.calculators._load_classical") as mock_load:
            mock_load.return_value = MagicMock()
            get_calculator("lennard-jones", device="cpu",
                            classical_params={"params": {}, "charges": {}, "cutoff": 5.0})
            assert mock_load.called

    def test_unknown_model_raises(self):
        with pytest.raises((ValueError, ImportError)):
            get_calculator("totally-not-a-model-xyz", device="cpu")


# ─── model_path takes priority over model name ────────────────────────────

class TestModelPath:
    def test_model_path_routes_to_mace(self):
        """A model_path argument should always route to the MACE loader,
        regardless of the model name string."""
        with patch("amorphgen.utils.calculators._load_mace") as mock_load:
            mock_load.return_value = MagicMock()
            get_calculator(model="chgnet",
                           device="cpu",
                           model_path="/fake/path/to/model.model")
            assert mock_load.called
            kwargs = mock_load.call_args.kwargs
            assert kwargs.get("model_path") == "/fake/path/to/model.model"


# ─── CLASSICAL_MODELS registry ────────────────────────────────────────────

class TestClassicalModelsRegistry:
    def test_known_aliases_present(self):
        for name in ("lennard-jones", "lj", "buckingham", "buck"):
            assert name in CLASSICAL_MODELS, f"{name} missing from CLASSICAL_MODELS"

    def test_classical_model_detected_by_name(self):
        """A classical model name should route to _load_classical even
        without classical_params kwarg surviving the dispatch (the loader
        will raise its own validation error later)."""
        with patch("amorphgen.utils.calculators._load_classical") as mock_load:
            mock_load.return_value = MagicMock()
            for name in ("lennard-jones", "lj", "buckingham", "buck"):
                get_calculator(name, device="cpu",
                                classical_params={"params": {}, "charges": {}, "cutoff": 5.0})
            assert mock_load.call_count == 4


# ─── CHGNet default_dtype handling (regression: was a silent no-op) ──────────

class TestChgnetDefaultDtype:
    """Pre-2026-05-11 the CHGNet loader accepted ``default_dtype`` via
    ``**kwargs`` and silently forwarded it to ``CHGNetCalculator``, which
    ignored unknown kwargs.  So YAML configs with ``default_dtype: float64``
    were a silent no-op — users believed they were running fp64 but got fp32.

    Post-fix: ``default_dtype`` is recognised, ``float32`` and ``None`` set
    torch's default + chgnet's module-level ``TORCH_DTYPE`` to float32, and
    ``float64`` raises ``NotImplementedError`` with a clear message (because
    CHGNet's composition_model submodule constructs input tensors via a path
    that bypasses ``TORCH_DTYPE`` and crashes at forward time on fp64).

    Skipped when chgnet is not installed.
    """

    def _chgnet_or_skip(self):
        pytest.importorskip("chgnet")
        return True

    def test_default_dtype_invalid_raises(self):
        self._chgnet_or_skip()
        from amorphgen.utils.calculators import _load_chgnet
        with pytest.raises(ValueError, match="default_dtype"):
            _load_chgnet(device="cpu", default_dtype="bfloat16")

    def test_default_dtype_float64_raises_not_implemented(self):
        """float64 must raise a clear NotImplementedError pointing the user
        to MACE — silently downcasting would re-introduce the original bug."""
        self._chgnet_or_skip()
        from amorphgen.utils.calculators import _load_chgnet
        with pytest.raises(NotImplementedError, match="composition_model"):
            _load_chgnet(device="cpu", default_dtype="float64")

    def test_default_dtype_float32_sets_torch_default(self):
        self._chgnet_or_skip()
        import torch
        from amorphgen.utils.calculators import _load_chgnet
        # Pre-set to float64 to verify the loader resets it.
        torch.set_default_dtype(torch.float64)
        _load_chgnet(device="cpu", default_dtype="float32")
        assert torch.get_default_dtype() is torch.float32
        # And chgnet's module-level constant matches.
        import chgnet.model.model as _mod
        assert _mod.TORCH_DTYPE is torch.float32

    def test_default_dtype_none_keeps_float32(self):
        """Passing default_dtype=None should give CHGNet's native float32."""
        self._chgnet_or_skip()
        import torch
        from amorphgen.utils.calculators import _load_chgnet
        _load_chgnet(device="cpu", default_dtype=None)
        assert torch.get_default_dtype() is torch.float32

    def test_model_weights_are_float32(self):
        """The loaded CHGNet model's parameters should all be float32."""
        self._chgnet_or_skip()
        import torch
        from amorphgen.utils.calculators import _load_chgnet
        calc = _load_chgnet(device="cpu", default_dtype="float32")
        assert all(p.dtype is torch.float32 for p in calc.model.parameters())

    def test_default_dtype_stripped_from_calculator_kwargs(self):
        """default_dtype must not end up in CHGNetCalculator(**kwargs)
        — if it did, future CHGNet versions with stricter signature
        validation would reject it."""
        self._chgnet_or_skip()
        from amorphgen.utils.calculators import _load_chgnet
        # If default_dtype were leaking through, CHGNetCalculator might
        # complain.  We just check the call doesn't raise.
        _load_chgnet(device="cpu", default_dtype="float32")

    def test_default_dtype_recovers_after_prior_fp64_caller(self):
        """If an earlier caller set torch default to float64 (e.g. a MACE
        loader), _load_chgnet must reset to float32 — otherwise CHGNet's
        forward pass crashes on dtype mismatch."""
        self._chgnet_or_skip()
        import torch
        from amorphgen.utils.calculators import _load_chgnet
        torch.set_default_dtype(torch.float64)
        import chgnet.model.model as _mod
        _mod.TORCH_DTYPE = torch.float64   # simulate fp64-poisoned state
        _load_chgnet(device="cpu", default_dtype="float32")
        assert torch.get_default_dtype() is torch.float32
        assert _mod.TORCH_DTYPE is torch.float32


# ─── Backend availability / fail-fast (DESIGN_MLIP_OPTIONAL.md D2/D4) ──────

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
        """The fail-fast message must contain a copy-pasteable install line
        (invariant 4 of DESIGN_MLIP_OPTIONAL.md)."""
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
