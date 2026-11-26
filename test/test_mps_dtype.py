"""
test/test_mps_dtype.py
----------------------
MACE on Apple-silicon GPUs (PyTorch "mps"). MPS has no float64 and mace-torch
casts energies to float64 inside the model, so MACE cannot run on MPS in any
precision. On a Mac, device "auto" resolves to "mps": get_calculator must
fall back to the CPU, and the CLI must refuse an explicit --device mps before
any work. Other backends are unchanged. No torch or MLIP package is needed.
"""

from unittest.mock import patch

import pytest

import amorphgen.utils.calculators as calc_mod
from amorphgen.utils.calculators import get_calculator, require_dtype


def _captured(loader):
    seen = {}

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return object()
    return seen, patch.object(calc_mod, loader, side_effect=fake)


class TestGetCalculator:

    @pytest.mark.parametrize("dtype", ["auto", "float32", "float64"])
    def test_mace_on_mps_falls_back_to_cpu(self, dtype, capsys):
        seen, p = _captured("_load_mace")
        with p, pytest.warns(UserWarning, match="cannot run on 'mps'"):
            get_calculator("mace-mpa-0", device="mps", default_dtype=dtype)
        assert seen["device"] == "cpu"
        assert "running on cpu" in capsys.readouterr().out

    def test_mace_model_path_on_mps_falls_back(self, tmp_path):
        seen, p = _captured("_load_mace")
        with p, pytest.warns(UserWarning):
            get_calculator(model_path=str(tmp_path / "my.model"), device="mps")
        assert seen["device"] == "cpu"

    def test_auto_device_on_a_mac(self, monkeypatch):
        """Stages pass "auto"; on a Mac it resolves to mps, then to cpu."""
        monkeypatch.setattr(calc_mod, "resolve_device", lambda d: "mps")
        seen, p = _captured("_load_mace")
        with p, pytest.warns(UserWarning):
            get_calculator("mace-mpa-0", device="auto")
        assert seen == {"device": "cpu", "default_dtype": "float64"}

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_mace_elsewhere_unchanged(self, device):
        seen, p = _captured("_load_mace")
        with p:
            get_calculator("mace-mpa-0", device=device)
        assert seen == {"device": device, "default_dtype": "float64"}

    def test_chgnet_on_mps_unchanged(self):
        seen, p = _captured("_load_chgnet")
        with p:
            get_calculator("chgnet", device="mps")
        assert seen["device"] == "mps"

    def test_sevennet_on_mps_unchanged(self):
        seen, p = _captured("_load_sevennet")
        with p:
            get_calculator("7net-mf-ompa", device="mps")
        assert seen["device"] == "mps"


class TestRequireDtype:

    @pytest.mark.parametrize("dtype", ["auto", "float32", "float64"])
    def test_explicit_mps_with_mace_fails_fast(self, dtype):
        with pytest.raises(ValueError, match="cannot run on device 'mps'"):
            require_dtype("mace-mpa-0", dtype, device="mps")

    def test_explicit_mps_with_mace_model_path(self):
        with pytest.raises(ValueError, match="--device cpu"):
            require_dtype("mace-mpa-0", "auto", model_path="my.model", device="mps")

    @pytest.mark.parametrize("device", ["auto", "cpu", "cuda", None])
    def test_other_devices_pass(self, device):
        require_dtype("mace-mpa-0", "float64", device=device)

    @pytest.mark.parametrize("model", ["chgnet", "7net-mf-ompa", "lj"])
    def test_other_backends_on_mps_pass(self, model):
        require_dtype(model, "float32", device="mps")

    def test_chgnet_float64_message_unchanged(self):
        with pytest.raises(NotImplementedError, match="CHGNet"):
            require_dtype("chgnet", "float64", device="mps")


class TestCli:

    def _run(self, tmp_path, monkeypatch, *extra):
        from amorphgen import cli
        poscar = tmp_path / "POSCAR"
        poscar.write_text("Si\n1.0\n5.43 0 0\n0 5.43 0\n0 0 5.43\nSi\n1\nDirect\n0 0 0\n")
        monkeypatch.setattr(calc_mod, "backend_available", lambda b: True)
        monkeypatch.setattr("sys.argv", ["amorphgen", str(poscar), *extra,
                                         "--work-dir", str(tmp_path / "run")])
        return cli

    def test_explicit_mps_exits_before_work(self, tmp_path, monkeypatch, capsys):
        cli = self._run(tmp_path, monkeypatch, "--device", "mps")
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        assert "cannot run on device 'mps'" in capsys.readouterr().out
        assert not (tmp_path / "run").exists()
