"""
test/test_symmetrix_backend.py
------------------------------
Symmetrix backend (bonan-group/symmetrix-xl, imported as ``symmetrix``):
routing, the loader's dtype handling and its refusal of the original
wcwitt/symmetrix package, CLI inference and config validation, all with
stand-in ``symmetrix`` modules. The class at the end compares the real package with
mace-torch when both are installed and SYMMETRIX_TEST_JSON / _MODEL are set.
"""

import os
import sys
from types import ModuleType

import numpy as np
import pytest
from ase.build import bulk

import amorphgen.utils.calculators as calc_mod
from amorphgen.utils.calculators import (
    backend_for, get_calculator, is_symmetrix_file, require_potential,
)


class _OriginalSymmetrix:
    """Signature of wcwitt/symmetrix's calculator."""
    def __init__(self, model_file, dtype="float64", use_kokkos=True, **kwargs):
        self.args = dict(model_file=model_file, dtype=dtype,
                         use_kokkos=use_kokkos, **kwargs)


class _XLSymmetrix:
    """Signature of symmetrix-xl's calculator (no use_kokkos)."""
    def __init__(self, model_file, dtype="float32", streamed_edges="direct",
                 execution_profile="capacity", **ase_kwargs):
        self.args = dict(model_file=model_file, dtype=dtype,
                         streamed_edges=streamed_edges,
                         execution_profile=execution_profile, **ase_kwargs)


def _install_fake(monkeypatch, cls):
    module = ModuleType("symmetrix")
    module.Symmetrix = cls
    monkeypatch.setitem(sys.modules, "symmetrix", module)


@pytest.fixture
def model_json(tmp_path):
    path = tmp_path / "mace-8-14.json"
    path.write_text("{}")
    return str(path)


class TestRouting:

    @pytest.mark.parametrize("model, path, expected", [
        ("symmetrix", "m.json", "symmetrix"),
        ("SYMMETRIX", "m.json", "symmetrix"),
        ("mace-mpa-0", "m.json", "symmetrix"),   # the suffix implies Symmetrix
        ("mace-mpa-0", "M.JSON", "symmetrix"),
        ("symmetrix", "m.model", "symmetrix"),   # explicit name wins
        ("mace-mpa-0", "m.model", "mace"),
    ])
    def test_backend_for(self, model, path, expected):
        assert backend_for(model, path) == expected

    def test_symmetrix_needs_a_file(self):
        with pytest.raises(ValueError, match="model.json"):
            backend_for("symmetrix")

    def test_is_symmetrix_file(self):
        assert is_symmetrix_file("a/b.json")
        assert not is_symmetrix_file("a/b.model")
        assert not is_symmetrix_file(None)

    def test_dtype_auto_resolves_to_float64(self, monkeypatch, model_json):
        seen = {}
        monkeypatch.setattr(calc_mod, "_load_symmetrix",
                            lambda path, **kw: seen.update(path=path, **kw))
        get_calculator(model_path=model_json, device="cpu", default_dtype="auto")
        assert seen == {"path": model_json, "device": "cpu",
                        "default_dtype": "float64"}

    def test_params_only_for_symmetrix(self):
        with pytest.raises(ValueError, match="symmetrix_params"):
            get_calculator("mace-mpa-0", device="cpu",
                           symmetrix_params={"execution_profile": "speed"})


class TestLoader:

    def test_original_package_is_refused(self, monkeypatch, model_json):
        """wcwitt/symmetrix imports under the same name; this version runs
        Symmetrix-XL only."""
        _install_fake(monkeypatch, _OriginalSymmetrix)
        with pytest.raises(calc_mod.BackendNotInstalledError) as exc:
            get_calculator(model_path=model_json, device="cpu")
        assert "supports Symmetrix-XL" in str(exc.value)
        assert "pip install symmetrix-xl" in str(exc.value)

    def test_xl_flavour_gets_float64_not_its_float32_default(
            self, monkeypatch, model_json, capsys):
        _install_fake(monkeypatch, _XLSymmetrix)
        calc = get_calculator(model_path=model_json, device="cpu",
                              symmetrix_params={"execution_profile": "speed"})
        assert calc.args["dtype"] == "float64"
        assert calc.args["execution_profile"] == "speed"
        assert "(symmetrix-xl, dtype=float64)" in capsys.readouterr().out

    def test_deepcopy_of_atoms_shares_the_calculator(self, monkeypatch, model_json):
        """The stages deep-copy their input Atoms; the C++ evaluator can't be."""
        from copy import deepcopy

        class _Uncopyable(_XLSymmetrix):
            def __reduce_ex__(self, protocol):
                raise TypeError("cannot pickle 'symmetrix.symmetrix.MACEKokkos' object")

        _install_fake(monkeypatch, _Uncopyable)
        calc = get_calculator(model_path=model_json, device="cpu")
        assert isinstance(calc, _Uncopyable)
        atoms = bulk("Si")
        atoms.calc = calc
        assert deepcopy(atoms).calc is calc

    def test_explicit_float32(self, monkeypatch, model_json):
        _install_fake(monkeypatch, _XLSymmetrix)
        calc = get_calculator(model_path=model_json, device="cpu",
                              default_dtype="float32")
        assert calc.args["dtype"] == "float32"

    def test_use_kokkos_rejected(self, monkeypatch, model_json):
        _install_fake(monkeypatch, _XLSymmetrix)
        with pytest.raises(ValueError, match="use_kokkos"):
            get_calculator(model_path=model_json, device="cpu",
                           symmetrix_params={"use_kokkos": True})

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="symmetrix_extract_mace"):
            get_calculator(model_path=str(tmp_path / "none.json"), device="cpu")

    def test_not_installed_message(self, monkeypatch, model_json):
        monkeypatch.setitem(sys.modules, "symmetrix", None)   # import fails
        with pytest.raises(calc_mod.BackendNotInstalledError) as exc:
            get_calculator(model_path=model_json, device="cpu")
        assert "pip install symmetrix-xl" in str(exc.value)

    def test_mps_device_note(self, monkeypatch, model_json, capsys):
        _install_fake(monkeypatch, _XLSymmetrix)
        get_calculator(model_path=model_json, device="mps")
        assert "Device 'mps' is not used" in capsys.readouterr().out


class TestChecks:

    def test_require_potential(self, tmp_path, model_json):
        require_potential("symmetrix", model_json)
        with pytest.raises(FileNotFoundError):
            require_potential("symmetrix", str(tmp_path / "none.json"))

    def test_torchsim_rejects_symmetrix(self, model_json):
        with pytest.raises(ValueError, match="torch-sim"):
            require_potential("symmetrix", model_json, engine="torchsim")

    def test_cli_infers_model_from_json(self):
        from amorphgen.cli import _infer_potential_model
        override = {"model_path": "m.json"}
        assert _infer_potential_model(override) is None
        assert override["model"] == "symmetrix"
        msg = _infer_potential_model({"model_path": "m.json", "model": "chgnet"})
        assert "--model symmetrix" in msg

    def test_yaml_schema(self, tmp_path):
        from amorphgen.configs.yaml_config import load_yaml_config
        good = tmp_path / "good.yaml"
        good.write_text("symmetrix_params:\n  execution_profile: speed\n  metal: true\n")
        cfg = load_yaml_config(str(good))
        assert cfg["symmetrix_params"] == {"execution_profile": "speed", "metal": True}
        for text in ("symmetrix_params:\n  bogus: 1\n",
                     "symmetrix_params:\n  use_kokkos: false\n"):     # the original's option
            bad = tmp_path / "bad.yaml"
            bad.write_text(text)
            with pytest.raises(ValueError, match="Unknown key"):
                load_yaml_config(str(bad))

    def test_list_models_shows_symmetrix(self, capsys):
        calc_mod.list_models()
        out = capsys.readouterr().out
        assert "Symmetrix (converted MACE .json)" in out
        assert "model.json" in out


# ═════════════════════════════════════════════════════════════════════════════
# Real package vs mace-torch (opt-in: needs both packages and model files)
#   SYMMETRIX_TEST_MODEL=my.model SYMMETRIX_TEST_JSON=my-14.json pytest ...
# The JSON must cover Si (Z=14).
# ═════════════════════════════════════════════════════════════════════════════

_MODEL = os.environ.get("SYMMETRIX_TEST_MODEL")
_JSON = os.environ.get("SYMMETRIX_TEST_JSON")


@pytest.mark.skipif(not (_MODEL and _JSON), reason="set SYMMETRIX_TEST_MODEL/_JSON")
class TestAgainstMace:

    def test_energy_forces_stress_match(self):
        pytest.importorskip("symmetrix")
        pytest.importorskip("mace")
        atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat(2)
        atoms.rattle(0.05, seed=1)
        ref = atoms.copy()
        ref.calc = get_calculator(model_path=_MODEL, device="cpu")
        atoms.calc = get_calculator(model_path=_JSON, device="cpu")
        n = len(atoms)
        assert abs(atoms.get_potential_energy() - ref.get_potential_energy()) / n < 1e-3
        assert np.abs(atoms.get_forces() - ref.get_forces()).max() < 1e-2
        assert np.abs(atoms.get_stress() - ref.get_stress()).max() < 1e-3
