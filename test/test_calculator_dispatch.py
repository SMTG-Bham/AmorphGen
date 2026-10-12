"""Calculator loaders: name resolution, kwargs and messages, with stand-ins for the ML backends."""
from __future__ import annotations

import copy
import re
import sys
import warnings
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from ase.build import bulk

import amorphgen.utils.calculators as calc_mod
from amorphgen.utils.calculators import (
    MACE_FOUNDATION_MODELS,
    _load_mace,
    _load_sevennet,
    get_calculator,
)
from amorphgen.utils.classical import BuckinghamCalculator
from amorphgen.utils.lammps_potential import LAMMPSCalculator

NACL_BUCK = {"Na-Cl": {"A": 2000.0, "rho": 0.30, "C": 0.0},
             "Cl-Cl": {"A": 1000.0, "rho": 0.35, "C": 20.0}}
LJ_CU_AG = {"pair_style": "lj/cut 6.0",
            "pair_coeff": ["1 1 0.40 2.30", "2 2 0.35 2.60", "1 2 0.37 2.45"],
            "elements": ["Cu", "Ag"]}
CPU_NOTE = "Runs on the CPU (device '{}' is not used)"
FLOAT64_NOTE = "Evaluates in float64 (default_dtype 'float32' is not used)"
NOTE_CASES = [
    ("cpu", "auto", []),
    ("cpu", "float64", []),
    ("cpu", "float32", [FLOAT64_NOTE]),
    ("cuda", "auto", [CPU_NOTE.format("cuda")]),
    ("mps", "float32", [CPU_NOTE.format("mps"), FLOAT64_NOTE]),
]


@pytest.fixture
def mace(monkeypatch):
    fake = SimpleNamespace(mace_mp=Mock(name="mace_mp"),
                           MACECalculator=Mock(name="MACECalculator"))
    monkeypatch.setitem(sys.modules, "mace.calculators", fake)
    return fake


@pytest.fixture
def sevenn(monkeypatch):
    fake = SimpleNamespace(SevenNetCalculator=Mock(name="SevenNetCalculator"))
    monkeypatch.setitem(sys.modules, "sevenn.calculator", fake)
    return fake.SevenNetCalculator


def _torch_stub(monkeypatch):
    torch = SimpleNamespace(float32="float32", float64="float64", set_default_dtype=Mock())
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


@pytest.fixture
def chgnet(monkeypatch):
    torch = _torch_stub(monkeypatch)
    package, model_pkg = ModuleType("chgnet"), ModuleType("chgnet.model")
    model_mod, dynamics = ModuleType("chgnet.model.model"), ModuleType("chgnet.model.dynamics")
    model_mod.CHGNet, model_mod.TORCH_DTYPE = Mock(name="CHGNet"), torch.float32
    dynamics.CHGNetCalculator = Mock(name="CHGNetCalculator")
    package.model, model_pkg.model, model_pkg.dynamics = model_pkg, model_mod, dynamics
    for module in (package, model_pkg, model_mod, dynamics):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return SimpleNamespace(CHGNet=model_mod.CHGNet, calculator=dynamics.CHGNetCalculator)


@pytest.fixture
def ace(monkeypatch):
    """Stand-in for amorphgen.utils.ace_potential, which imports pyace."""
    fake = ModuleType("amorphgen.utils.ace_potential")
    fake.ACE_PARAM_KEYS = frozenset({"recursive_evaluator", "recursive", "fast_nl"})
    fake.ACECalculator = Mock(name="ACECalculator")
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    return fake.ACECalculator


# ─── Missing backends ─────────────────────────────────────────────────────

@pytest.mark.parametrize("model, module, hint", [
    ("mace-mpa-0", "mace.calculators", "pip install amorphgen[mace]"),
    ("chgnet", "chgnet.model.model", "pip install chgnet"),
    ("sevennet", "sevenn.calculator", "pip install sevenn"),
])
def test_missing_backend_package_gives_install_command(monkeypatch, model, module, hint):
    _torch_stub(monkeypatch)                     # the CHGNet loader imports torch first
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match=re.escape(hint)):
        get_calculator(model, device="cpu")


def test_missing_pyace_is_a_backend_not_installed_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "amorphgen.utils.ace_potential", None)
    pot = tmp_path / "pot.yace"
    pot.write_text("")
    with pytest.raises(calc_mod.BackendNotInstalledError) as exc:
        get_calculator(model_path=str(pot), device="cpu")
    assert 'pip install "amorphgen[ace]"' in str(exc.value)
    assert "python-ace" in str(exc.value)


# ─── MACE ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model, resolved, head", [
    ("mace-mpa-0", "medium-mpa-0", None),
    ("MACE-OMAT-0", "medium-omat-0", None),
    ("mace-mh-1:omat_pbe", MACE_FOUNDATION_MODELS["mace-mh-1"], "omat_pbe"),
])
def test_mace_registry_names_reach_mace_mp(mace, model, resolved, head):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)     # registered: no "not in registry"
        result = get_calculator(model, device="cpu")
    extra = {"head": head} if head else {}
    mace.mace_mp.assert_called_once_with(model=resolved, device="cpu",
                                         default_dtype="float64", **extra)
    mace.MACECalculator.assert_not_called()
    assert result is mace.mace_mp.return_value


def test_mace_explicit_head_and_url_without_head(mace):
    _load_mace("mace-mh-1", device="cpu", head="matpes_r2scan")
    mace.mace_mp.assert_called_once_with(model=MACE_FOUNDATION_MODELS["mace-mh-1"],
                                         device="cpu", head="matpes_r2scan")
    mace.mace_mp.reset_mock()
    url = "https://example.org/models/custom.model"      # '://' is not a head separator
    _load_mace(url, device="cpu")
    mace.mace_mp.assert_called_once_with(model=url, device="cpu")


@pytest.mark.parametrize("dtype, expected", [("auto", "float64"), ("float32", "float32")])
def test_mace_model_path_loads_the_file(mace, tmp_path, capsys, dtype, expected):
    path = tmp_path / "finetuned.model"
    path.write_text("")
    result = get_calculator(model_path=str(path), device="cpu", default_dtype=dtype)
    mace.MACECalculator.assert_called_once_with(model_paths=str(path), device="cpu",
                                                default_dtype=expected)
    mace.mace_mp.assert_not_called()
    assert result is mace.MACECalculator.return_value
    assert f"Loading custom model: {path}" in capsys.readouterr().out


def test_unregistered_mace_name_warns_then_loads_a_matching_local_file(mace, tmp_path,
                                                                        monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mace-finetuned.model").write_text("")
    with pytest.warns(UserWarning, match="not in MACE registry"):
        get_calculator("mace-finetuned.model", device="cpu")
    mace.MACECalculator.assert_called_once_with(model_paths="mace-finetuned.model",
                                                device="cpu", default_dtype="float64")
    mace.mace_mp.assert_not_called()


def test_get_mace_calculator_is_deprecated_but_still_loads(mace):
    with pytest.warns(DeprecationWarning, match="get_calculator"):
        result = calc_mod.get_mace_calculator("mace-mpa-0")
    mace.mace_mp.assert_called_once_with(model="medium-mpa-0", device="cuda")
    assert result is mace.mace_mp.return_value


# ─── CHGNet / SevenNet ────────────────────────────────────────────────────

def test_chgnet_on_mps_loads_on_cpu_then_casts_and_moves(chgnet):
    result = get_calculator("chgnet", device="mps")
    chgnet.CHGNet.load.assert_called_once_with(use_device="cpu")
    loaded = chgnet.CHGNet.load.return_value
    loaded.float.assert_called_once_with()
    loaded.float.return_value.to.assert_called_once_with("mps")
    chgnet.calculator.assert_called_once_with(
        model=loaded.float.return_value.to.return_value, use_device="mps")
    assert result is chgnet.calculator.return_value


@pytest.mark.parametrize("model, kwargs, checkpoint, modal", [
    ("sevennet", {}, "7net-mf-ompa", "mpa"),
    ("SevenNet-MF", {}, "7net-mf-ompa", "mpa"),
    ("7net-mf-0", {}, "7net-mf-0", "mpa"),
    ("sevennet", {"modal": "omat24"}, "7net-mf-ompa", "omat24"),
    ("7net-l3i5", {}, "7net-l3i5", None),
    ("7net-omni", {}, "7net-omni", None),
])
def test_sevennet_checkpoint_and_multi_fidelity_modal(sevenn, capsys, model, kwargs,
                                                      checkpoint, modal):
    result = _load_sevennet(model, device="cpu", **kwargs)
    extra = {"modal": modal} if modal else {}
    sevenn.assert_called_once_with(model=checkpoint, device="cpu", **extra)
    assert result is sevenn.return_value
    defaulted = "mf" in checkpoint and "modal" not in kwargs
    assert ("defaulting modal='mpa'" in capsys.readouterr().out) is defaulted


# ─── Classical ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs", [{}, {"classical_params": None}, {"classical_params": {}}])
def test_classical_model_without_parameters_says_what_is_needed(kwargs):
    with pytest.raises(ValueError, match="requires 'classical_params'"):
        get_calculator("buckingham", device="cpu", **kwargs)


def test_classical_params_from_yaml_build_the_same_calculator_without_mutation():
    params = {"params": NACL_BUCK, "charges": {"Na": 1.0, "Cl": -1.0}, "cutoff": 6.0,
              "alpha": 0.3, "coulomb_method": "Wolf"}
    original = copy.deepcopy(params)
    calc = get_calculator("buck", device="cpu", classical_params=params)
    assert params == original
    assert isinstance(calc, BuckinghamCalculator)
    assert (calc.cutoff, calc.alpha, calc.coulomb_method) == (6.0, 0.3, "wolf")
    assert set(calc.pair_params) == {("Na", "Cl"), ("Cl", "Na"), ("Cl", "Cl")}

    direct = BuckinghamCalculator(
        {tuple(key.split("-")): value for key, value in NACL_BUCK.items()},
        charges={"Na": 1.0, "Cl": -1.0}, cutoff=6.0, alpha=0.3, coulomb_method="wolf")
    atoms = bulk("NaCl", "rocksalt", a=5.64).repeat(2)
    atoms.rattle(0.1, seed=3)
    reference = atoms.copy()
    atoms.calc, reference.calc = calc, direct
    assert atoms.get_potential_energy() == reference.get_potential_energy()


# ─── ACE / LAMMPS ─────────────────────────────────────────────────────────

def test_ace_rejects_unknown_params_before_loading(ace, tmp_path):
    pot = tmp_path / "pot.yace"
    pot.write_text("")
    with pytest.raises(ValueError, match=r"Unknown ace_params key\(s\): active_set, fast$"):
        get_calculator(model_path=str(pot), device="cpu",
                       ace_params={"recursive": True, "fast": 1, "active_set": "a.asi"})
    ace.assert_not_called()


@pytest.mark.parametrize("device, dtype, notes", NOTE_CASES)
def test_ace_loads_absolute_path_and_notes_unused_settings(ace, tmp_path, monkeypatch,
                                                           capsys, device, dtype, notes):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pot.yace").write_text("")
    result = get_calculator(model_path="pot.yace", device=device, default_dtype=dtype,
                            ace_params={"recursive": True})
    ace.assert_called_once_with(str(tmp_path / "pot.yace"), recursive=True)
    assert result is ace.return_value
    assert capsys.readouterr().out.splitlines() == (
        [f"[ACE] {note}" for note in notes]
        + [f"[ACE] Loading potential: {tmp_path / 'pot.yace'}"])


@pytest.mark.parametrize("device, dtype, notes", NOTE_CASES)
def test_lammps_builds_calculator_and_notes_unused_settings(capsys, device, dtype, notes):
    calc = get_calculator("lammps", device=device, default_dtype=dtype,
                          lammps_params=LJ_CU_AG)
    assert isinstance(calc, LAMMPSCalculator)
    assert calc.parameters.lmpcmds == ["pair_style lj/cut 6.0", "pair_coeff 1 1 0.40 2.30",
                                       "pair_coeff 2 2 0.35 2.60", "pair_coeff 1 2 0.37 2.45"]
    assert calc.parameters.atom_types == {"Cu": 1, "Ag": 2}
    assert calc.potential_files == []
    assert capsys.readouterr().out.splitlines() == (
        [f"[LAMMPS] {note}" for note in notes]
        + ["[LAMMPS] pair_style lj/cut 6.0, types 1=Cu 2=Ag"])


# ─── Detection helpers ────────────────────────────────────────────────────

@pytest.mark.parametrize("model", ["ace", "lammps"])
def test_float64_backends_pass_the_dtype_check(model):
    calc_mod.require_dtype(model, "float64")


def test_unknown_backend_name_is_unavailable():
    assert calc_mod.backend_available("not-a-backend") is False
