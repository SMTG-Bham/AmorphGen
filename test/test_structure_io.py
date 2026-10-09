"""Structure format and discovery contracts shared across entry points."""

from amorphgen.utils.structure_io import first_structure_files
from amorphgen.utils.convert import _gather_inputs
from amorphgen.utils.calculators import calculator_kwargs


def test_discovery_keeps_ensemble_priority_distinct_from_conversion(tmp_path):
    for name in ("z.xyz", "a.xyz", "a.extxyz", "b.cif", "POSCAR"):
        (tmp_path / name).touch()
    expected = [str(tmp_path / name) for name in ("a.xyz", "z.xyz")]
    assert first_structure_files(tmp_path) == expected
    assert len(_gather_inputs(str(tmp_path))) == 5
    for path in tmp_path.glob("*.xyz"):
        path.unlink()
    assert first_structure_files(tmp_path) == [str(tmp_path / "a.extxyz")]
    assert first_structure_files(tmp_path / "absent") == []


def test_calculator_arguments_preserve_explicit_none_and_backend_parameters():
    params = {"epsilon": 0.1, "sigma": 2.3}
    cfg = {"model": "lj", "device": None, "classical_params": params,
           "ace_params": {}, "opt": {"fmax": .01}}
    assert calculator_kwargs(cfg, defaults={"device": "cpu", "default_dtype": "float64"}) == {
        "model": "lj", "device": None, "model_path": None,
        "default_dtype": "float64", "classical_params": params,
    }
    assert cfg["device"] is None
