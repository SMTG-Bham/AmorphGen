"""Regressions for preserving crystal geometry and effective stage settings."""

from copy import deepcopy
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.io import read, write
from ase.spacegroup import crystal

from amorphgen.cli import _build_override, _get_parser, main
from amorphgen.pipeline import final_opt, melt_cell
from amorphgen.pipeline.run_pipeline import MeltQuenchPipeline
from amorphgen.utils import merge_config
from amorphgen.utils.calculators import get_calculator


@pytest.mark.parametrize("shape", ["quartz", "orthorhombic"])
def test_melt_preserves_input_geometry_before_first_step(tmp_path, shape):
    if shape == "quartz":
        atoms = crystal(
            ["Si", "O"], basis=[(0.4697, 0, 0), (0.4133, 0.2672, 0.1188)],
            spacegroup=152, cellpar=[4.913, 4.913, 5.405, 90, 90, 120],
        )
    else:
        atoms = bulk("Cu", cubic=True).repeat((1, 2, 3))
    cell_before = atoms.cell.array.copy()
    positions_before = atoms.positions.copy()
    distances_before = atoms.get_all_distances(mic=True)

    result = melt_cell.run(
        atoms, calc=LennardJones(), work_dir=tmp_path,
        cfg_override={"seed": 11, "melt": {
            "ensemble": "NVT", "T_start": 300, "T_end": 300,
            "steps_per_T": 1, "timestep": 0.001,
        }},
    )

    # The first trajectory frame records the structure before any MD
    # displacement; a cubic reshape would already have shortened bonds.
    first = read(tmp_path / "stage3_melt_traj.xyz", index=0)
    np.testing.assert_allclose(first.cell.array, cell_before, atol=1e-8)
    np.testing.assert_allclose(first.positions, positions_before, atol=1e-8)
    np.testing.assert_allclose(first.get_all_distances(mic=True), distances_before,
                               atol=1e-8)
    np.testing.assert_allclose(result.cell.array, cell_before, atol=1e-8)
    np.testing.assert_array_equal(atoms.cell.array, cell_before)
    np.testing.assert_array_equal(atoms.positions, positions_before)


def test_melt_explicit_cubic_reshape_is_still_supported(tmp_path):
    atoms = bulk("Cu", cubic=True).repeat((1, 2, 3))
    result = melt_cell.run(
        atoms, calc=EMT(), work_dir=tmp_path,
        cfg_override={"seed": 11, "melt": {
            "ensemble": "NVT", "make_cubic": True,
            "T_start": 300, "T_end": 300,
            "steps_per_T": 1, "timestep": 0.001,
        }},
    )
    np.testing.assert_allclose(result.cell.array,
                               np.eye(3) * atoms.get_volume() ** (1 / 3))


def test_cli_format_does_not_reset_yaml_final_optimisation(tmp_path, monkeypatch):
    input_path = tmp_path / "input.xyz"
    write(input_path, bulk("Cu", cubic=True))
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        "model: lj\ndevice: cpu\n"
        "opt:\n  optimizer: FIRE\n  fmax: 0.123\n"
        "  max_steps: 2\n  cell_filter: none\n"
    )
    work_dir = tmp_path / "run"
    monkeypatch.setattr("amorphgen.pipeline.run_pipeline.get_calculator",
                        lambda **kwargs: EMT())
    monkeypatch.setattr(sys, "argv", [
        "amorphgen", str(input_path), "--config", str(config_path),
        "--stages", "1", "7", "--format", "vasp", "-o", str(work_dir),
    ])

    main()

    for stage in (1, 7):
        log = (work_dir / f"stage{stage}_opt.log").read_text()
        assert "Optimizer: FIRE  fmax=0.123  max_steps=2" in log
        assert "Cell filter: none" in log
        assert (work_dir / f"stage{stage}_opt.vasp").is_file()


def test_final_opt_specific_values_override_inherited_settings(tmp_path):
    cfg = {"opt": {
        "optimizer": "FIRE", "fmax": 0.123, "max_steps": 2,
        "cell_filter": "none",
    }, "final_opt": {"fmax": 0.234, "max_steps": 3}}
    original = deepcopy(cfg)
    final_opt.run(bulk("Cu", cubic=True), cfg_override=cfg, calc=EMT(),
                  work_dir=tmp_path)
    log = (tmp_path / "stage7_opt.log").read_text()
    assert "Optimizer: FIRE  fmax=0.234  max_steps=3" in log
    assert "Cell filter: none" in log
    assert cfg == original


@pytest.mark.parametrize("dtype", ["float32", "float64", "auto"])
@pytest.mark.parametrize("share_calc", [True, False])
def test_pipeline_forwards_cli_dtype_to_calculator(tmp_path, monkeypatch,
                                                 dtype, share_calc):
    parser = _get_parser()
    argv = ["input.xyz", "--dtype", dtype, "--device", "cpu"]
    cfg = merge_config({"default_dtype": "float64"}, _build_override(
        parser.parse_args(argv), parser, explicit_only=True, argv=argv,
    ))
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return EMT()

    monkeypatch.setattr("amorphgen.pipeline.run_pipeline.get_calculator", factory)
    pipe = MeltQuenchPipeline("input.xyz", work_dir=str(tmp_path),
                             cfg_override=cfg, share_calc=share_calc)
    pipe._get_calc()
    pipe._get_calc()
    assert len(calls) == (1 if share_calc else 2)
    assert all(call["default_dtype"] == dtype for call in calls)


@pytest.mark.parametrize("cuda, mps, expected", [
    (True, True, "cuda"),
    (False, True, "mps"),
    (False, False, "cpu"),
])
def test_pipeline_and_direct_calculator_resolve_auto_consistently(
        tmp_path, monkeypatch, cuda, mps, expected):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    ))
    devices = []

    def loader(device, **kwargs):
        devices.append(device)
        return EMT()

    monkeypatch.setattr("amorphgen.utils.calculators._load_chgnet", loader)
    get_calculator("chgnet", device="auto")
    pipe = MeltQuenchPipeline(
        "input.xyz", work_dir=str(tmp_path),
        cfg_override={"model": "chgnet", "device": "auto"},
    )
    pipe._get_calc()
    assert devices == [expected, expected]


@pytest.mark.parametrize("stage", ["opt_cell", "melt_cell", "equilibrate",
                                  "quench", "batch_quench"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "auto"])
def test_standalone_stages_forward_dtype(tmp_path, monkeypatch, stage, dtype):
    module = importlib.import_module(f"amorphgen.pipeline.{stage}")
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return EMT()

    monkeypatch.setattr(module, "get_calculator", factory)
    cfg = {
        "device": "cpu", "default_dtype": dtype, "seed": 11,
        "opt": {"cell_filter": "none", "max_steps": 1},
        "melt": {"ensemble": "NVT", "T_start": 300, "T_end": 300,
                 "steps_per_T": 1},
        "quench": {"ensemble": "NVT", "T_start": 300, "T_end": 300,
                   "steps_per_T": 1},
        "eq_high": {"ensemble": "NVT", "T": 300, "steps": 1},
    }
    atoms = bulk("Cu", cubic=True)
    if stage == "batch_quench":
        input_path = tmp_path / "snapshot_0000.xyz"
        write(input_path, atoms)
        module.run([str(input_path)], cfg_override=cfg, stages=[7],
                   work_dir=str(tmp_path / "batch"))
    else:
        module.run(atoms, cfg_override=cfg, work_dir=tmp_path)
    assert len(calls) == 1
    assert calls[0]["default_dtype"] == dtype
