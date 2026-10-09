"""Boundary contracts shared by ASE ramps and batched torch-sim output."""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

from amorphgen.pipeline import melt_cell, quench
from amorphgen.utils.common import (
    MDLogger, MD_LOG_HEADER, compute_density_gcm3,
    resolve_ramp_schedule, select_frame_indices,
)
from amorphgen.utils.equilibration import parse_md_log
from amorphgen.utils.torchsim_md import _RunWriter


@pytest.mark.parametrize("step, rate", [(50, 200), (-50, 200), (50, -200)])
def test_schedule_preserves_partial_endpoint_and_rate_magnitude(step, rate):
    temperatures, steps = resolve_ramp_schedule(
        300, 475, step, timestep_fs=.5, rate=rate, steps_per_T=99)
    assert temperatures == [350, 400, 450, 475]
    assert steps == 500
    assert resolve_ramp_schedule(475, 300, step, timestep_fs=.5, rate=rate)[0] == [425, 375, 325, 300]


@pytest.mark.parametrize("module, section, stage", [(melt_cell, "melt", 3), (quench, "quench", 5)])
@pytest.mark.parametrize("invalid", [{"T_step": 0}, {"rate": 0}, {"steps_per_T": 0}])
def test_invalid_schedule_preserves_existing_outputs(tmp_path, module, section, stage, invalid):
    paths = [tmp_path / f"stage{stage}_{section}{suffix}" for suffix in (".log", "_traj.xyz")]
    for path in paths:
        path.write_bytes(b"earlier run output\n")
    cfg = {"seed": 42, section: {"ensemble": "NVT", **invalid}}
    with pytest.raises((ValueError, ZeroDivisionError)):
        module.run(bulk("Cu", cubic=True), cfg, EMT(), work_dir=tmp_path)
    assert all(path.read_bytes() == b"earlier run output\n" for path in paths)


@pytest.mark.parametrize("has_stress", [False, True])
def test_ase_and_torch_writers_emit_the_same_table_with_resume_offset(tmp_path, has_stress):
    atoms = bulk("Cu", cubic=True)
    atoms.set_momenta(np.full((len(atoms), 3), .2))
    stress = np.diag([-2., -3., -4.]) * units.GPa
    properties = {"stress": stress} if has_stress else {}
    atoms.calc = SinglePointCalculator(atoms, energy=-1.25,
                                      forces=np.zeros((len(atoms), 3)), **properties)
    dyn = SimpleNamespace(nsteps=200, dt=.5 * units.fs)
    dyn.get_time = lambda: dyn.nsteps * dyn.dt
    ase_path = tmp_path / "ase.log"
    logger = MDLogger(ase_path, step_offset=100)
    writer = _RunWriter(tmp_path, "torch.log", "torch.xyz", step_offset=100)
    try:
        logger.log(dyn, atoms)
        writer.write(atoms, dyn.nsteps, .5)
    finally:
        logger.close()
    assert ase_path.read_bytes() == (tmp_path / "torch.log").read_bytes()
    table = parse_md_log(ase_path)
    np.testing.assert_array_equal(table["step"], [300])
    np.testing.assert_allclose(table["time_ps"], [.15])
    np.testing.assert_allclose(table["Etot_eV"], table["Epot_eV"] + table["Ekin_eV"], atol=1e-4)
    assert ase_path.read_text().splitlines()[0].split()[-2:] == ["P_GPa", "density_g_cm3"]
    row = np.loadtxt(ase_path, skiprows=2)
    assert row[8] == pytest.approx(compute_density_gcm3(atoms), abs=5e-7)
    np.testing.assert_allclose(table["P_GPa"], [row[7]], equal_nan=True)
    np.testing.assert_allclose(table["density_g_cm3"], [row[8]])
    if has_stress:
        kinetic_pressure = 2 * atoms.get_kinetic_energy() / (3 * atoms.get_volume() * units.GPa)
        assert row[7] == pytest.approx(3 + kinetic_pressure, abs=5e-7)
        # Torch trajectories retain the stress used by pressure diagnostics.
        np.testing.assert_allclose(read(tmp_path / "torch.xyz").get_stress(voigt=False), stress)
    else:
        assert np.isnan(row[7])


@pytest.mark.parametrize("engine", ["ase", "torchsim"])
def test_resuming_a_legacy_log_labels_the_new_columns(tmp_path, engine):
    legacy = "Step Time_ps T_K Epot_eV Ekin_eV Etot_eV Vol_A3\n0 0 300 -1 1 0 100\n"
    path = tmp_path / "md.log"
    path.write_text(legacy)
    atoms = bulk("Cu", cubic=True)
    atoms.calc = SinglePointCalculator(atoms, energy=-1, forces=np.zeros((len(atoms), 3)))
    if engine == "ase":
        dyn = SimpleNamespace(nsteps=100, dt=units.fs, get_time=lambda: 100 * units.fs)
        logger = MDLogger(path, mode="a")
        try:
            logger.log(dyn, atoms)
        finally:
            logger.close()
    else:
        _RunWriter(tmp_path, "md.log", "md.xyz", append=True).write(atoms, 100, 1)
    text = path.read_text()
    assert text.startswith(legacy)
    assert MD_LOG_HEADER in text
    assert len(text.splitlines()[-1].split()) == 9
    table = parse_md_log(path)
    np.testing.assert_array_equal(table["step"], [0, 100])
    assert np.isnan(table["P_GPa"]).all()
    assert np.isnan(table["density_g_cm3"][0])
    assert table["density_g_cm3"][1] == pytest.approx(compute_density_gcm3(atoms), abs=5e-7)


def test_simple_selection_caps_counts_and_preserves_singleton_semantics():
    assert select_frame_indices(10, 4, "uniform", 2) == [2, 4, 6, 9]
    assert select_frame_indices(10, 4, "last", 2) == [6, 7, 8, 9]
    assert select_frame_indices(10, 99, "uniform", 8) == [8, 9]
    assert select_frame_indices(10, 1, "uniform", 2) == [2]
    assert select_frame_indices(10, 1, "last", 2) == [9]
