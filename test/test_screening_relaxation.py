"""Convergence evidence survives writing and uses the actual stopping criterion."""

import json

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read, write

from amorphgen.pipeline.opt_cell import run
from amorphgen.utils.relaxation import read_relaxation_metadata, write_relaxation_metadata


@pytest.mark.parametrize("fmax, expected", [(10.0, True), (1e-12, False)])
def test_ase_relaxation_metadata_survives_extxyz_and_cif(tmp_path, fmax, expected):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.positions[0, 0] += 0.1
    result = run(atoms, calc=EMT(), work_dir=str(tmp_path), cfg_override={
        "opt": {"cell_filter": "none", "fmax": fmax, "max_steps": 1},
    })
    assert result.info["relaxation_converged"] is expected
    assert result.info["relaxation_steps"] == 1
    assert result.info["relaxation_max_force"] == pytest.approx(
        np.linalg.norm(result.get_forces(), axis=1).max())
    saved = read(tmp_path / "stage1_opt.xyz")
    assert bool(saved.info["relaxation_converged"]) is expected
    # Collection/conversion reads extxyz values back as NumPy scalars.
    converted = tmp_path / "collected.vasp"
    write(converted, saved, format="vasp")
    write_relaxation_metadata(converted, saved)
    assert read_relaxation_metadata(converted)["relaxation_converged"] is expected
    cif_path = tmp_path / "stage1_opt.cif"
    cif_atoms = read(cif_path)
    metadata = read_relaxation_metadata(cif_path, cif_atoms)
    assert metadata["relaxation_converged"] is expected
    assert cif_atoms.info["relaxation_fmax"] == fmax


def test_zero_step_cell_relaxation_checks_cell_forces(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    result = run(atoms, calc=EMT(), work_dir=str(tmp_path), cfg_override={
        "opt": {"cell_filter": "cubic", "fmax": 1e-8, "max_steps": 0},
    })
    assert np.linalg.norm(result.get_forces(), axis=1).max() < 1e-8
    assert result.info["relaxation_max_force"] > 1e-8
    assert result.info["relaxation_converged"] is False
    assert result.info["relaxation_steps"] == 0


@pytest.mark.parametrize("ensemble", ["NVT", "NPT"])
def test_ase_md_discards_previous_geometry_convergence(tmp_path, ensemble):
    from amorphgen.utils.common import build_md_dynamics

    atoms = run(bulk("Cu", "fcc", a=3.6, cubic=True), calc=EMT(),
                work_dir=str(tmp_path), cfg_override={
                    "opt": {"cell_filter": "none", "fmax": 0.1, "max_steps": 1},
                })
    assert atoms.info["relaxation_converged"] is True
    atoms.info["relax_converged"] = True
    atoms.info["source"] = "crystal"
    build_md_dynamics(atoms, ensemble=ensemble)
    assert not any(key.startswith("relaxation_") for key in atoms.info)
    assert "relax_converged" not in atoms.info
    assert atoms.info["source"] == "crystal"


def test_stale_and_invalid_sidecars_do_not_label_replacement_files(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    run(atoms, calc=EMT(), work_dir=str(tmp_path), cfg_override={
        "opt": {"cell_filter": "none", "fmax": 0.1, "max_steps": 1},
    })
    path = tmp_path / "stage1_opt.cif"
    assert read_relaxation_metadata(path)["relaxation_converged"] is True
    write(path, bulk("Cu", "fcc", a=4.0, cubic=True))
    replacement = read(path)
    assert read_relaxation_metadata(path, replacement) == {}
    assert "relaxation_converged" not in replacement.info
    sidecar = tmp_path / "stage1_opt.cif.relaxation.json"
    sidecar.write_text("{broken", encoding="utf-8")
    assert read_relaxation_metadata(path) == {}
    sidecar.write_text(json.dumps(["invalid-schema"]), encoding="utf-8")
    assert read_relaxation_metadata(path) == {}
    sidecar.unlink()
    assert read_relaxation_metadata(path) == {}


@pytest.mark.parametrize("output_format, extension", [("extxyz", "xyz"), ("vasp", "vasp")])
def test_random_relaxation_persists_result_and_resume_retains_evidence(
        tmp_path, monkeypatch, output_format, extension):
    from amorphgen.pipeline import random_gen

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.positions[0, 0] += 0.1
    monkeypatch.setattr(random_gen, "generate_random", lambda *args, **kwargs: atoms.copy())
    kwargs = dict(composition={"Cu": 4}, n_structures=1, output_dir=str(tmp_path),
                  output_format=output_format, relax=True, calc=EMT(), fmax=1e-12,
                  max_relax_steps=1, cell_filter="none", cell_length_ang=3.6)
    paths = random_gen.batch_random(**kwargs)
    assert len(paths) == 1
    assert paths[0].endswith("." + extension)
    metadata = read_relaxation_metadata(paths[0])
    assert metadata["relaxation_converged"] is False
    assert metadata["relaxation_steps"] == 1
    initial = read(tmp_path / "random_initial" / ("random_0000." + extension))
    assert "relaxation_converged" not in initial.info
    monkeypatch.setattr(random_gen, "generate_random",
                        lambda *args, **kwargs: pytest.fail("resume regenerated a completed file"))
    assert random_gen.batch_random(**kwargs, resume=True) == paths
    assert read_relaxation_metadata(paths[0]) == metadata


@pytest.mark.parametrize("cell_filter, expected", [("none", True), ("cubic", False)])
def test_torchsim_convergence_includes_pressure_and_records_steps(cell_filter, expected):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_sim")
    from torch_sim.models.interface import ModelInterface
    from amorphgen.utils.torchsim_engine import batch_relax

    class ConstantStressModel(ModelInterface):
        def __init__(self):
            super().__init__()
            self._device = torch.device("cpu")
            self._dtype = torch.float64
            self._compute_forces = self._compute_stress = True

        def forward(self, state, **kwargs):
            return {
                "energy": torch.zeros(state.n_systems, dtype=self.dtype),
                "forces": torch.zeros_like(state.positions),
                "stress": torch.eye(3, dtype=self.dtype).expand(state.n_systems, -1, -1) * 0.001,
            }

    out = batch_relax([bulk("Cu", "fcc", a=3.6, cubic=True)], ConstantStressModel(),
                      max_steps=1, cell_filter=cell_filter, pressure_tol_gpa=0.02,
                      log=lambda *args: None)
    info = out[0].info
    assert info["relaxation_converged"] is expected
    assert info["relaxation_max_force"] == 0.0
    assert info["relaxation_engine"] == "torchsim"
    assert info["relaxation_steps"] == 1
    if cell_filter == "cubic":
        assert abs(info["relaxation_pressure_gpa"]) > info["relaxation_pressure_tol_gpa"]

    from amorphgen.utils.torchsim_md import batch_nvt
    frames = batch_nvt(out, ConstantStressModel(), 300, n_steps=1,
                       log=lambda *args: None)
    assert "relaxation_converged" in out[0].info  # input remains the relaxed geometry
    assert not any(key.startswith("relaxation_") for key in frames[0].info)
