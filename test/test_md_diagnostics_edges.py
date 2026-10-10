"""Ramp-schedule alignment in stage diagnostics and argument checks of batched torch-sim NVT."""

import json

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from amorphgen.utils.common import format_md_log_row, open_md_log
from amorphgen.utils.md_diagnostics import write_stage_diagnostics


def write_stage_outputs(directory, steps):
    """Paired log and extxyz trajectory, one static Cu2 sample per logged step."""
    frames = []
    with open_md_log(directory / "stage5_quench.log") as log:
        for step in steps:
            atoms = Atoms("Cu2", positions=[[0, 0, 0], [1.8, 1.8, 0]], cell=[6, 6, 6], pbc=True)
            atoms.calc = SinglePointCalculator(atoms, energy=-7.0)
            frames.append(atoms)
            log.write(format_md_log_row(step, step * 5e-4, 300.0, -7.0, 0.1,
                                        atoms.get_volume()) + "\n")
    write(directory / "stage5_quench_traj.xyz", frames, format="extxyz")
    return directory / "stage5_quench.log", directory / "stage5_quench_traj.xyz"


def test_boundary_samples_belong_to_the_hold_just_finished(tmp_path):
    log, traj = write_stage_outputs(tmp_path, [0, 100, 200, 300])
    report = write_stage_diagnostics(log, traj, timestep_fs=0.5, stage=5,
                                     temperatures=[1000, 900, 800], steps_per_T=100)
    windows = report["temperature_windows"]
    assert [window["temperature_K"] for window in windows] == [1000, 900, 800]
    assert [window["n_frames"] for window in windows] == [2, 1, 1]
    assert report["sample_steps"] == [0, 100, 200, 300]


@pytest.mark.parametrize("temperatures, steps_per_T, steps, message", [
    ([], 100, [0, 100], "A ramp needs temperatures and positive steps_per_T"),
    ([[1000, 900]], 100, [0, 100], "A ramp needs temperatures and positive steps_per_T"),
    ([1000, 900], None, [0, 100], "A ramp needs temperatures and positive steps_per_T"),
    ([1000, 900], 0, [0, 100], "A ramp needs temperatures and positive steps_per_T"),
    ([1000, 900, 800], 100, [0, 100, 200, 300, 400],
     "Logged steps extend beyond the resolved temperature ramp"),
])
def test_unresolvable_ramp_is_saved_as_unavailable(tmp_path, temperatures, steps_per_T,
                                                   steps, message):
    log, traj = write_stage_outputs(tmp_path, steps)
    report = write_stage_diagnostics(log, traj, timestep_fs=0.5, stage=5,
                                     temperatures=temperatures, steps_per_T=steps_per_T)
    saved = json.loads((tmp_path / "stage5_quench_diagnostics.json").read_text())
    assert saved == report
    assert saved["status"] == "unavailable"
    assert saved["warnings"] == [f"ValueError: {message}."]
    assert saved["temperature_windows"] == [] and saved["freezing_temperature_K"] is None
    assert (tmp_path / "stage5_quench_diagnostics.txt").read_text().startswith(
        "MD CONVERGENCE DIAGNOSTICS\nUnavailable: ValueError: " + message)


def zero_force_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_sim")
    from torch_sim.models.interface import ModelInterface

    class ZeroModel(ModelInterface):
        def __init__(self):
            super().__init__()
            self._device, self._dtype = torch.device("cpu"), torch.float64
            self._compute_forces = self._compute_stress = True
            self.calls = 0

        def forward(self, state, **kwargs):
            self.calls += 1
            return {"energy": torch.zeros(state.n_systems, dtype=self.dtype),
                    "forces": torch.zeros_like(state.positions),
                    "stress": torch.zeros_like(state.cell)}

    return ZeroModel()


def cu_pair():
    return Atoms("Cu2", positions=[[2, 2, 2], [3.5, 2, 2]], cell=[8, 8, 8], pbc=True)


def test_batch_nvt_of_no_structures_is_empty():
    from amorphgen.utils.torchsim_md import batch_nvt

    model = zero_force_model()
    assert batch_nvt([], model, 300, n_steps=10) == []
    assert model.calls == 0


@pytest.mark.parametrize("kwargs, message", [
    ({"n_steps": 0}, "n_steps must be a positive integer"),
    ({"n_steps": 2.5}, "n_steps must be a positive integer"),
    ({"interval": 0}, "interval must be a positive integer"),
    ({"interval": 1.5}, "interval must be a positive integer"),
    ({"writers": []}, "one writer per structure"),
    ({"temperatures": [300.0, 300.0]}, "temperature schedule must contain n_steps"),
    ({"temperatures": [300.0, np.nan, 300.0]}, "temperature schedule must contain n_steps"),
    ({"temperatures": [300.0, -1.0, 300.0]}, "temperature schedule must contain n_steps"),
])
def test_batch_nvt_rejects_invalid_arguments_before_any_force_call(kwargs, message):
    from amorphgen.utils.torchsim_md import batch_nvt

    model = zero_force_model()
    atoms = cu_pair()
    arguments = {"temperatures": 300.0, "n_steps": 3, **kwargs}
    with pytest.raises(ValueError, match=message):
        batch_nvt([atoms], model, log=lambda *a: None, **arguments)
    assert model.calls == 0
    np.testing.assert_array_equal(atoms.positions, cu_pair().positions)
