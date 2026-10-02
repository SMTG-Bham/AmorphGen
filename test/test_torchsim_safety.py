"""Per-step torch-sim safeguards with small, deterministic CPU models."""
from unittest.mock import Mock

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


torch = pytest.importorskip("torch")
ts = pytest.importorskip("torch_sim")
from torch_sim.models.interface import ModelInterface

from amorphgen.utils.common import DivergenceError
from amorphgen.utils.torchsim_engine import _TorchSafetyBridge, batch_relax
from amorphgen.utils.torchsim_md import batch_nvt


def atoms_pair(distance=1.5, cell=8.0):
    return Atoms("Cu2", positions=[[2, 2, 2], [2 + distance, 2, 2]],
                 cell=[cell] * 3, pbc=True)


class TestModel(ModelInterface):
    __test__ = False

    def __init__(self, failure=None, fail_call=1):
        super().__init__()
        self._device = torch.device("cpu")
        self._dtype = torch.float64
        self._compute_forces = self._compute_stress = True
        self.calls = 0
        self.failure, self.fail_call = failure, fail_call

    def forward(self, state, **kwargs):
        self.calls += 1
        result = {
            "energy": torch.zeros(state.n_systems, dtype=self.dtype),
            "free_energy": torch.zeros(state.n_systems, dtype=self.dtype),
            "forces": torch.zeros_like(state.positions),
            "stress": torch.zeros_like(state.cell),
        }
        if self.calls >= self.fail_call and self.failure:
            if self.failure == "jump":
                result["energy"] += 100.0
            elif self.failure == "kick":
                result["forces"][:, 0] = 1e8
            else:
                result[self.failure][(-1,) * result[self.failure].ndim] = float("nan")
        return result


@pytest.mark.parametrize("property", ["energy", "free_energy", "forces", "stress"])
@pytest.mark.parametrize("runner", ["relax", "md"])
def test_nonfinite_model_outputs_are_rejected(property, runner):
    model = TestModel(failure=property)
    with pytest.raises(DivergenceError, match=r"Non-finite.*step 0.*system 1"):
        if runner == "relax":
            batch_relax([atoms_pair(), atoms_pair()], model, cell_filter="none", max_steps=5)
        else:
            batch_nvt([atoms_pair(), atoms_pair()], model, 300, n_steps=5)


@pytest.mark.parametrize("failure, message", [("forces", "Non-finite.*forces"),
                                                ("jump", "Energy jump"),
                                                ("kick", "Temperature runaway")])
def test_md_checks_each_step_before_trajectory_interval(failure, message):
    # First two calls initialize MD and its first output block. The third is
    # the first integration step, well before a trajectory would be written.
    model = TestModel(failure=failure, fail_call=3)
    writer = Mock()
    with pytest.raises(DivergenceError, match=message + r".*step 1.*system 0"):
        batch_nvt([atoms_pair()], model, 300, n_steps=20, interval=20,
                  writers=[writer], seed=12, log=lambda *a: None)
    writer.write.assert_not_called()
    assert model.calls == 3


@pytest.mark.parametrize("runner", ["relax", "md"])
@pytest.mark.parametrize("failure, message", [("close", "Close contact"),
                                               ("position", "Non-finite positions"),
                                               ("momentum", "Non-finite momenta"),
                                               ("temperature", "Temperature runaway")])
def test_bad_initial_geometry_precedes_model_call(runner, failure, message):
    atoms = atoms_pair(0.1 if failure == "close" else 1.5)
    if failure == "position":
        atoms.positions[0, 0] = np.inf
    if failure == "momentum":
        atoms.set_momenta([[np.nan, 0, 0], [0, 0, 0]])
    if failure == "temperature":
        atoms.set_momenta(np.full((2, 3), 1e5))
    model = TestModel()
    with pytest.raises(DivergenceError, match=message):
        if runner == "relax":
            batch_relax([atoms], model, max_steps=5)
        else:
            batch_nvt([atoms], model, 300, n_steps=5)
    assert model.calls == 0


def test_energy_histories_follow_system_ids_through_shrinking_and_reordering():
    inputs = [atoms_pair(), atoms_pair(cell=10)]
    bridge = _TorchSafetyBridge(inputs)
    model = TestModel()
    state = bridge.attach(ts.initialize_state(inputs, model.device, model.dtype))

    def results(state, values):
        return {"energy": torch.tensor(values, dtype=model.dtype),
                "forces": torch.zeros_like(state.positions)}

    bridge.check(state, results(state, [0, 1000]))
    reordered = state[[1, 0]]
    bridge.check(reordered, results(reordered, [1000, 0]))
    reduced = reordered[[0]]
    bridge.advance(reduced)
    bridge.check(reduced, results(reduced, [1001]))
    assert bridge.steps == [0, 1]
    with pytest.raises(DivergenceError, match=r"Energy jump.*step 1.*system 1"):
        bridge.check(reduced, results(reduced, [1030]))


def test_volume_baselines_follow_system_ids_in_smaller_batches():
    inputs = [atoms_pair(), atoms_pair(cell=20)]
    bridge = _TorchSafetyBridge(inputs)
    model = TestModel()
    state = bridge.attach(ts.initialize_state(inputs, model.device, model.dtype))
    reduced = state[[1]]
    bridge.check(reduced, geometry_only=True)
    reduced.cell *= 2
    with pytest.raises(DivergenceError, match=r"Volume runaway.*system 1"):
        bridge.check(reduced, geometry_only=True)


def test_missing_identity_support_is_explicit():
    bridge = _TorchSafetyBridge([atoms_pair()])
    with pytest.raises(RuntimeError, match="requires SimState.system_extras"):
        bridge.attach(object())


class ZeroReference(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}


def test_reference_spot_checks_exclude_added_core(monkeypatch):
    from amorphgen.utils import calculators
    monkeypatch.setattr(calculators, "get_calculator", lambda **kwargs: ZeroReference())
    out = batch_nvt([atoms_pair(0.8)], TestModel(), 0.0, n_steps=1,
                    timestep_fs=0.001, log=lambda *a: None,
                    repulsive_core={"enabled": True, "cutoff": 1.0, "strength": 1.0},
                    safety={"reference": {"model": "reference", "interval": 1,
                              "max_force_rmse": 1e-10,
                              "max_energy_difference_per_atom": 1e-10}})
    assert out[0].get_potential_energy() > 0


def test_reference_disagreement_stops_torch_md(monkeypatch):
    from amorphgen.utils import calculators
    monkeypatch.setattr(calculators, "get_calculator", lambda **kwargs: ZeroReference())
    with pytest.raises(DivergenceError, match="Reference model disagreement"):
        batch_nvt([atoms_pair()], TestModel(failure="jump"), 0.0, n_steps=1,
                  safety={"reference": {"model": "reference",
                                         "max_energy_difference_per_atom": 1.0}})


def test_reference_model_is_loaded_once_for_the_batch(monkeypatch):
    from amorphgen.utils import calculators
    factory = Mock(side_effect=lambda **kwargs: ZeroReference())
    monkeypatch.setattr(calculators, "get_calculator", factory)
    batch_nvt([atoms_pair(), atoms_pair(cell=9)], TestModel(), 0.0, n_steps=2,
              log=lambda *a: None,
              safety={"reference": {"model": "reference", "interval": 1}})
    assert factory.call_count == 1


def test_optimizer_checks_each_step_before_convergence_swap():
    model = TestModel(failure="forces", fail_call=2)
    with pytest.raises(DivergenceError, match=r"Non-finite forces.*step 1.*system 0"):
        batch_relax([atoms_pair()], model, cell_filter="none", max_steps=20,
                    log=lambda *a: None)
    assert model.calls == 2


def test_memory_probe_rejects_invalid_geometry_even_with_cpu_fallback():
    from amorphgen.utils.torchsim_engine import estimate_batch_size
    model = TestModel()
    with pytest.raises(DivergenceError, match="Close contact.*memory probe"):
        estimate_batch_size(model, [atoms_pair(0.1)])
    assert model.calls == 0


@pytest.mark.parametrize("end_exists", [False, True])
@pytest.mark.parametrize("failure", ["energy", "jump"])
def test_resumed_checkpoint_is_checked_before_new_files(tmp_path, monkeypatch, end_exists, failure):
    from ase.io import write
    from amorphgen.pipeline.batch_quench import run_torchsim
    from amorphgen.utils import calculators, torchsim_engine
    source = tmp_path / "s0.xyz"
    write(source, atoms_pair(), format="extxyz")
    work = tmp_path / "work"
    run = work / "run_0000"
    run.mkdir(parents=True)
    end = run / "stage4_eq.xyz"
    write(end if end_exists else run / "stage4_eq_traj.xyz", atoms_pair(), format="extxyz")
    monkeypatch.setattr(torchsim_engine, "build_model", lambda *a, **k: TestModel(failure=failure))
    monkeypatch.setattr(calculators, "get_calculator", lambda **kwargs: ZeroReference())
    config = {"eq_high": {"ensemble": "NVT", "T": 300, "steps": 100}}
    if failure == "jump":
        config["safety"] = {"reference": {"model": "reference",
                            "max_energy_difference_per_atom": 1.0}}
    expected = "Reference model disagreement" if failure == "jump" else "Non-finite"
    with pytest.raises(DivergenceError, match=expected + ".*resumed checkpoint"):
        run_torchsim([str(source)], cfg_override=config, work_dir=str(work),
                     stages=[4], resume=True, batch_size=1)
    assert end.exists() == end_exists
    assert not (run / "final_amorphous.xyz").exists()
