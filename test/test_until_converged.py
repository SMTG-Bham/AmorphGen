"""Native independent batches, sequential stopping, and durable continuation."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


@pytest.fixture
def driver():
    from amorphgen.pipeline import until_converged
    return until_converged


@pytest.fixture
def simulated(driver, monkeypatch):
    """Exercise the real controller and descriptor extraction without TorchSim."""
    hooks = SimpleNamespace(
        generated=[], relaxed=[], built=[], fail_batch=None, fail_generation=None,
        force=np.zeros(3), stress=np.zeros((3, 3)), drop_output=False,
        reverse_outputs=False, mutate_generation=False,
    )

    def generate(composition, seed=None, **kwargs):
        hooks.generated.append({"seed": seed, "generation": deepcopy(kwargs)})
        if hooks.fail_generation == len(hooks.generated):
            raise RuntimeError("placement failure")
        if hooks.mutate_generation:
            kwargs["minsep"]["Cu-Cu"] = 999.0
        shift = (int(seed) % 19) / 1000.
        return Atoms("Cu2", positions=[[1. + shift, 1., 1.], [3., 3., 3.]],
                     cell=[6.] * 3, pbc=True)

    def build(*args, **kwargs):
        hooks.built.append((args, kwargs))
        model = SimpleNamespace(device="cpu", dtype="float64", training=True)
        def evaluate():
            model.training = False
            return model
        model.eval = evaluate
        return model

    def relax(frames, model, **kwargs):
        assert model.training is False
        hooks.relaxed.append({"seeds": [atoms.info["sequential_seed"] for atoms in frames],
                              "indices": [atoms.info["sequential_index"] for atoms in frames],
                              "kwargs": deepcopy(kwargs)})
        if hooks.fail_batch == len(hooks.relaxed):
            raise RuntimeError("simulated TorchSim batch failure")
        output = []
        for original in frames:
            atoms = original.copy()
            atoms.calc = SinglePointCalculator(
                atoms, energy=-1., forces=np.tile(hooks.force, (len(atoms), 1)),
                stress=hooks.stress.copy(),
            )
            # A component maximum can conceal a failed vector-norm tolerance.
            atoms.info["max_force"] = float(np.abs(hooks.force).max())
            output.append(atoms)
        if hooks.reverse_outputs:
            output.reverse()
        return output[:-1] if hooks.drop_output else output

    monkeypatch.setattr(driver, "generate_random", generate)
    monkeypatch.setattr(driver, "build_model", build)
    monkeypatch.setattr(driver, "batch_relax", relax)
    monkeypatch.setattr(driver, "_model_identity", lambda model, cfg: {"model": "test-model-v1"})
    return hooks


def options(tmp_path, **overrides):
    return {
        "composition": {"Cu": 2}, "output_dir": str(tmp_path / "adaptive"),
        "targets": {"density": {"bounds": [0., 5.], "tolerance": 100.}},
        "batch_size": 2, "min_structures": 4, "max_structures": 12,
        "confidence": .95, "cutoff": 3.,
        "generation": {"seed": 37, "target_density": 1., "target_cn": {}, "retry_mode": "none"},
        "cfg_override": {"model": "lennard-jones", "device": "cpu", "engine": "torchsim",
                         "classical_params": {"params": {"Cu-Cu": {"sigma": 2., "epsilon": .1}},
                                              "cutoff": 5.},
                         "opt": {"cell_filter": "none", "fmax": .3, "max_steps": 10}},
        **overrides,
    }


def saved(config):
    from pathlib import Path
    path = Path(config["output_dir"]) / "adaptive_convergence.json"
    return path, json.loads(path.read_text())


def test_stops_only_after_complete_independent_batches(driver, simulated, tmp_path):
    from amorphgen.pipeline.random_gen import _derive_structure_seed
    config = options(tmp_path)
    config["cfg_override"]["safety"] = {"min_distance": .8}
    config["cfg_override"]["repulsive_core"] = {"enabled": True, "cutoff": .7, "strength": .25}
    result = driver.run_until_converged(**config)
    assert result["status"] == "converged"
    assert result["converged"] is True
    assert result["n_structures"] == 4
    assert result["convergence"]["inference_contract"]["sequentially_valid"] is True
    assert [entry["indices"] for entry in simulated.relaxed] == [[0, 1], [2, 3]]
    assert [entry["seed"] for entry in simulated.generated] == [
        _derive_structure_seed(37, index, 0) for index in range(4)]
    assert len(simulated.built) == 1
    assert all(entry["kwargs"]["autobatch"] is False for entry in simulated.relaxed)
    assert all(entry["kwargs"]["safety"]["min_distance"] == .8 for entry in simulated.relaxed)
    assert all(entry["kwargs"]["repulsive_core"]["strength"] == .25 for entry in simulated.relaxed)
    _, checkpoint = saved(config)
    assert checkpoint["state_sha256"]
    assert len(checkpoint["structures"]) == 4
    for structure in checkpoint["structures"]:
        assert structure["initial_sha256"] and structure["final_sha256"]
        assert np.isfinite(structure["observations"]["density"])


def test_constant_observations_do_not_get_zero_uncertainty(driver, simulated, tmp_path):
    config = options(tmp_path, max_structures=6,
                     targets={"density": {"bounds": [0., 5.], "tolerance": 1e-10}})
    result = driver.run_until_converged(**config)
    assert result["status"] == "max_structures_reached"
    assert result["converged"] is False
    assert result["n_structures"] == 6
    target = result["convergence"]["targets"]["density"]
    assert target["zero_observed_variance"] is True
    assert target["ci_halfwidth"] > 0
    assert len(simulated.relaxed) == 3


def test_native_structural_and_energy_descriptors_are_joint_targets(driver, simulated, tmp_path):
    targets = {
        "coordination.Cu-Cu": {"bounds": [0., 20.], "tolerance": 100.},
        "total_coordination.Cu": {"bounds": [0., 20.], "tolerance": 100.},
        "bond_distance.Cu-Cu": {"bounds": [0., 4.], "tolerance": 100.},
        "energy.per_atom": {"bounds": [-2., 0.], "tolerance": 100.},
        "energy.total": {"bounds": [-2., 0.], "tolerance": 100.},
    }
    config = options(tmp_path, targets=targets, cutoff=4., max_structures=12)
    result = driver.run_until_converged(**config)
    assert result["status"] == "converged"
    assert result["convergence"]["n_components"] == len(targets)
    for row in result["structures"]:
        assert set(row["observations"]) == set(targets)
        assert row["observations"]["energy.per_atom"] == -.5
        assert row["observations"]["energy.total"] == -1.
        assert row["observations"]["coordination.Cu-Cu"] == 1.


def test_missing_structural_descriptor_fails_instead_of_dropping_sample(driver, simulated, tmp_path):
    config = options(tmp_path, cutoff=4., targets={
        "bond_angle.Cu-Cu-Cu": {"bounds": [0., 180.], "tolerance": 100.},
    })
    # Each Cu has one neighbour, so no angle exists for this structure.
    with pytest.raises(ValueError, match="undefined|missing|nonfinite"):
        driver.run_until_converged(**config)
    _, checkpoint = saved(config)
    assert checkpoint["structures"] == []


def test_symmetric_descriptors_accept_either_species_order(driver):
    atoms = Atoms("SiON", positions=[[0., 0., 0.], [1.5, 0., 0.], [0., 1.5, 0.]],
                  cell=[9.] * 3, pbc=True)
    names = ("bond_distance.Si-O", "bond_distance.O-Si",
             "bond_angle.O-Si-N", "bond_angle.N-Si-O")
    observed = driver._observations(atoms, dict.fromkeys(names), cutoff=1.8)
    assert observed["bond_distance.Si-O"] == observed["bond_distance.O-Si"] == 1.5
    assert observed["bond_angle.O-Si-N"] == observed["bond_angle.N-Si-O"] == 90.


def test_resume_completed_run_does_not_generate_more_samples(driver, simulated, tmp_path):
    config = options(tmp_path)
    first = driver.run_until_converged(**config)
    before = len(simulated.generated), len(simulated.relaxed)
    resumed = driver.run_until_converged(**config, resume=True)
    assert (len(simulated.generated), len(simulated.relaxed)) == before
    assert resumed["structures"] == first["structures"]
    assert resumed["convergence"] == first["convergence"]
    assert resumed["history"] == first["history"]


def test_failed_batch_resume_retains_prefix_and_seed_stream(driver, simulated, tmp_path):
    config = options(tmp_path, max_structures=6,
                     targets={"density": {"bounds": [0., 5.], "tolerance": 1e-10}})
    simulated.fail_batch = 2
    with pytest.raises(RuntimeError, match="batch failure"):
        driver.run_until_converged(**config)
    _, interrupted = saved(config)
    assert interrupted["status"] == "failed"
    assert interrupted["n_structures"] == 2
    assert len(interrupted["structures"]) == 2
    first_prefix = deepcopy(interrupted["structures"])
    failed_seeds = simulated.relaxed[-1]["seeds"]
    simulated.fail_batch = None
    resumed = driver.run_until_converged(**config, resume=True)
    assert simulated.relaxed[2]["seeds"] == failed_seeds
    assert resumed["structures"][:2] == first_prefix
    assert [row["index"] for row in resumed["structures"]] == list(range(6))
    assert [row["n_structures"] for row in resumed["history"]] == [2, 4, 6]
    assert resumed["status"] == "max_structures_reached"


def test_generation_failure_does_not_skip_failed_structure(driver, simulated, tmp_path):
    config = options(tmp_path)
    simulated.fail_generation = 2
    with pytest.raises(RuntimeError, match="placement failure"):
        driver.run_until_converged(**config)
    _, failed = saved(config)
    assert failed["n_structures"] == 0
    assert failed["structures"] == []
    assert not simulated.relaxed
    first_seeds = [row["seed"] for row in simulated.generated]
    simulated.fail_generation = None
    driver.run_until_converged(**config, resume=True)
    assert [row["seed"] for row in simulated.generated[2:4]] == first_seeds


def test_each_structure_gets_fresh_generation_settings(driver, simulated, tmp_path):
    config = options(tmp_path)
    config["generation"]["minsep"] = {"Cu-Cu": 2.}
    simulated.mutate_generation = True
    driver.run_until_converged(**config)
    assert [row["generation"]["minsep"] for row in simulated.generated] == [{"Cu-Cu": 2.}] * 4
    assert config["generation"]["minsep"] == {"Cu-Cu": 2.}


@pytest.mark.parametrize("change", [
    {"targets": {"density": {"bounds": [0., 6.], "tolerance": 100.}}},
    {"confidence": .9}, {"batch_size": 3}, {"cutoff": 3.1},
    {"generation": {"seed": 38, "target_density": 1., "target_cn": {}, "retry_mode": "none"}},
])
def test_resume_rejects_changed_contract_before_generation(driver, simulated, tmp_path, change):
    config = options(tmp_path)
    driver.run_until_converged(**config)
    before = len(simulated.generated)
    with pytest.raises(ValueError, match="[Cc]hanged|[Cc]ontract|[Cc]ompatib|[Rr]esume"):
        driver.run_until_converged(**{**config, **change}, resume=True)
    assert len(simulated.generated) == before


@pytest.mark.parametrize("artifact", ["initial_file", "final_file"])
def test_resume_verifies_all_structure_artifacts(driver, simulated, tmp_path, artifact):
    from pathlib import Path
    config = options(tmp_path)
    result = driver.run_until_converged(**config)
    path = Path(config["output_dir"]) / result["structures"][0][artifact]
    path.write_text(path.read_text() + "\nmodified\n")
    before = len(simulated.generated)
    with pytest.raises(ValueError, match="[Aa]rtifact|[Cc]hanged|[Hh]ash|[Ii]ntegrity|[Rr]esume"):
        driver.run_until_converged(**config, resume=True)
    assert len(simulated.generated) == before


def test_resume_rejects_tampered_checkpoint(driver, simulated, tmp_path):
    config = options(tmp_path)
    driver.run_until_converged(**config)
    path, checkpoint = saved(config)
    checkpoint["structures"][0]["observations"]["density"] += .1
    path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="[Ii]ntegrity|[Hh]ash|[Rr]esume|[Cc]heckpoint"):
        driver.run_until_converged(**config, resume=True)


def test_resume_rejects_changed_loaded_model_identity(driver, simulated, monkeypatch, tmp_path):
    config = options(tmp_path)
    driver.run_until_converged(**config)
    before = len(simulated.generated)
    monkeypatch.setattr(driver, "_model_identity", lambda model, cfg: {"model": "different-weights"})
    with pytest.raises(ValueError, match="[Rr]esume|[Mm]odel|[Cc]ontract|[Cc]hanged"):
        driver.run_until_converged(**config, resume=True)
    assert len(simulated.generated) == before


def test_population_bound_violation_is_not_dropped_or_clipped(driver, simulated, tmp_path):
    config = options(tmp_path, targets={"density": {"bounds": [0., .1], "tolerance": 100.}})
    with pytest.raises(ValueError, match="bounds|support"):
        driver.run_until_converged(**config)
    _, failed = saved(config)
    assert failed["n_structures"] == 0
    assert failed["structures"] == []
    assert failed["status"] == "failed"


def test_missing_model_or_settings_provenance_is_not_adopted(driver, simulated, tmp_path):
    from pathlib import Path
    config = options(tmp_path)
    output = Path(config["output_dir"])
    output.mkdir()
    (output / "random_0000.xyz").write_text("untracked initial structure")
    with pytest.raises(ValueError, match="[Nn]onempty|[Mm]etadata|[Cc]heckpoint|[Ee]mpty"):
        driver.run_until_converged(**config)
    assert not simulated.generated


def test_duplicate_seeds_fail_closed(driver, simulated, monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "_derive_structure_seed", lambda *args: 7)
    with pytest.raises(ValueError, match="[Dd]uplicate|[Cc]olli|[Ss]eed"):
        driver.run_until_converged(**options(tmp_path))
    assert not simulated.relaxed


@pytest.mark.parametrize("mode", ["drop", "reverse", "force", "pressure"])
def test_bad_relaxation_batch_is_never_committed(driver, simulated, tmp_path, mode):
    config = options(tmp_path)
    if mode == "drop":
        simulated.drop_output = True
    elif mode == "reverse":
        simulated.reverse_outputs = True
    elif mode == "force":
        simulated.force = np.full(3, .2)  # norm=.346 > fmax=.3; max component=.2.
    else:
        config["cfg_override"]["opt"]["cell_filter"] = "cubic"
        config["cfg_override"]["opt"]["pressure_tol_gpa"] = .02
        simulated.stress = np.eye(3) * .001  # .1602 GPa.
    with pytest.raises((ValueError, RuntimeError)):
        driver.run_until_converged(**config)
    _, checkpoint = saved(config)
    assert checkpoint["status"] == "failed"
    assert checkpoint["structures"] == []
    assert checkpoint["n_structures"] == 0


def test_output_directory_lock_prevents_concurrent_writers(driver, simulated, tmp_path):
    from amorphgen.utils.run_lock import run_lock
    config = options(tmp_path)
    with run_lock(config["output_dir"]):
        with pytest.raises(RuntimeError, match="Another run"):
            driver.run_until_converged(**config)
    assert not simulated.generated
