"""Run-until-converged setting validation, stopping rule and checkpoint guards."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

from amorphgen.pipeline import until_converged as driver


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


@pytest.fixture
def stubs(monkeypatch):
    """Record generation, model loading and relaxation without torch-sim."""
    hooks = SimpleNamespace(generated=[], built=[], relaxed=[], on_build=None)

    def generate(composition, seed=None, **kwargs):
        hooks.generated.append(seed)
        shift = (int(seed) % 19) / 1000.
        return Atoms("Cu2", positions=[[1. + shift, 1., 1.], [3., 3., 3.]],
                     cell=[6.] * 3, pbc=True)

    def build(*args, **kwargs):
        hooks.built.append(kwargs)
        if hooks.on_build is not None:
            hooks.on_build()
        return SimpleNamespace(eval=lambda: None)

    def relax(frames, model, **kwargs):
        hooks.relaxed.append([atoms.info["sequential_index"] for atoms in frames])
        output = []
        for original in frames:
            atoms = original.copy()
            atoms.calc = SinglePointCalculator(atoms, energy=-1., forces=np.zeros((2, 3)),
                                               stress=np.zeros((3, 3)))
            output.append(atoms)
        return output

    monkeypatch.setattr(driver, "generate_random", generate)
    monkeypatch.setattr(driver, "build_model", build)
    monkeypatch.setattr(driver, "batch_relax", relax)
    monkeypatch.setattr(driver, "_model_identity", lambda model, cfg: {"model": "stub"})
    return hooks


def _change(config, path, value):
    *parents, key = path.split(".")
    target = config
    for name in parents:
        target = target[name]
    target[key] = value


@pytest.mark.parametrize("path, value, message", [
    ("batch_size", 0, "batch_size must be an integer >= 1"),
    ("batch_size", True, "batch_size must be an integer >= 1"),
    ("batch_size", 2.0, "batch_size must be an integer >= 1"),
    ("min_structures", 1, "min_structures must be an integer >= 2"),
    ("max_structures", 3, "max_structures must be an integer >= 4"),
    ("composition", {}, "composition must be a nonempty mapping"),
    ("composition", [("Cu", 2)], "composition must be a nonempty mapping"),
    ("composition", {"Xx": 2}, "Unknown composition species 'Xx'"),
    ("composition", {"Cu": 0}, "composition.Cu must be an integer >= 1"),
    ("composition", {"Cu": np.bool_(True)}, "composition.Cu must be an integer >= 1"),
    ("targets", {}, "targets must be a nonempty mapping"),
    ("targets", {"density": {"bounds": [0., 5.], "tolerance": 1., "components": 2}},
     "scalar descriptors only"),
    ("targets", {"rdf.Cu-Cu": {"bounds": [0., 5.], "tolerance": 1.}},
     "Unsupported sequential descriptor 'rdf.Cu-Cu'"),
    ("targets", {"coordination.Cu-O": {"bounds": [0., 5.], "tolerance": 1.}},
     "Unsupported sequential descriptor 'coordination.Cu-O'"),
    ("cutoff", float("nan"), "cutoff must be finite and positive"),
    ("cutoff", True, "cutoff must be finite and positive"),
    ("confidence", 1.0, "confidence must be strictly between zero and one"),
    ("generation", None, "generation must specify an explicit nonnegative seed"),
    ("generation", {"target_density": 1.}, "generation.seed must be an integer >= 0"),
    ("generation", {"seed": -1}, "generation.seed must be an integer >= 0"),
    ("generation", {"seed": 1, "_soft_pack": False}, "Unsupported or private"),
    ("generation", {"seed": 1, "temperature": 300}, "Unsupported or private"),
    ("cfg_override.engine", "ase", "requires engine='torchsim'"),
    ("cfg_override.opt.fmax", 0, "opt.fmax must be finite and positive"),
    ("cfg_override.opt.fmax", float("inf"), "opt.fmax must be finite and positive"),
    ("cfg_override.opt.max_steps", 0, "opt.max_steps must be an integer >= 1"),
    ("cfg_override.opt.pressure_tol_gpa", -.01, "opt.pressure_tol_gpa must be finite and positive"),
    ("cfg_override.opt.batch_size", "two", "opt.batch_size must be an integer >= 1"),
    ("cfg_override.opt.batch_size", 0, "opt.batch_size must be an integer >= 1"),
    ("cfg_override.default_dtype", "float16", "default_dtype must be auto, float32 or float64"),
])
def test_invalid_settings_fail_before_the_output_directory_is_touched(
        stubs, tmp_path, path, value, message):
    config = options(tmp_path)
    _change(config, path, value)
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        driver.run_until_converged(**config)
    assert not (tmp_path / "adaptive").exists()
    assert not stubs.built and not stubs.generated


def test_structural_descriptor_requires_a_fixed_cutoff(stubs, tmp_path):
    config = options(tmp_path, cutoff=None,
                     targets={"coordination.Cu-Cu": {"bounds": [0., 12.], "tolerance": 1.}})
    with pytest.raises(ValueError, match="fixed numeric cutoff"):
        driver.run_until_converged(**config)
    # A scalar-only target needs no cutoff at all.
    result = driver.run_until_converged(**options(tmp_path, cutoff=None))
    assert result["status"] == "converged"


@pytest.mark.parametrize("chunk, expected", [
    ("1", [[0], [1], [2], [3]]),
    ("auto", [[0, 1], [2, 3]]),
    (None, [[0, 1], [2, 3]]),
    (3, [[0, 1], [2, 3]]),
])
def test_relaxation_chunks_never_span_generation_batches(stubs, tmp_path, chunk, expected):
    config = options(tmp_path)
    if chunk is not None:
        config["cfg_override"]["opt"]["batch_size"] = chunk
    result = driver.run_until_converged(**config)
    assert stubs.relaxed == expected
    assert [row["n_structures"] for row in result["history"]] == [2, 4]


@pytest.mark.parametrize("requested, loaded", [(None, "float64"), ("auto", "float64"),
                                               ("float32", "float32"), ("float64", "float64")])
def test_model_precision_defaults_to_float64(stubs, tmp_path, requested, loaded):
    config = options(tmp_path)
    if requested is not None:
        config["cfg_override"]["default_dtype"] = requested
    driver.run_until_converged(**config)
    assert [kwargs["dtype"] for kwargs in stubs.built] == [loaded]


def test_minimum_count_is_reached_with_whole_batches(stubs, tmp_path):
    # Precision is met from the first batch, but min_structures=3 forces a second
    # complete batch: the run never stops at a partial batch of one.
    result = driver.run_until_converged(**options(tmp_path, min_structures=3))
    assert result["n_structures"] == 4
    assert result["history"] == [{"n_structures": 2, "converged": False},
                                 {"n_structures": 4, "converged": True}]
    assert result["stopping_reason"] == "precision_met"


def test_cap_truncates_last_batch_and_resumes_without_new_draws(stubs, tmp_path):
    targets = {"density": {"bounds": [0., 5.], "tolerance": 1e-10}}
    config = options(tmp_path, max_structures=5, targets=targets)
    first = driver.run_until_converged(**config)
    assert stubs.relaxed == [[0, 1], [2, 3], [4]]
    assert first["status"] == "max_structures_reached"
    assert first["stopping_reason"] == "max_structures_reached"
    assert first["converged"] is False
    assert first["n_structures"] == 5
    # Five structures are not a multiple of the batch size; at the cap that is valid.
    generated = len(stubs.generated)
    resumed = driver.run_until_converged(**config, resume=True)
    assert len(stubs.generated) == generated
    assert resumed["structures"] == first["structures"]


def test_resume_requires_an_existing_checkpoint(stubs, tmp_path):
    with pytest.raises(ValueError, match="Cannot resume without adaptive_convergence.json"):
        driver.run_until_converged(**options(tmp_path), resume=True)
    assert not stubs.built


def test_model_file_changed_while_loading_is_rejected(stubs, tmp_path):
    weights = tmp_path / "weights.model"
    weights.write_bytes(b"original weights")
    config = options(tmp_path)
    config["cfg_override"]["model_path"] = str(weights)
    stubs.on_build = lambda: weights.write_bytes(b"replaced during load")
    with pytest.raises(ValueError, match="Model files changed during loading"):
        driver.run_until_converged(**config)
    assert not (tmp_path / "adaptive" / driver.CHECKPOINT).exists()
    assert not stubs.generated


def test_reference_model_file_is_part_of_the_resume_contract(stubs, tmp_path):
    reference = tmp_path / "reference.model"
    reference.write_bytes(b"reference weights")
    config = options(tmp_path)
    config["cfg_override"]["safety"] = {"reference": {"model_path": str(reference)}}
    driver.run_until_converged(**config)
    checkpoint = json.loads((tmp_path / "adaptive" / driver.CHECKPOINT).read_text())
    assert checkpoint["contract"]["model_files"] == {
        str(reference.resolve()): hashlib.sha256(b"reference weights").hexdigest()}
    reference.write_bytes(b"retrained reference")
    generated = len(stubs.generated)
    with pytest.raises(ValueError, match="frozen generation, targets, model or software changed"):
        driver.run_until_converged(**config, resume=True)
    assert len(stubs.generated) == generated


def _rewrite(path, mutate):
    """Tamper with a checkpoint and re-seal it with a valid self-checksum."""
    state = json.loads(path.read_text())
    mutate(state)
    state["state_sha256"] = driver._digest({k: v for k, v in state.items() if k != "state_sha256"})
    path.write_text(json.dumps(state))


def _bump(row, key):
    row[key] += 1


@pytest.mark.parametrize("mutate, message", [
    (lambda s: s.update(schema="amorphgen.adaptive_random.v0"), "Invalid adaptive convergence checkpoint"),
    (lambda s: s.update(n_structures=3), "structure count is inconsistent"),
    (lambda s: s.update(structures=s["structures"][:3], n_structures=3), "incomplete or oversized batch"),
    (lambda s: _bump(s["structures"][0], "seed"), "contiguous seed prefix"),
    (lambda s: s["structures"][1].update(index=0), "contiguous seed prefix"),
    (lambda s: s["structures"][0].update(final_file="random_opt/random_0001_opt.xyz"),
     "Invalid adaptive checkpoint structure path"),
    (lambda s: _bump(s["structures"][0]["observations"], "density"),
     "observations or identities disagree"),
    (lambda s: s["convergence"]["targets"]["density"].update(ci_halfwidth=1.0),
     "confidence sequence disagrees"),
])
def test_resealed_checkpoint_tampering_is_still_detected(stubs, tmp_path, mutate, message):
    config = options(tmp_path)
    driver.run_until_converged(**config)
    _rewrite(Path(config["output_dir"]) / driver.CHECKPOINT, mutate)
    generated = len(stubs.generated)
    with pytest.raises(ValueError, match=message):
        driver.run_until_converged(**config, resume=True)
    assert len(stubs.generated) == generated


def test_symlinked_checkpoint_or_structure_directory_is_refused(stubs, tmp_path):
    config = options(tmp_path)
    driver.run_until_converged(**config)
    output = Path(config["output_dir"])
    real = output / "random_opt_real"
    (output / "random_opt").rename(real)
    (output / "random_opt").symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="Invalid adaptive checkpoint structure path"):
        driver.run_until_converged(**config, resume=True)

    checkpoint = output / driver.CHECKPOINT
    copy = tmp_path / "copy.json"
    copy.write_bytes(checkpoint.read_bytes())
    checkpoint.unlink()
    checkpoint.symlink_to(copy)
    with pytest.raises(ValueError, match="symlink checkpoint"):
        driver.run_until_converged(**config, resume=True)


def test_structure_writer_refuses_symlinks_and_replaces_stale_pending_file(tmp_path):
    atoms = Atoms("Cu2", positions=[[0, 0, 0], [1.8, 0, 0]], cell=[5.] * 3, pbc=True)
    target = tmp_path / "random_opt" / "random_0000_opt.xyz"
    target.parent.mkdir()
    stale = target.with_name("." + target.name + ".pending")
    stale.write_text("interrupted write")
    driver._write_atoms(target, atoms)
    assert not stale.exists()
    np.testing.assert_allclose(read(target).positions, atoms.positions)

    elsewhere = tmp_path / "elsewhere.xyz"
    link = target.with_name("random_0001_opt.xyz")
    link.symlink_to(elsewhere)
    with pytest.raises(ValueError, match="Refusing symlink"):
        driver._write_atoms(link, atoms)
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(target.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Refusing symlink"):
        driver._write_atoms(linked_dir / "random_0002_opt.xyz", atoms)
    assert not elsewhere.exists()
    assert sorted(os.listdir(target.parent)) == ["random_0000_opt.xyz", "random_0001_opt.xyz"]


@pytest.mark.parametrize("atoms", [
    Atoms(cell=[5.] * 3, pbc=True),
    Atoms("Cu", positions=[[np.nan, 0, 0]], cell=[5.] * 3, pbc=True),
    Atoms("Cu", positions=[[0, 0, 0]], cell=[[5, 0, 0], [0, 5, 0], [5, 5, 0]]),
], ids=["empty", "nonfinite", "zero-volume"])
def test_observations_reject_degenerate_structures(atoms):
    with pytest.raises(ValueError, match="finite structures with positive volume"):
        driver._observations(atoms, {"density": {}}, None)


def test_observations_match_mass_over_volume_and_reject_nonfinite_energy():
    atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[6.] * 3, pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=-3.0)
    observed = driver._observations(atoms, {"density": {}, "energy.per_atom": {}}, None)
    # 2 x 63.546 amu in 216 A^3; 1 amu/A^3 = 1.66053907 g/cm^3.
    assert observed["density"] == pytest.approx(2 * 63.546 * 1.66053907 / 216, rel=1e-4)
    assert observed["energy.per_atom"] == -1.5
    atoms.calc = SinglePointCalculator(atoms, energy=float("nan"))
    with pytest.raises(ValueError, match="'energy.total' is missing or nonfinite"):
        driver._observations(atoms, {"energy.total": {}}, None)


class _Weights:
    def __init__(self, value):
        self.value = np.array([value, 2.0])

    def state_dict(self):
        return {"weight": self.value}


def test_model_identity_requires_weight_fingerprint_except_for_native_lj():
    cfg = {"model": "mace-mpa-0", "device": "cpu"}
    with pytest.raises(ValueError, match="fingerprint of loaded model weights"):
        driver._model_identity(SimpleNamespace(), cfg)
    for name in ("lj", "Lennard-Jones"):
        identity = driver._model_identity(SimpleNamespace(), {**cfg, "model": name})
        assert identity["model"]["sha256"] is None
        assert "hash_unavailable_reason" not in identity["model"]
    first = driver._model_identity(_Weights(1.0), cfg)["model"]["sha256"]
    assert len(first) == 64
    assert driver._model_identity(_Weights(1.0), cfg)["model"]["sha256"] == first
    assert driver._model_identity(_Weights(1.5), cfg)["model"]["sha256"] != first
