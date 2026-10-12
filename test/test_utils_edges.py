"""Edge cases of the shared MD, snapshot, repulsion, provenance and lock utilities."""

import errno
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from amorphgen.utils import common
from amorphgen.utils.common import (
    compute_pressure_GPa, extract_snapshots, parse_index_spec, read_md_checkpoint,
    resolve_ramp, resume_md_stage, select_frame_indices, set_md_temperature,
)
from amorphgen.utils.repulsion import with_repulsive_core, wrap_torch_model
from amorphgen.utils.run_lock import run_lock
from amorphgen.utils.run_provenance import calculator_provenance
from amorphgen.utils.snapshot_sampling import _autocorrelation, analyze_snapshot_sampling


CORE = {"enabled": True, "cutoff": 1.0, "strength": 2.0}


def _frames(n=6, energies=True):
    frames = []
    for k in range(n):
        atoms = Atoms("Cu2", positions=[[1 + 0.05 * k, 1, 1], [3, 3, 3]], cell=[6.0] * 3, pbc=True)
        if energies:
            atoms.calc = SinglePointCalculator(atoms, energy=-1.0 - 0.01 * k)
        frames.append(atoms)
    return frames


@pytest.fixture
def trajectory(tmp_path):
    path = tmp_path / "md_traj.xyz"
    write(path, _frames(), format="extxyz")
    return path


# -- index, ramp and frame selection -----------------------------------------

def test_index_spec_ignores_empty_parts_and_rejects_bad_tokens():
    assert parse_index_spec("1,,3, 5-6,") == {1, 3, 5, 6}
    for spec, message in (("a-3", "bad index range 'a-3'"), ("1-b", "bad index range '1-b'"),
                          ("x", "bad index 'x'"), ("-2", "bad index range '-2'")):
        with pytest.raises(ValueError, match=message):
            parse_index_spec(spec)


@pytest.mark.parametrize("start, end, step, expected", [
    (0, 1.001, 0.5, [0.5, 1.0]),        # 1.0 and round(1.001) coincide: kept once
    (1.001, 0, 0.5, [0.5, 0.0]),
    (300, 1000, 300, [600.0, 900.0, 1000.0]),
])
def test_ramp_never_repeats_a_rounded_endpoint(start, end, step, expected):
    assert resolve_ramp(start, end, step) == expected


def test_unknown_frame_selection_is_rejected():
    with pytest.raises(ValueError, match="Unknown selection strategy 'random'"):
        select_frame_indices(10, 3, "random")


def test_pressure_without_calculator_is_nan():
    assert np.isnan(compute_pressure_GPa(Atoms("Cu", cell=[3.0] * 3, pbc=True)))


# -- temperature control and MD checkpoints -----------------------------------

def test_temperature_attributes_are_set_without_thermostat_objects():
    dyn = SimpleNamespace(_kT=0.0, _temperature_K=0.0)
    set_md_temperature(dyn, 500.0)
    assert dyn._kT == pytest.approx(500.0 * units.kB)
    assert dyn._temperature_K == 500.0
    with pytest.raises(AttributeError, match="SimpleNamespace has no set_temperature"):
        set_md_temperature(SimpleNamespace(), 500.0)


def test_unreadable_trajectory_has_no_checkpoint(tmp_path):
    path = tmp_path / "md_traj.xyz"
    path.write_text("not a trajectory\n")
    assert read_md_checkpoint(str(path)) is None


def test_torn_trajectory_keeps_complete_frames_when_rewrite_fails(trajectory, monkeypatch):
    with open(trajectory, "a") as handle:
        handle.write("2\nLattice=\"6 0 0 0 6 0 0 0 6\" Properties=species:S:1:pos:R:3\nCu 1.0")
    torn = trajectory.read_bytes()

    def refuse(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(common, "write", refuse)
    with pytest.warns(UserWarning, match="last frame is incomplete.*6 complete frame"):
        atoms, elapsed = read_md_checkpoint(str(trajectory), interval=10)
    assert elapsed == 50
    np.testing.assert_allclose(atoms.positions, _frames()[-1].positions)
    assert trajectory.read_bytes() == torn


def test_resume_falls_back_to_the_legacy_trajectory(trajectory, tmp_path, capsys):
    missing = str(tmp_path / "stage4_eq_traj.xyz")
    atoms, elapsed = resume_md_stage(missing, True, "4", legacy_trajfile=str(trajectory))
    assert elapsed == 5 * common.TRAJ_LOG_INTERVAL
    np.testing.assert_allclose(atoms.positions, _frames()[-1].positions)
    assert "Using legacy trajectory" in capsys.readouterr().out
    assert resume_md_stage(missing, True, "4", legacy_trajfile=missing) == (None, 0)


# -- snapshot extraction -----------------------------------------------------

def test_snapshot_format_and_burn_in_are_validated(trajectory, tmp_path):
    with pytest.raises(ValueError, match="Unknown output_format 'pdb'"):
        extract_snapshots(str(trajectory), output_dir=str(tmp_path / "s"), output_format="pdb")
    with pytest.raises(ValueError, match="burn_in_frames must be >= 0, got -1"):
        extract_snapshots(str(trajectory), output_dir=str(tmp_path / "s"), burn_in_frames=-1)
    assert not (tmp_path / "s").exists()


def test_oversized_request_uses_every_frame_after_burn_in(trajectory, tmp_path, capsys):
    paths = extract_snapshots(str(trajectory), n_snapshots=10, burn_in_frames=2,
                              output_dir=str(tmp_path / "s"))
    assert "only 4 frames are available after burn-in" in capsys.readouterr().out
    assert [p.rsplit("frame", 1)[1] for p in paths] == [f"{k:05d}.xyz" for k in range(2, 6)]
    for path, k in zip(paths, range(2, 6)):
        np.testing.assert_allclose(read(path).positions, _frames()[k].positions)


def test_report_name_gets_json_suffix_and_matches_files(trajectory, tmp_path):
    paths = extract_snapshots(str(trajectory), n_snapshots=3, output_dir=str(tmp_path / "s"),
                              report_path=str(tmp_path / "selection"))
    report = json.loads((tmp_path / "selection.json").read_text())
    assert report["selected_frame_indices"] == [0, 2, 5]
    assert report["snapshot_files"] == paths
    assert (tmp_path / "selection.txt").is_file()


def test_unreadable_resume_report_stops_extraction(trajectory, tmp_path):
    report = tmp_path / "selection.json"
    report.write_text("{truncated")
    with pytest.raises(ValueError, match="Cannot validate snapshot selection for resume"):
        extract_snapshots(str(trajectory), n_snapshots=3, output_dir=str(tmp_path / "s"),
                          report_path=str(report), resume=True)
    assert not (tmp_path / "s").exists()


# -- snapshot sampling diagnostics -------------------------------------------

def test_constant_series_autocorrelation_is_one():
    np.testing.assert_array_equal(_autocorrelation(np.full(5, 2.5)), np.ones(5))


def _changed(frames, k, change):
    change(frames[k])
    return frames


@pytest.mark.parametrize("frames, message", [
    ([Atoms(cell=[5.0] * 3, pbc=True)] * 3, "containing no atoms"),
    (_changed(_frames(energies=False), 2, lambda a: a.set_chemical_symbols(["Cu", "Ag"])),
     "identities, masses and periodicity must be constant"),
    (_changed(_frames(energies=False), 2, lambda a: a.positions.__setitem__((0, 0), np.nan)),
     "positive masses must be finite"),
    ([Atoms("Cu", cell=[5.0, 5.0, 0.0], pbc=True)] * 3, "nonzero length"),
], ids=["empty", "identity", "nonfinite", "flat-cell"])
def test_sampling_rejects_inconsistent_trajectories(frames, message):
    with pytest.raises(ValueError, match=message):
        analyze_snapshot_sampling(frames, n_snapshots=2, select="uniform")


def test_sampling_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown selection strategy 'random'"):
        analyze_snapshot_sampling(_frames(), select="random")


def test_nonfinite_cached_energy_is_not_used():
    frames = _frames(10)
    frames[4].calc = SinglePointCalculator(frames[4], energy=np.inf)
    report = analyze_snapshot_sampling(frames, n_snapshots=3, select="uniform")
    assert report["autocorrelation"]["energy_per_atom"] == {"status": "unavailable"}
    assert any("energies are unavailable" in w for w in report["warnings"])
    json.dumps(report, allow_nan=False)


def test_isolated_atom_has_no_diffusion_distance_and_one_snapshot():
    frames = [Atoms("Si", positions=[[1 + 0.01 * k, 1, 1]]) for k in range(12)]
    report = analyze_snapshot_sampling(frames, n_snapshots=4)
    assert report["decorrelation_distance_angstrom"] is None
    assert any("nearest-neighbor distance could not be determined" in w for w in report["warnings"])
    assert report["selected_frame_indices"] == [11]


# -- repulsive core ----------------------------------------------------------

class _ZeroCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3)),
                        "stress": np.zeros(6), "magmom": 7.0}


def test_core_returns_only_advertised_properties():
    atoms = Atoms("He2", positions=[[0, 0, 0], [0.8, 0, 0]], cell=[4.0] * 3)
    atoms.calc = with_repulsive_core(_ZeroCalculator(), CORE)
    # U = strength (cutoff / r - 1)^2 = 2 * 0.25^2 for one pair at 0.8 A.
    assert atoms.get_potential_energy() == pytest.approx(0.125)
    assert set(atoms.calc.results) == {"energy", "forces", "stress"}


def test_core_stress_needs_a_three_dimensional_cell():
    atoms = Atoms("He2", positions=[[0, 0, 0], [0.8, 0, 0]], cell=[4.0, 4.0, 0.0])
    atoms.calc = with_repulsive_core(_ZeroCalculator(), CORE)
    with pytest.raises(ValueError, match="nonzero three-dimensional cell"):
        atoms.get_stress()


def test_torch_core_wrapper_delegates_attributes_and_flags():
    torch = pytest.importorskip("torch")
    ts = pytest.importorskip("torch_sim")
    from torch_sim.models.interface import ModelInterface

    class ZeroModel(ModelInterface):
        def __init__(self):
            super().__init__()
            self._device = torch.device("cpu")
            self._dtype = torch.float64
            self._compute_forces = True
            self._compute_stress = True
            self.label = "zero-model"

        @property
        def compute_forces(self):
            return self._compute_forces

        @compute_forces.setter
        def compute_forces(self, value):
            self._compute_forces = value

        @property
        def compute_stress(self):
            return self._compute_stress

        @compute_stress.setter
        def compute_stress(self, value):
            self._compute_stress = value

        def forward(self, state, **kwargs):
            return {"energy": state.positions.new_zeros(state.n_systems),
                    "forces": torch.zeros_like(state.positions),
                    "stress": torch.zeros_like(state.cell)}

    model = ZeroModel()
    wrapped = wrap_torch_model(model, CORE)
    assert wrapped.label == "zero-model"
    wrapped.compute_forces = False
    wrapped.compute_stress = False
    assert (model._compute_forces, model._compute_stress) == (False, False)
    assert (wrapped.compute_forces, wrapped.compute_stress) == (False, False)
    wrapped.compute_stress = True
    flat = Atoms("He2", positions=[[0, 0, 0], [0.6, 0, 0]], cell=[4.0, 4.0, 0.0])
    state = ts.io.atoms_to_state([flat], device=model.device, dtype=model.dtype)
    with pytest.raises(ValueError, match="nonzero three-dimensional cells"):
        wrapped(state)


# -- provenance --------------------------------------------------------------

def test_broken_parameter_iterator_falls_back_to_model_attributes():
    def parameters():
        raise RuntimeError("meta tensors")

    model = SimpleNamespace(parameters=parameters, dtype="torch.float32", device="cuda:0")
    result = calculator_provenance({}, SimpleNamespace(model=model), injected=True)
    assert result["precision"]["resolved"] == "float32"
    assert result["device"]["resolved"] == "cuda:0"


@pytest.mark.parametrize("cuda, mps, expected", [
    (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu"), (RuntimeError, False, None),
])
def test_auto_device_resolution_uses_an_already_imported_torch(monkeypatch, cuda, mps, expected):
    def available(value):
        def check():
            if value is RuntimeError:
                raise RuntimeError("driver failure")
            return value
        return check

    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=available(cuda)),
                            backends=SimpleNamespace(mps=SimpleNamespace(is_available=available(mps))))
    monkeypatch.setitem(sys.modules, "torch", torch)
    result = calculator_provenance({"model": "custom", "device": "auto"}, SimpleNamespace())
    assert result["device"]["resolved"] == expected


def test_non_mapping_state_dict_has_no_weight_hash():
    model = SimpleNamespace(state_dict=lambda: [np.zeros(2)])
    result = calculator_provenance({}, SimpleNamespace(models=[model]), injected=True)
    assert result["model"]["sha256"] is None
    assert result["model"]["hash_unavailable_reason"] == "ValueError: Model state_dict is not a mapping"


def test_loaded_lammps_calculator_hashes_its_own_potential_files(tmp_path):
    potential = tmp_path / "Cu.eam"
    potential.write_bytes(b"eam potential v1")

    def identity(commands):
        calc = SimpleNamespace(potential_files=[str(potential)],
                               parameters=SimpleNamespace(lmpcmds=commands))
        return calculator_provenance({"model": "lammps"}, calc)["model"]

    commands = ["pair_style eam", f"pair_coeff 1 1 {potential}"]
    first = identity(commands)
    assert first["hash_source"] == "lammps-v1"
    assert first["files"] == {str(potential): hashlib.sha256(b"eam potential v1").hexdigest()}
    assert identity(list(commands))["sha256"] == first["sha256"]
    assert identity(commands + ["neighbor 2.0 bin"])["sha256"] != first["sha256"]
    potential.write_bytes(b"eam potential v2")
    assert identity(commands)["sha256"] != first["sha256"]


# -- output directory lock ---------------------------------------------------

def test_unexpected_lock_errors_are_not_reported_as_concurrent_runs(tmp_path, monkeypatch):
    import fcntl

    def unsupported(fd, operation):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", unsupported)
    with pytest.raises(OSError) as info:
        with run_lock(tmp_path):
            pass
    assert info.value.errno == errno.ENOLCK
    assert not isinstance(info.value, RuntimeError)
