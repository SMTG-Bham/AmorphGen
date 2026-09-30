"""Regression tests for collecting the singleton and multi-run batch layouts."""

from pathlib import Path

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read, write

from amorphgen import cli
from amorphgen.pipeline import batch_quench
from amorphgen.pipeline.random_gen import _FORMAT_MAP


@pytest.fixture
def fast_batch(monkeypatch):
    """Run the actual batch file handling without spending time in MD."""
    import amorphgen.utils

    calls = []

    def unchanged(atoms, **kwargs):
        calls.append(kwargs)
        return atoms

    monkeypatch.setattr(amorphgen.utils, "get_calculator", lambda **kwargs: EMT())
    monkeypatch.setattr(batch_quench.quench, "run", unchanged)
    monkeypatch.setattr(batch_quench.equilibrate, "run", unchanged)
    monkeypatch.setattr(batch_quench.final_opt, "run", unchanged)
    return calls


def _write_final(directory, name="final_amorphous.xyz", symbol="Cu"):
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / name, bulk(symbol, cubic=True), format="extxyz")


def test_mq_ensemble_default_one_structure_is_collected(tmp_path, monkeypatch, fast_batch):
    from amorphgen.pipeline import run_pipeline

    class SharedPipeline:
        def __init__(self, input_file, work_dir, cfg_override):
            self.work_dir = Path(work_dir)

        def run(self, stages, resume):
            self.work_dir.mkdir(parents=True, exist_ok=True)
            write(self.work_dir / "stage4_eq_traj.xyz",
                  [bulk("Cu", cubic=True)] * 3, format="extxyz")

    monkeypatch.setattr(run_pipeline, "MeltQuenchPipeline", SharedPipeline)
    work = tmp_path / "ensemble"
    args = cli._get_parser().parse_args([
        "crystal.xyz", "--mq-ensemble", "-o", str(work),
    ])
    assert args.n_structures == 1

    cli._run_mq_ensemble(args, {})

    assert (work / "quench_runs" / "final_amorphous.xyz").is_file()
    assert not list((work / "quench_runs").glob("run_*"))
    final = work / "final" / "mq_0000.xyz"
    assert final.is_file()
    assert read(final).get_chemical_formula() == "Cu4"
    assert len(fast_batch) == 3


def test_one_input_array_hybrid_collects_snapshot_index_and_resumes(
        tmp_path, monkeypatch, fast_batch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write(inputs / "snapshot_0007_frame00184.xyz", bulk("Cu", cubic=True))
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    work = tmp_path / "task_7"
    args = cli._get_parser().parse_args([
        "--hybrid-ensemble", "--input-dir", str(inputs),
        "--format", "vasp", "-o", str(work),
    ])

    cli._run_hybrid_ensemble(args, {})
    assert len(fast_batch) == 4
    assert (work / "quench_runs" / "final_amorphous.xyz").is_file()
    final = work / "final" / "hybrid_0007.vasp"
    assert read(final).get_chemical_formula() == "Cu4"

    final.unlink()
    args.resume = True
    cli._run_hybrid_ensemble(args, {})
    assert final.is_file()
    assert len(fast_batch) == 4, "resume should reuse the flat batch output"


def test_multi_input_collection_keeps_indices_and_legacy_outputs(tmp_path):
    work = tmp_path / "quench_runs"
    _write_final(work / "run_0007")
    _write_final(work / "run_0011", "final_amorphous.extxyz", symbol="Ag")
    _write_final(work / "run_0099")  # unrelated output from an earlier batch
    _write_final(work)  # unrelated flat output must not count in a multi-input batch
    final = tmp_path / "final"

    cli._collect_ensemble_final(
        str(work), str(final), "xyz", "mq", _FORMAT_MAP,
        snapshot_files=["snapshot_0007.xyz", "snapshot_0011.xyz"],
    )

    assert sorted(p.name for p in final.iterdir()) == ["mq_0007.xyz", "mq_0011.xyz"]
    assert read(final / "mq_0011.xyz").get_chemical_formula() == "Ag4"


@pytest.mark.parametrize("extension", ["xyz", "extxyz"])
def test_flat_legacy_and_current_outputs_are_collected(tmp_path, extension):
    work = tmp_path / "quench_runs"
    _write_final(work, f"final_amorphous.{extension}")
    final = tmp_path / "final"
    cli._collect_ensemble_final(str(work), str(final), "xyz", "mq", _FORMAT_MAP)
    assert (final / "mq_0000.xyz").is_file()


def test_torchsim_singleton_uses_nested_output_even_with_stale_flat_file(tmp_path):
    work = tmp_path / "quench_runs"
    _write_final(work, symbol="Ag")
    _write_final(work / "run_0007", "final_amorphous.extxyz")
    final = tmp_path / "final"

    cli._collect_ensemble_final(
        str(work), str(final), "xyz", "hybrid", _FORMAT_MAP,
        snapshot_files=["snapshot_0007.xyz"], flat_single_run=False,
    )
    assert read(final / "hybrid_0007.xyz").get_chemical_formula() == "Cu4"


def test_collection_rejects_missing_outputs_instead_of_reporting_success(tmp_path):
    work = tmp_path / "quench_runs"
    _write_final(work / "run_0007")
    final = tmp_path / "final"
    with pytest.raises(FileNotFoundError, match="snapshot_0011"):
        cli._collect_ensemble_final(
            str(work), str(final), "xyz", "mq", _FORMAT_MAP,
            snapshot_files=["snapshot_0007.xyz", "snapshot_0011.xyz"],
        )
    assert not final.exists()

    with pytest.raises(FileNotFoundError, match="No final batch outputs"):
        cli._collect_ensemble_final(
            str(tmp_path / "empty"), str(final), "xyz", "mq", _FORMAT_MAP,
        )
    assert not final.exists()
