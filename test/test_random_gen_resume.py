"""Tests for --resume support in random structure generation."""

import os
import pytest
import numpy as np
from ase.io import read, write

from amorphgen.pipeline.random_gen import batch_random


class TestRandomGenResume:

    def test_resume_skips_existing(self, tmp_path, monkeypatch):
        """Generate 5, delete 2, resume should produce 5 total."""
        from unittest.mock import Mock
        from amorphgen.pipeline import random_gen
        out = str(tmp_path / "structures")
        initial = os.path.join(out, "random_initial")   # v1.0.0rc2 subdir
        comp = {"Si": 8, "O": 16}

        # First run: generate 5
        paths1 = batch_random(comp, n_structures=5, output_dir=out, seed=42)
        assert len(paths1) == 5

        existing = sorted((tmp_path / "structures" / "random_initial").glob("*.xyz"))[:3]
        before = {path: path.read_bytes() for path in existing}

        # Delete structures 3 and 4
        os.remove(os.path.join(initial, "random_0003.xyz"))
        os.remove(os.path.join(initial, "random_0004.xyz"))

        # Resume: should regenerate 3 and 4, skip 0-2
        generate = Mock(wraps=random_gen.generate_random)
        monkeypatch.setattr(random_gen, "generate_random", generate)
        paths2 = batch_random(comp, n_structures=5, output_dir=out,
                              seed=42, resume=True)
        assert sorted(paths2) == sorted(paths1)
        assert generate.call_count == 2

        # Should have 5 paths total (3 existing + 2 new)
        all_files = sorted(f for f in os.listdir(initial)
                           if f.startswith("random_") and f.endswith(".xyz")
                           and "_opt" not in f)
        assert len(all_files) == 5

        # Existing structures should be unchanged
        assert {path: path.read_bytes() for path in existing} == before

    def test_resume_regenerates_corrupted(self, tmp_path):
        """Corrupted (empty) file should be regenerated."""
        out = str(tmp_path / "structures")
        initial = os.path.join(out, "random_initial")   # v1.0.0rc2 subdir
        comp = {"Si": 8, "O": 16}

        # Generate 3
        batch_random(comp, n_structures=3, output_dir=out, seed=42)

        # Corrupt structure 1 (make it empty)
        corrupt_path = os.path.join(initial, "random_0001.xyz")
        with open(corrupt_path, "w") as f:
            f.write("")

        # Resume: should regenerate structure 1
        batch_random(comp, n_structures=3, output_dir=out,
                     seed=42, resume=True)

        # File should now be valid
        atoms = read(os.path.join(initial, "random_0001.xyz"))
        assert len(atoms) == 24  # Si8O16

    def test_resume_empty_dir(self, tmp_path):
        """Resume on empty directory should behave like fresh run."""
        out = str(tmp_path / "structures")
        comp = {"Si": 8, "O": 16}

        paths = batch_random(comp, n_structures=3, output_dir=out,
                             seed=42, resume=True)
        assert len(paths) == 3

    def test_resume_all_complete(self, tmp_path, monkeypatch):
        """Resume when all structures exist should skip everything."""
        from amorphgen.pipeline import random_gen
        out = str(tmp_path / "structures")
        comp = {"Si": 8, "O": 16}

        # First run
        batch_random(comp, n_structures=3, output_dir=out, seed=42)

        # Resume: everything exists
        def unexpected_generation(*args, **kwargs):
            pytest.fail("Completed structures must not be regenerated")

        monkeypatch.setattr(random_gen, "generate_random", unexpected_generation)
        paths = batch_random(comp, n_structures=3, output_dir=out,
                             seed=42, resume=True)
        assert len(paths) == 3

    def test_resume_composition_mismatch_fails_without_writes(self, tmp_path):
        """Equal atom counts must not let SiO2 outputs stand in for GeO2."""
        out = str(tmp_path / "structures")

        # First run with SiO2
        batch_random({"Si": 4, "O": 8}, n_structures=2,
                     output_dir=out, seed=42)
        before = {p: p.read_bytes() for p in (tmp_path / "structures").rglob("*")
                  if p.is_file()}

        # Resume with different composition
        with pytest.raises(ValueError, match="Cannot resume: composition changed"):
            batch_random({"Ge": 4, "O": 8}, n_structures=2,
                         output_dir=out, seed=42, resume=True)
        assert {p: p.read_bytes() for p in (tmp_path / "structures").rglob("*")
                if p.is_file()} == before

    @pytest.mark.parametrize("composition", [{"Ge": 4, "O": 8}, {"Si": 5, "O": 7}])
    def test_resume_checks_saved_composition(self, tmp_path, composition):
        """A readable checkpoint needs the correct elements and counts."""
        from ase import Atoms

        batch_random(composition, n_structures=0, output_dir=str(tmp_path))
        initial = tmp_path / "random_initial"
        path = initial / "random_0000.xyz"
        write(path, Atoms("Si4O8", cell=[8, 8, 8], pbc=True))
        before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
        with pytest.raises(ValueError, match="Cannot resume: composition of"):
            batch_random(composition, output_dir=str(tmp_path), resume=True)
        assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before

    def test_cli_resume_rejects_incompatible_formula(self, tmp_path):
        import subprocess
        import sys

        batch_random({"Si": 4, "O": 8}, n_structures=2,
                     output_dir=str(tmp_path), seed=42)
        before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
        result = subprocess.run(
            [sys.executable, "-m", "amorphgen.cli", "--random-gen",
             "--composition", "GeO2*4", "-n", "2", "--resume", "-o", str(tmp_path)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "composition changed" in result.stdout + result.stderr
        assert "already exist" not in result.stdout
        assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before

    @pytest.mark.parametrize("metadata", ["{", "[]", "{}"])
    def test_resume_refuses_invalid_metadata(self, tmp_path, metadata):
        path = tmp_path / "run_metadata.json"
        path.write_text(metadata)
        with pytest.raises(ValueError, match="Cannot resume: .*metadata"):
            batch_random({"Si": 4, "O": 8}, output_dir=str(tmp_path), resume=True)
        assert path.read_text() == metadata
        assert not (tmp_path / "random_gen.log").exists()

    @pytest.mark.parametrize("changed", [{"output_format": "cif"}, {"relax": True}])
    def test_resume_checks_other_stored_settings(self, tmp_path, changed):
        import json
        metadata = {"composition": {"Si": 4, "O": 8},
                    "output_format": "xyz", "relax": False}
        path = tmp_path / "run_metadata.json"
        path.write_text(json.dumps(metadata))
        before = path.read_bytes()
        with pytest.raises(ValueError, match=f"{next(iter(changed))} changed"):
            batch_random(metadata["composition"], output_dir=str(tmp_path),
                         resume=True, **changed)
        assert path.read_bytes() == before


class TestSeedReproducibility:
    """Per-structure seeds are derived from the structure index, so a resumed
    run reproduces exactly the structures a fresh run would generate."""

    def test_seed_helper_index_stable_and_retry_varying(self):
        from amorphgen.pipeline.random_gen import _derive_structure_seed
        # Deterministic for a fixed (base_seed, index, attempt)
        assert _derive_structure_seed(42, 3, 0) == _derive_structure_seed(42, 3, 0)
        # Distinct per index
        seeds = {_derive_structure_seed(42, i, 0) for i in range(6)}
        assert len(seeds) == 6
        # Retries of the same index draw different randomness
        assert _derive_structure_seed(42, 3, 0) != _derive_structure_seed(42, 3, 1)

    def test_resume_reproduces_fresh_structures(self, tmp_path):
        import shutil
        comp = {"Si": 8, "O": 16}
        fresh = str(tmp_path / "fresh")
        resumed = str(tmp_path / "resumed")

        batch_random(comp, n_structures=4, output_dir=fresh, seed=123)

        # Simulate an interruption: pre-seed the resume dir with the first
        # two structures from the fresh run, then resume.
        init = os.path.join(resumed, "random_initial")
        os.makedirs(init, exist_ok=True)
        for i in (0, 1):
            src = os.path.join(fresh, "random_initial", f"random_{i:04d}.xyz")
            write(os.path.join(init, f"random_{i:04d}.xyz"), read(src))
        shutil.copyfile(os.path.join(fresh, "run_metadata.json"),
                        os.path.join(resumed, "run_metadata.json"))
        batch_random(comp, n_structures=4, output_dir=resumed, seed=123,
                     resume=True)

        # Indices generated on resume (2, 3) must match the fresh run exactly.
        for i in (2, 3):
            a = read(os.path.join(fresh, "random_initial", f"random_{i:04d}.xyz"))
            b = read(os.path.join(resumed, "random_initial", f"random_{i:04d}.xyz"))
            assert a.get_chemical_symbols() == b.get_chemical_symbols()
            np.testing.assert_array_equal(a.get_positions(), b.get_positions())
            np.testing.assert_array_equal(a.cell, b.cell)


@pytest.mark.parametrize("changed, setting", [
    ({"seed": 43}, "generation.seed"),
    ({"target_density": 2.1}, "generation.target_density"),
    ({"density_scale": 0.9}, "generation.density_scale"),
    ({"cell_length_ang": 12.0}, "generation.cell_length_ang"),
    ({"minsep": {"Si-Si": 1.0}}, "generation.minsep"),
    ({"minsep_scale": 0.8}, "generation.minsep_scale"),
    ({"target_cn": {}}, "generation.target_cn"),
    ({"dmax": {"Si-O": 2.2}}, "generation.dmax"),
    ({"dmax_factor": 1.4}, "generation.dmax_factor"),
    ({"cn_tolerance": 2}, "generation.cn_tolerance"),
    ({"max_attempts_per_atom": 10}, "generation.max_attempts_per_atom"),
    ({"pbc": False}, "generation.pbc"),
    ({"repair_iters": 1}, "generation.repair_iters"),
    ({"min_cn": 1}, "generation.min_cn"),
    ({"repair_floor": False}, "generation.repair_floor"),
    ({"retry_mode": "none"}, "generation.retry_mode"),
    ({"max_retries": 3}, "max_retries"),
    ({"fmax": 0.01}, "relaxation.fmax"),
    ({"max_relax_steps": 10}, "relaxation.max_steps"),
    ({"optimizer": "BFGS"}, "relaxation.optimizer"),
    ({"cell_filter": "none"}, "relaxation.cell_filter"),
    ({"safety": {"min_distance": 0.7}}, "relaxation.safety.min_distance"),
    ({"repulsive_core": {"enabled": True}}, "relaxation.repulsive_core.enabled"),
])
def test_resume_refuses_changed_settings_before_writing(tmp_path, changed, setting):
    options = {"seed": 42, "n_structures": 0, "output_dir": str(tmp_path)}
    batch_random({"Si": 4, "O": 8}, **options)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match=f"Cannot resume: {setting}.*changed"):
        batch_random({"Si": 4, "O": 8}, **{**options, **changed}, resume=True)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("metadata", ["missing", "legacy", "incomplete"])
def test_resume_refuses_unverifiable_settings(tmp_path, metadata):
    import json

    batch_random({"Si": 4, "O": 8}, output_dir=str(tmp_path), seed=42)
    path = tmp_path / "run_metadata.json"
    if metadata == "missing":
        path.unlink()
    else:
        saved = json.loads(path.read_text())
        if metadata == "legacy":
            saved.pop("settings")
        else:
            saved["settings"]["generation"].pop("seed")
        path.write_text(json.dumps(saved))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Cannot resume:"):
        batch_random({"Si": 4, "O": 8}, output_dir=str(tmp_path), seed=42, resume=True)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_resume_normalizes_defaults_and_extxyz_alias(tmp_path):
    from amorphgen.pipeline.random_gen import _GENERATION_DEFAULTS
    from amorphgen.utils.safety import DEFAULT_SAFETY_CONFIG

    batch_random({"Si": 4, "O": 8}, output_dir=str(tmp_path), seed=42)
    paths = batch_random(
        {"Si": 4, "O": 8}, output_dir=str(tmp_path), resume=True,
        output_format="extxyz", safety=DEFAULT_SAFETY_CONFIG,
        repulsive_core={"enabled": False, "cutoff": 1.0, "strength": 1.0},
        **{**_GENERATION_DEFAULTS, "seed": 42},
    )
    assert len(paths) == 1


def test_resume_allows_extending_count_and_indices(tmp_path):
    options = {"seed": 42, "output_dir": str(tmp_path), "n_structures": 2}
    batch_random({"Si": 4, "O": 8}, **options, indices="0")
    first = tmp_path / "random_initial" / "random_0000.xyz"
    before = first.read_bytes()
    paths = batch_random({"Si": 4, "O": 8}, **{**options, "n_structures": 3},
                         indices="1-2", resume=True)
    assert len(paths) == 3
    assert first.read_bytes() == before


def test_fresh_partial_run_cannot_relabel_old_outputs(tmp_path):
    batch_random({"Si": 4, "O": 8}, output_dir=str(tmp_path),
                 n_structures=2, seed=42)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Cannot reuse output directory: generation.seed"):
        batch_random({"Si": 4, "O": 8}, output_dir=str(tmp_path),
                     n_structures=2, seed=43, indices="0")
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("change", ["parameters", "class", "weights", "model", "checkpoint",
                                    "head", "named_model"])
def test_resume_refuses_changed_calculator(tmp_path, change):
    from ase.calculators.emt import EMT
    from ase.calculators.lj import LennardJones

    calc = LennardJones(epsilon=1.0)
    if change in ("head", "named_model"):
        setattr(calc, "head" if change == "head" else "model", "original")
    config = {"model": "test-model"}
    checkpoint = tmp_path / "test.model"
    if change == "checkpoint":
        checkpoint.write_bytes(b"original weights")
        config["model_path"] = str(checkpoint)
    if change == "weights":
        class Model:
            def __init__(self, value):
                self.value = value

            def state_dict(self):
                return {"weights": np.array([self.value])}

        calc.models = [Model(1.0)]
    options = {"output_dir": str(tmp_path / "run"), "n_structures": 0,
               "relax": True, "calc": calc, "resume_settings": config}
    batch_random({"Cu": 4}, **options)
    if change == "parameters":
        options["calc"] = LennardJones(epsilon=2.0)
    elif change == "class":
        options["calc"] = EMT()
    elif change == "weights":
        calc.models = [Model(2.0)]
    elif change == "model":
        options["resume_settings"] = {"model": "another-model"}
    elif change in ("head", "named_model"):
        setattr(calc, "head" if change == "head" else "model", "changed")
    else:
        checkpoint.write_bytes(b"replacement weights")
    before = {p: p.read_bytes() for p in (tmp_path / "run").rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Cannot resume: relaxation.*changed"):
        batch_random({"Cu": 4}, **options, resume=True)
    assert {p: p.read_bytes() for p in (tmp_path / "run").rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("change", ["pair_params", "cutoff", "wrapper", "wrapped_parameters"])
def test_resume_checks_classical_and_wrapped_calculators(tmp_path, change):
    from amorphgen.utils.classical import LennardJonesCalculator
    from amorphgen.utils.repulsion import with_repulsive_core

    calc = LennardJonesCalculator({("Cu", "Cu"): {"epsilon": 1.0, "sigma": 2.0}})
    if change in ("wrapper", "wrapped_parameters"):
        calc = with_repulsive_core(calc, {"enabled": True, "cutoff": 1.0})
    options = {"calc": calc, "relax": True, "n_structures": 0,
               "output_dir": str(tmp_path)}
    batch_random({"Cu": 4}, **options)
    if change == "pair_params":
        calc.pair_params[("Cu", "Cu")]["epsilon"] = 2.0
    elif change == "cutoff":
        calc.cutoff = 11.0
    elif change == "wrapper":
        calc.repulsive_core_config["cutoff"] = 1.2
    else:
        calc.base_calculator.cutoff = 11.0
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Cannot resume: relaxation.calculator.*changed"):
        batch_random({"Cu": 4}, **options, resume=True)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_batch_refuses_locked_directory_without_writing(tmp_path):
    from amorphgen.utils.run_lock import run_lock

    batch_random({"Si": 4, "O": 8}, output_dir=str(tmp_path), n_structures=0)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="locked|another|active"):
            batch_random({"Si": 4, "O": 8}, output_dir=str(tmp_path), resume=True)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_cli_torchsim_keeps_lock_and_refuses_changed_model(tmp_path, monkeypatch):
    import sys
    from amorphgen.cli import main
    from amorphgen.utils.run_lock import run_lock

    optimized = []

    def optimize(**kwargs):
        with pytest.raises(RuntimeError, match="Another run"):
            with run_lock(tmp_path):
                pytest.fail("The random-generation lock must cover relaxation")
        optimized.append(kwargs)
        return []

    monkeypatch.setattr("amorphgen.pipeline.opt_cell.batch_optimize", optimize)
    args = ["amorphgen", "--random-gen", "--relax", "--engine", "torchsim",
            "--composition", "Cu=4", "--model", "lj", "--seed", "42",
            "-n", "1", "-o", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", args)
    main()
    assert len(optimized) == 1
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    args[args.index("lj")] = "buckingham"
    monkeypatch.setattr(sys, "argv", [*args, "--resume"])
    with pytest.raises(ValueError, match="Cannot resume: relaxation.*changed"):
        main()
    assert len(optimized) == 1
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_cli_refuses_changed_seed(tmp_path, monkeypatch):
    import sys
    from amorphgen.cli import main

    args = ["amorphgen", "--random-gen", "--composition", "Si=4",
            "--seed", "42", "-n", "1", "-o", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", args)
    main()
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    args[args.index("42")] = "43"
    monkeypatch.setattr(sys, "argv", [*args, "--resume"])
    with pytest.raises(ValueError, match="Cannot resume: generation.seed changed"):
        main()
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_batch_path_keeps_auto_cn_and_tolerance(tmp_path, monkeypatch):
    """batch_random must forward the automatic CN targets and tolerance."""
    import amorphgen.pipeline.random_gen as rg
    from ase import Atoms
    captured = {}
    comp = {"Cu": 8, "Zr": 8}                        # alloy: auto tolerance is 2

    def fake_generate(composition, seed=None, **kwargs):
        captured.update(kwargs)
        n = sum(composition.values())
        return Atoms("Cu8Zr8", positions=np.random.default_rng(0).uniform(0, 8, (n, 3)),
                     cell=[8, 8, 8], pbc=True)
    monkeypatch.setattr(rg, "generate_random", fake_generate)
    rg.batch_random(comp, n_structures=1, output_dir=str(tmp_path), output_format="xyz")
    target_cn, auto_tol = rg._auto_target_cn(comp)
    assert captured.get("target_cn") == target_cn
    assert captured.get("cn_tolerance") == auto_tol and auto_tol > 0


class TestIndexSelection:
    def test_parse_index_spec(self):
        from amorphgen.utils.common import parse_index_spec
        assert parse_index_spec("80-90") == set(range(80, 91))
        assert parse_index_spec("0,5,7-9") == {0, 5, 7, 8, 9}
        assert parse_index_spec([1, 2]) == {1, 2}
        assert parse_index_spec("2-3", n_total=3) == {2}
        with pytest.raises(ValueError):
            parse_index_spec("9-2")

    def test_random_gen_indices_match_full_run(self, tmp_path):
        from amorphgen.pipeline.random_gen import batch_random
        comp = {"Si": 8, "O": 16}
        batch_random(comp, n_structures=6, output_dir=str(tmp_path / "full"), seed=7)
        batch_random(comp, n_structures=6, output_dir=str(tmp_path / "part"), seed=7, indices="2-3")
        made = sorted(p.name for p in (tmp_path / "part" / "random_initial").glob("*.xyz"))
        assert made == ["random_0002.xyz", "random_0003.xyz"]
        for k in (2, 3):
            a = read(str(tmp_path / "part" / "random_initial" / f"random_{k:04d}.xyz"))
            b = read(str(tmp_path / "full" / "random_initial" / f"random_{k:04d}.xyz"))
            assert np.allclose(a.positions, b.positions)

    def test_batch_optimize_indices_filter(self, tmp_path, monkeypatch, capsys):
        from amorphgen.pipeline import opt_cell
        src = tmp_path / "in"; src.mkdir()
        from ase.build import bulk
        for k in range(6):
            write(str(src / f"random_{k:04d}.xyz"), bulk("Cu", "fcc", a=3.6, cubic=True), format="extxyz")
        seen = []
        monkeypatch.setattr(opt_cell, "run", lambda path, **kw: seen.append(os.path.basename(path)) or read(path))
        opt_cell.batch_optimize(str(src), str(tmp_path / "o"), cfg_override={}, indices="1,4-5")
        assert seen == ["random_0001.xyz", "random_0004.xyz", "random_0005.xyz"]
        assert "index selection 1,4-5: 3 of 6 files" in capsys.readouterr().out
