"""
test/test_external_potentials.py
--------------------------------
ACE (pyace) and LAMMPS (LAMMPSlib) backends: model resolution, the
lammps_params setup, CLI wiring and provenance run without either package;
the classes at the end exercise the real backends when installed.
"""

import copy
import json
import os
import subprocess
import sys
from importlib.resources import as_file, files
from unittest.mock import patch

import numpy as np
import pytest
from ase.build import bulk

import amorphgen.utils.calculators as calc_mod
from amorphgen.utils.calculators import backend_for, get_calculator
from amorphgen.utils.lammps_potential import lammps_setup

# Stillinger-Weber silicon (Phys. Rev. B 31, 5262), in LAMMPS's sw format.
SI_SW = ("Si Si Si 2.1683 2.0951 1.80 21.0 1.20 -0.333333333333 "
         "7.049556277 0.6022245584 4.0 0.0 0.0\n")
SW_PARAMS = {"pair_style": "sw", "pair_coeff": "* * Si.sw Si"}


@pytest.fixture
def sw_dir(tmp_path, monkeypatch):
    """Run from a directory holding Si.sw, as a user would."""
    (tmp_path / "Si.sw").write_text(SI_SW)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ═════════════════════════════════════════════════════════════════════════════
# Model resolution and routing (no backend packages needed)
# ═════════════════════════════════════════════════════════════════════════════

class TestBackendFor:

    @pytest.mark.parametrize("model, path, expected", [
        ("ace", "pot.yace", "ace"),
        ("ACE", "pot.yaml", "ace"),
        ("mace-mpa-0", "pot.yaml", "ace"),      # the suffix implies ACE
        ("mace-mpa-0", "pot.ace", "ace"),
        ("mace-mpa-0", "my.model", "mace"),
        ("chgnet", "my.model", "mace"),         # a path still wins over the name
        ("lammps", None, "lammps"),
        ("LAMMPS", None, "lammps"),
        ("chgnet", None, "chgnet"),
    ])
    def test_resolution(self, model, path, expected):
        assert backend_for(model, path) == expected

    def test_ace_needs_a_file(self):
        with pytest.raises(ValueError, match="--model-path"):
            backend_for("ace")

    def test_lammps_takes_no_model_path(self):
        with pytest.raises(ValueError, match="pair_coeff"):
            backend_for("lammps", "my.model")


class TestRouting:

    def test_ace_file_routes_to_ace_without_dtype(self):
        with patch.object(calc_mod, "_load_ace") as load:
            get_calculator(model_path="pot.yace", device="cpu")
        load.assert_called_once_with("pot.yace", device="cpu")

    def test_explicit_dtype_reaches_the_loader_for_its_note(self):
        with patch.object(calc_mod, "_load_ace") as load:
            get_calculator("ace", device="cpu", model_path="pot.yaml",
                           default_dtype="float32")
        load.assert_called_once_with("pot.yaml", device="cpu", default_dtype="float32")

    def test_lammps_routes_with_its_params(self):
        with patch.object(calc_mod, "_load_lammps") as load:
            get_calculator("lammps", device="cpu", lammps_params=SW_PARAMS,
                           classical_params=None)
        load.assert_called_once_with(device="cpu", lammps_params=SW_PARAMS)

    @pytest.mark.parametrize("model, kwargs", [
        ("mace-mpa-0", {"lammps_params": SW_PARAMS}),
        ("lammps", {"classical_params": {"params": {"Si-Si": {"sigma": 2.0, "epsilon": 0.1}}}}),
        ("buckingham", {"ace_params": {"recursive": True}}),
    ])
    def test_params_for_another_backend_are_refused(self, model, kwargs):
        with pytest.raises(ValueError, match="is for"):
            get_calculator(model, device="cpu", **kwargs)

    def test_ace_loader_reports_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ACE potential file not found"):
            get_calculator(model_path=str(tmp_path / "none.yace"), device="cpu")


# ═════════════════════════════════════════════════════════════════════════════
# lammps_setup
# ═════════════════════════════════════════════════════════════════════════════

class TestLammpsSetup:

    def test_elements_and_absolute_file(self, sw_dir):
        setup = lammps_setup(SW_PARAMS)
        sw = str(sw_dir / "Si.sw")
        assert setup["elements"] == ["Si"]
        assert setup["atom_types"] == {"Si": 1}
        assert setup["files"] == [sw]
        assert setup["lmpcmds"] == ["pair_style sw", f"pair_coeff * * {sw} Si"]

    def test_element_and_style_tokens_are_never_paths(self, sw_dir):
        # Files named like an element or the style must not be rewritten.
        (sw_dir / "Si").write_text("")
        (sw_dir / "sw").write_text("")
        setup = lammps_setup(SW_PARAMS)
        assert setup["lmpcmds"][0] == "pair_style sw"
        assert setup["lmpcmds"][1].split()[-1] == "Si"

    def test_command_names_are_optional(self, sw_dir):
        setup = lammps_setup({"pair_style": "pair_style sw",
                              "pair_coeff": ["pair_coeff * * Si.sw Si"]})
        assert setup["lmpcmds"][0] == "pair_style sw"
        assert setup["lmpcmds"][1].startswith("pair_coeff * * /")

    def test_hybrid_lines_merge_by_position(self):
        setup = lammps_setup({
            "pair_style": "hybrid/overlay sw tersoff",
            "pair_coeff": ["* * sw a.sw Si NULL", "* * tersoff b.tersoff NULL C"],
        })
        assert setup["elements"] == ["Si", "C"]

    @pytest.mark.parametrize("coeffs, match", [
        (["* * a Si O", "* * b O Si"], "disagree"),
        (["* * a Si NULL"], "NULL in every"),
        (["* * a Si O", "* * b Si"], "different numbers"),
        (["1 1 0.01 2.3"], "type order"),
    ])
    def test_unusable_type_orders(self, coeffs, match):
        with pytest.raises(ValueError, match=match):
            lammps_setup({"pair_style": "x", "pair_coeff": coeffs})

    def test_numeric_coeffs_take_explicit_elements(self):
        setup = lammps_setup({"pair_style": "lj/cut 6.0", "pair_coeff": "1 1 0.01 2.3",
                              "elements": "Cu"})
        assert setup["atom_types"] == {"Cu": 1}

    def test_explicit_elements_must_match_the_coeffs(self):
        with pytest.raises(ValueError, match="disagrees"):
            lammps_setup({"pair_style": "x", "pair_coeff": "* * f Si O",
                          "elements": ["O", "Si"]})
        setup = lammps_setup({"pair_style": "x", "pair_coeff": "* * f Si O",
                              "elements": ["Si", "O"]})
        assert setup["atom_types"] == {"Si": 1, "O": 2}

    @pytest.mark.parametrize("params, match", [
        (None, "needs a pair style"),
        ({"pair_coeff": "* * f Si"}, "pair_style"),
        ({"pair_style": "sw"}, "pair_coeff is required"),
        ({"pair_style": "sw", "pair_coeff": "* * f Si", "pair_coef": "x"}, "Unknown"),
        ({"pair_style": "sw", "pair_coeff": "* * f Si", "masses": {"O": 16.0}}, "masses"),
        ({"pair_style": "sw", "pair_coeff": "* * f Si", "elements": ["Xx"]}, "element symbols"),
    ])
    def test_invalid_params(self, params, match):
        with pytest.raises(ValueError, match=match):
            lammps_setup(params)

    def test_commands_and_log_file(self, sw_dir):
        setup = lammps_setup({**SW_PARAMS, "commands": ["pair_modify shift yes"],
                              "log_file": "lammps.log"})
        assert setup["lmpcmds"][-1] == "pair_modify shift yes"
        assert setup["log_file"] == str(sw_dir / "lammps.log")


# ═════════════════════════════════════════════════════════════════════════════
# Availability, fail-fast checks and --list-models
# ═════════════════════════════════════════════════════════════════════════════

class TestChecks:

    @pytest.mark.parametrize("model, path, extra, note", [
        ("ace", "pot.yace", "ace", "python-ace"),
        ("lammps", None, "lammps", "pair_style pace"),
    ])
    def test_install_hint(self, monkeypatch, model, path, extra, note):
        monkeypatch.setattr(calc_mod, "backend_available", lambda b: b == "classical")
        with pytest.raises(calc_mod.BackendNotInstalledError) as exc:
            calc_mod.require_backend(model, model_path=path)
        assert f'pip install "amorphgen[{extra}]"' in str(exc.value)
        assert note in str(exc.value)

    def test_require_potential(self, tmp_path):
        from amorphgen.utils.calculators import require_potential
        require_potential("mace-mpa-0")                      # other backends pass
        with pytest.raises(FileNotFoundError):
            require_potential("ace", str(tmp_path / "none.yace"))
        with pytest.raises(ValueError, match="needs a pair style"):
            require_potential("lammps")
        pot = tmp_path / "pot.yace"
        pot.write_text("")
        with pytest.raises(ValueError, match="torch-sim"):
            require_potential("mace-mpa-0", str(pot), engine="torchsim")

    @pytest.mark.parametrize("model, path", [
        ("ace", None), ("lammps", None), ("mace-mpa-0", "pot.yace")])
    def test_torchsim_rejects(self, model, path):
        from amorphgen.utils.torchsim_engine import check_model
        with pytest.raises(ValueError, match="no torch-sim implementation"):
            check_model(model, path)

    def test_torchsim_accepts_its_models(self):
        from amorphgen.utils.torchsim_engine import check_model
        check_model("mace-mpa-0")
        check_model("lj")
        check_model("mace-mpa-0", "my.model")

    def test_list_models_sections(self, capsys):
        calc_mod.list_models()
        out = capsys.readouterr().out
        assert "ACE (pacemaker potential file)" in out
        assert "LAMMPS (any pair style)" in out


# ═════════════════════════════════════════════════════════════════════════════
# YAML config
# ═════════════════════════════════════════════════════════════════════════════

class TestYaml:

    def test_blocks_are_accepted(self, tmp_path):
        from amorphgen.configs import load_yaml_config
        path = tmp_path / "c.yaml"
        path.write_text(
            "model: lammps\n"
            "lammps_params:\n"
            "  pair_style: tersoff\n"
            "  pair_coeff: ['* * SiC.tersoff Si C']\n"
            "  elements: [Si, C]\n"
            "  masses: {Si: 28.0855}\n"
            "  commands: pair_modify shift yes\n"
            "ace_params:\n"
            "  recursive_evaluator: true\n")
        cfg = load_yaml_config(str(path))
        assert cfg["lammps_params"]["elements"] == ["Si", "C"]

    @pytest.mark.parametrize("text", [
        "lammps_params:\n  pair_styl: sw\n",
        "lammps_params:\n  masses: {Xx: 1.0}\n",
        "ace_params:\n  active_set: pot.asi\n",
    ])
    def test_unknown_keys_are_rejected(self, tmp_path, text):
        from amorphgen.configs import load_yaml_config
        path = tmp_path / "c.yaml"
        path.write_text(text)
        with pytest.raises(ValueError):
            load_yaml_config(str(path))

    def test_bundled_lammps_example(self):
        from amorphgen.configs import load_yaml_config
        with as_file(files("amorphgen.configs").joinpath("example_lammps.yaml")) as path:
            cfg = load_yaml_config(str(path))
        assert cfg["model"] == "lammps"
        lammps_setup(cfg["lammps_params"])


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

class TestCli:

    def _override(self, argv):
        from amorphgen.cli import _build_override, _get_parser
        parser = _get_parser()
        return _build_override(parser.parse_args(argv), parser, explicit_only=True,
                               argv=argv)

    def test_flags_build_lammps_params(self):
        ov = self._override(["--pair-style", "sw", "--pair-coeff", "* * Si.sw Si",
                             "--pair-coeff", "* * x", "--lammps-elements", "Si"])
        assert ov["lammps_params"] == {"pair_style": "sw",
                                       "pair_coeff": ["* * Si.sw Si", "* * x"],
                                       "elements": "Si"}
        assert "lammps_params" not in self._override([])

    def test_model_and_model_path_together(self):
        ov = self._override(["-m", "ace", "--model-path", "pot.yace"])
        assert ov["model"] == "ace" and ov["model_path"] == "pot.yace"

    @pytest.mark.parametrize("override, model", [
        ({"model_path": "pot.yace"}, "ace"),
        ({"lammps_params": SW_PARAMS}, "lammps"),
        ({"model": "LAMMPS", "lammps_params": SW_PARAMS}, "LAMMPS"),
        ({"model_path": "my.model"}, None),
    ])
    def test_model_inference(self, override, model):
        from amorphgen.cli import _infer_potential_model
        assert _infer_potential_model(override) is None
        assert override.get("model") == model

    @pytest.mark.parametrize("override", [
        {"model": "mace-mpa-0", "model_path": "pot.yace"},
        {"model": "chgnet", "lammps_params": SW_PARAMS},
    ])
    def test_conflicting_model_is_an_error(self, override):
        from amorphgen.cli import _infer_potential_model
        assert _infer_potential_model(override)

    def _main(self, monkeypatch, tmp_path, argv):
        from amorphgen.cli import main
        monkeypatch.setattr(calc_mod, "backend_available", lambda b: True)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["amorphgen", *argv, "-o", "out"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert not (tmp_path / "out").exists()          # no setup work done

    @pytest.mark.parametrize("argv, message", [
        (["in.xyz", "--model-path", "missing.yace"], "ACE potential file not found"),
        (["in.xyz", "-m", "chgnet", "--model-path", "pot.yace"], "is an ACE potential"),
        (["in.xyz", "--pair-style", "lj/cut 5", "--pair-coeff", "1 1 0.1 2.0"], "type order"),
        (["in.xyz", "-m", "lammps"], "needs a pair style"),
        (["--batch-opt", "--input-dir", "in", "--engine", "torchsim",
          "--model-path", "pot.yace"],
         "torch-sim"),
    ])
    def test_fail_fast(self, monkeypatch, tmp_path, capsys, argv, message):
        (tmp_path / "pot.yace").write_text("")
        self._main(monkeypatch, tmp_path, argv)
        assert message in capsys.readouterr().out


# ═════════════════════════════════════════════════════════════════════════════
# Provenance
# ═════════════════════════════════════════════════════════════════════════════

class TestProvenance:

    def test_ace_file_is_named_and_hashed(self, tmp_path):
        from amorphgen.utils.run_provenance import calculator_provenance
        pot = tmp_path / "pot.yace"
        pot.write_text("elements: [Si]\n")
        info = calculator_provenance({"model": "mace-mpa-0", "model_path": str(pot)})
        assert info["model"]["name"] == "ace"
        assert info["model"]["hash_source"] == "file"
        assert info["model"]["path"] == str(pot)
        assert info["precision"]["resolved"] == "float64"
        assert info["device"]["resolved"] == "cpu"

    def test_lammps_files_are_hashed(self, sw_dir):
        from amorphgen.utils.run_provenance import calculator_provenance
        cfg = {"model": "lammps", "lammps_params": SW_PARAMS}
        first = calculator_provenance(cfg)["model"]
        assert first["hash_source"] == "lammps-v1"
        assert list(first["files"]) == [str(sw_dir / "Si.sw")]
        (sw_dir / "Si.sw").write_text(SI_SW.replace("21.0", "21.5"))
        second = calculator_provenance(cfg)["model"]
        assert second["sha256"] != first["sha256"]
        json.dumps(second)


# ═════════════════════════════════════════════════════════════════════════════
# Real LAMMPS (pip install "amorphgen[lammps]")
# ═════════════════════════════════════════════════════════════════════════════

LJ_PAIRS = {("Cu", "Cu"): (0.40, 2.30), ("Ag", "Ag"): (0.35, 2.60),
            ("Cu", "Ag"): (0.37, 2.45)}


def _cu_ag():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
    atoms.symbols[::3] = "Ag"
    atoms.rattle(0.05, seed=2)
    return atoms


@pytest.fixture
def lj_lammps():
    pytest.importorskip("lammps")
    return get_calculator("lammps", device="cpu", lammps_params={
        "pair_style": "lj/cut 6.0",
        "pair_coeff": ["1 1 0.40 2.30", "2 2 0.35 2.60", "1 2 0.37 2.45"],
        "elements": ["Cu", "Ag"],
    })


class TestRealLammps:

    def test_matches_builtin_lennard_jones(self, lj_lammps):
        from ase.calculators.fd import calculate_numerical_stress
        from amorphgen.utils.classical import LennardJonesCalculator
        ref = LennardJonesCalculator(
            {pair: {"epsilon": e, "sigma": s} for pair, (e, s) in LJ_PAIRS.items()},
            cutoff=6.0)
        atoms, expected = _cu_ag(), _cu_ag()
        atoms.calc, expected.calc = lj_lammps, ref
        assert atoms.get_potential_energy() == pytest.approx(
            expected.get_potential_energy(), abs=1e-9)
        np.testing.assert_allclose(atoms.get_forces(), expected.get_forces(), atol=1e-9)
        np.testing.assert_allclose(atoms.get_stress(),
                                   calculate_numerical_stress(atoms), atol=1e-6)

    def test_type_map_does_not_follow_atom_order(self, lj_lammps):
        # LAMMPSlib's default numbers types by first appearance; reversing
        # the atoms would then swap the Cu and Ag parameters.
        atoms = _cu_ag()
        atoms.calc = lj_lammps
        energy = atoms.get_potential_energy()
        reverse = atoms[::-1]
        reverse.calc = lj_lammps
        assert reverse.get_potential_energy() == pytest.approx(energy, abs=1e-9)

    def test_shared_across_structures_and_deep_copies(self, lj_lammps):
        atoms = _cu_ag()
        atoms.calc = lj_lammps
        atoms.get_potential_energy()
        small = atoms[:10]
        small.calc = lj_lammps
        assert np.isfinite(small.get_potential_energy())
        assert copy.deepcopy(atoms).calc is lj_lammps

    def test_relative_potential_survives_chdir(self, sw_dir):
        pytest.importorskip("lammps")
        calc = get_calculator("lammps", device="cpu", lammps_params=SW_PARAMS)
        os.chdir(sw_dir.parent)       # LAMMPS starts here, on the first call
        si = bulk("Si", "diamond", a=5.431, cubic=True)
        si.calc = calc
        # Stillinger-Weber's diamond-Si cohesive energy.
        assert si.get_potential_energy() / len(si) == pytest.approx(-4.3366, abs=1e-3)
        carbon = bulk("C", "diamond", a=3.57)
        carbon.calc = calc
        with pytest.raises(ValueError, match="covers Si, not C"):
            carbon.get_potential_energy()

    def test_pipeline_stages(self, sw_dir):
        pytest.importorskip("lammps")
        from amorphgen.pipeline.run_pipeline import MeltQuenchPipeline
        from ase.io import write
        si = bulk("Si", "diamond", a=5.431, cubic=True).repeat(2)
        si.rattle(0.05, seed=1)
        write("si.xyz", si)
        md = {"T": 600, "steps": 10, "timestep": 1.0}
        cfg = {"model": "lammps", "lammps_params": SW_PARAMS, "device": "cpu",
               "opt": {"max_steps": 20, "fmax": 0.1, "cell_filter": "none"},
               "eq_premelt": {"ensemble": "NVT", **md},
               "eq_high": {"ensemble": "NPT", **md},
               "final_opt": {"max_steps": 20, "fmax": 0.1}}
        pipe = MeltQuenchPipeline(input_file="si.xyz", work_dir="run", cfg_override=cfg)
        pipe.run(stages=[1, 2, 4, 7])
        manifest = json.loads((sw_dir / "run" / "run_manifest.json").read_text())
        assert '"lammps-v1"' in json.dumps(manifest)


# ═════════════════════════════════════════════════════════════════════════════
# Real ACE (pip install "amorphgen[ace]")
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def ace_files(tmp_path_factory):
    """A small random-coefficient Si-O ACE potential as .yaml and .yace."""
    pyace = pytest.importorskip("pyace")
    out = tmp_path_factory.mktemp("ace")
    conf = pyace.create_multispecies_basis_config({
        "deltaSplineBins": 0.001,
        "elements": ["Si", "O"],
        "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled",
                               "fs_parameters": [1, 1, 1, 0.5], "ndensity": 2}},
        "bonds": {"ALL": {"radbase": "ChebExpCos", "radparameters": [5.25],
                          "rcut": 4.0, "dcut": 0.01}},
        "functions": {"ALL": {"nradmax_by_orders": [5, 2], "lmax_by_orders": [0, 1]}},
    }, func_coefs_initializer="random")
    conf.save(str(out / "pot.yaml"))
    pyace.ACEBBasisSet(conf).to_ACECTildeBasisSet().save_yaml(str(out / "pot.yace"))
    return out


def _si_o():
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.symbols[[1, 3]] = "O"
    atoms.rattle(0.1, seed=1)
    return atoms


class TestRealAce:

    @pytest.mark.parametrize("name", ["pot.yaml", "pot.yace"])
    def test_forces_and_stress_are_consistent(self, ace_files, name):
        from ase.calculators.fd import (calculate_numerical_forces,
                                        calculate_numerical_stress)
        atoms = _si_o()
        atoms.calc = get_calculator(model_path=str(ace_files / name), device="cpu")
        assert np.isfinite(atoms.get_potential_energy())
        np.testing.assert_allclose(atoms.get_forces(),
                                   calculate_numerical_forces(atoms), atol=1e-8)
        stress = atoms.get_stress()
        assert np.abs(stress).max() > 0
        np.testing.assert_allclose(stress, calculate_numerical_stress(atoms),
                                   atol=1e-8)

    def test_both_formats_agree_and_deep_copy(self, ace_files):
        energies = []
        for name in ("pot.yaml", "pot.yace"):
            atoms = _si_o()
            calc = get_calculator("ace", model_path=str(ace_files / name), device="cpu")
            atoms.calc = calc
            energies.append(atoms.get_potential_energy())
            assert copy.deepcopy(atoms).calc is calc
        assert energies[0] == pytest.approx(energies[1], abs=1e-10)

    def test_stages_share_the_calculator(self, ace_files, tmp_path, monkeypatch):
        from amorphgen.pipeline import equilibrate, opt_cell
        monkeypatch.chdir(tmp_path)
        cfg = {"model": "ace", "model_path": str(ace_files / "pot.yaml"), "device": "cpu",
               "opt": {"max_steps": 5, "fmax": 0.1, "cell_filter": "none"},
               "eq_premelt": {"ensemble": "NVT", "T": 300, "steps": 5, "timestep": 1.0}}
        atoms = opt_cell.run(_si_o(), cfg_override=cfg)
        atoms = equilibrate.run(atoms, cfg_override=cfg, stage="premelt")
        assert np.isfinite(atoms.get_potential_energy())

    def test_import_leaves_logging_alone(self, ace_files):
        script = ("import logging\n"
                  "from amorphgen.utils.ace_potential import ACECalculator\n"
                  "root = logging.getLogger()\n"
                  "assert not root.handlers and root.level == logging.WARNING\n")
        subprocess.run([sys.executable, "-c", script], check=True)
