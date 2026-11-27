"""
test/test_accelerator.py
------------------------
--accelerator: settings, package selection, option checks, the model store
(conversion, reuse, locking metadata), the check against MACE, routing in
get_calculator, the CLI gate and YAML. Neither Symmetrix package, mace-torch
nor torch is needed: the converter and calculators are stand-ins.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import amorphgen.utils.accelerator as acc
import amorphgen.utils.calculators as calc_mod
from amorphgen.utils.accelerator import (
    AcceleratorError, AcceleratorSettings, check_options, prepare,
    require_accelerator, resolve_flavour, stored_name,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "store"
    monkeypatch.setenv("AMORPHGEN_MODEL_STORE", str(path))
    return path


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "my_finetuned.model"
    path.write_bytes(b"fake mace weights v1")
    return path


@pytest.fixture
def installed(monkeypatch):
    def set_(**flavours):
        monkeypatch.setattr(acc, "installed_flavours",
                            lambda: {k.replace("_", "-"): v for k, v in flavours.items()})
    return set_


@pytest.fixture
def converter(monkeypatch):
    """Replace the converter subprocess; record each call."""
    calls = []

    def fake_run(command, capture_output, text):
        calls.append(command)
        out = Path(command[command.index("--output") + 1])
        out.write_text(json.dumps({"converted_from": command[command.index("--model") + 1]}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


class TestSettings:

    @pytest.mark.parametrize("value, engine", [
        (None, "none"), ("symmetrix-xl", "symmetrix-xl"), ("AUTO", "auto"),
        ({"engine": "Symmetrix-XL"}, "symmetrix-xl"), ({"options": {"metal": True}}, "none"),
    ])
    def test_from_config(self, value, engine):
        assert AcceleratorSettings.from_config(value).engine == engine

    def test_defaults(self):
        s = AcceleratorSettings.from_config("auto")
        assert (s.spline_points, s.validate, s.convert, s.tolerance) == (1024, True, True, 1.0)

    def test_elements_sorted_by_atomic_number(self):
        s = AcceleratorSettings.from_config({"engine": "auto", "elements": ["O", "In", "Ga", "O"]})
        assert s.elements == ["O", "Ga", "In"]

    @pytest.mark.parametrize("value, match", [
        ("fast", "must be one of"), ("symmetrix", "must be one of"),
        ({"engine": "auto", "bogus": 1}, "Unknown accelerator key"),
        ({"engine": "auto", "elements": ["Xx"]}, "Unknown element"), (3, "name or a mapping"),
    ])
    def test_invalid(self, value, match):
        with pytest.raises(AcceleratorError, match=match):
            AcceleratorSettings.from_config(value)


class TestResolveFlavour:

    def test_none(self, installed):
        installed(symmetrix_xl="0.2.0")
        assert resolve_flavour("none") is None

    def test_auto_uses_symmetrix_xl(self, installed):
        installed(symmetrix_xl="0.2.0")
        assert resolve_flavour("auto") == ("symmetrix-xl", "0.2.0")

    def test_original_symmetrix_is_not_used(self, installed, capsys):
        """The original package (same import name) is not supported in this version."""
        installed(symmetrix="0.3.1")
        assert resolve_flavour("auto") is None
        out = capsys.readouterr().out
        assert "original Symmetrix (0.3.1)" in out and "using plain MACE" in out
        with pytest.raises(AcceleratorError) as exc:
            resolve_flavour("symmetrix-xl")
        assert "original Symmetrix (0.3.1)" in str(exc.value)
        assert "separate environment" in str(exc.value)

    def test_auto_without_any_is_plain_mace(self, installed, capsys):
        installed()
        assert resolve_flavour("auto") is None
        assert "using plain MACE" in capsys.readouterr().out

    def test_explicit_match(self, installed):
        installed(symmetrix_xl="0.2.0")
        assert resolve_flavour("symmetrix-xl") == ("symmetrix-xl", "0.2.0")

    def test_explicit_none_installed(self, installed):
        installed()
        with pytest.raises(AcceleratorError, match="pip install"):
            resolve_flavour("symmetrix-xl")

    def test_both_installed(self, installed):
        installed(symmetrix="0.3.1", symmetrix_xl="0.2.0")
        with pytest.raises(AcceleratorError, match="Both"):
            resolve_flavour("auto")


class TestOptions:

    def test_valid(self):
        check_options("symmetrix-xl", {"execution_profile": "speed"}, "float64")

    def test_option_of_the_original_package(self):
        with pytest.raises(AcceleratorError, match="is for the original Symmetrix"):
            check_options("symmetrix-xl", {"use_kokkos": False}, "float64")

    def test_unknown_option(self):
        with pytest.raises(AcceleratorError, match="accepted"):
            check_options("symmetrix-xl", {"turbo": True}, "float64")

    def test_metal_needs_float32(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        with pytest.raises(AcceleratorError, match="float32"):
            check_options("symmetrix-xl", {"metal": True}, "float64")
        check_options("symmetrix-xl", {"metal": True}, "float32")

    def test_metal_needs_macos(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        with pytest.raises(AcceleratorError, match="macOS"):
            check_options("symmetrix-xl", {"metal": True}, "float32")


class TestStore:

    def test_stored_name(self):
        assert stored_name("mace-mpa-0", "abcdef0123456789", 1024, None) \
            == "mace-mpa-0__abcdef012345__1024.json"
        assert stored_name("m", "abcdef0123456789", 256, "omat_pbe") \
            == "m__abcdef012345__256__omat_pbe.json"

    def test_default_store_location(self, monkeypatch):
        monkeypatch.delenv("AMORPHGEN_MODEL_STORE", raising=False)
        assert acc.model_store() == Path("~/.amorphgen/models").expanduser()

    def test_convert_then_reuse(self, store, model_file, converter, capsys):
        path = prepare(model_path=str(model_file), flavour="symmetrix-xl", version="0.2.0")
        assert path.parent == store / "symmetrix-xl-0.2.0"
        assert path.name.startswith("my_finetuned__") and path.name.endswith("__1024.json")
        assert len(converter) == 1
        command = converter[0]
        assert command[0] == sys.executable
        assert "--chemical-symbols" not in command          # XL: universal export
        assert command[command.index("--num-spline-points") + 1] == "1024"
        index = json.loads((store / "index.json").read_text())
        entry = index[f"symmetrix-xl-0.2.0/{path.name}"]
        assert entry["flavour"] == "symmetrix-xl" and entry["source"] == str(model_file.resolve())
        again = prepare(model_path=str(model_file), flavour="symmetrix-xl", version="0.2.0")
        assert again == path and len(converter) == 1        # reused
        assert "Using stored model" in capsys.readouterr().out

    def test_changed_model_is_converted_again(self, store, model_file, converter):
        first = prepare(model_path=str(model_file), flavour="symmetrix-xl", version="0.2.0")
        model_file.write_bytes(b"fake mace weights v2 (retrained)")
        second = prepare(model_path=str(model_file), flavour="symmetrix-xl", version="0.2.0")
        assert first != second and len(converter) == 2

    @pytest.mark.parametrize("change", [
        {"version": "0.3.0"}, {"spline_points": 4096}, {"model_suffix": ":dft"},
    ])
    def test_key_parts(self, store, model_file, converter, change):
        base = dict(flavour="symmetrix-xl", version="0.2.0")
        first = prepare(model_path=str(model_file), **base)
        model = "x" + change.pop("model_suffix", "")
        second = prepare(model, model_path=str(model_file), **{**base, **change})
        assert first != second

    def test_export_holds_every_element(self, store, model_file, converter):
        """No element list goes to the converter: one file serves every system."""
        prepare(model_path=str(model_file), flavour="symmetrix-xl", version="0.2.0")
        command = converter[-1]
        assert "--chemical-symbols" not in command and "--atomic-numbers" not in command

    def test_head_passed_to_converter(self, store, model_file, converter):
        prepare("mace-mh-1:omat_pbe", model_path=str(model_file),
                flavour="symmetrix-xl", version="0.2.0")
        assert converter[0][converter[0].index("--head") + 1] == "omat_pbe"

    def test_convert_disabled(self, store, model_file, converter):
        with pytest.raises(AcceleratorError, match="conversion is disabled"):
            prepare(model_path=str(model_file), flavour="symmetrix-xl", version="0.2.0",
                    convert=False)
        assert converter == []

    def test_force(self, store, model_file, converter):
        prepare(model_path=str(model_file), flavour="symmetrix-xl", version="0.2.0")
        prepare(model_path=str(model_file), flavour="symmetrix-xl", version="0.2.0", force=True)
        assert len(converter) == 2

    def test_converter_failure(self, store, model_file, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
            returncode=1, stdout="", stderr="Traceback ...\nValueError: unsupported readout"))
        with pytest.raises(AcceleratorError, match="unsupported readout"):
            prepare(model_path=str(model_file), flavour="symmetrix-xl", version="0.2.0")
        assert not list((store / "symmetrix-xl-0.2.0").glob("*.json"))

    def test_missing_model_file(self, store, tmp_path):
        with pytest.raises(FileNotFoundError):
            prepare(model_path=str(tmp_path / "none.model"), flavour="symmetrix-xl",
                    version="0.2.0")


class TestValidate:

    @pytest.fixture
    def fake_calcs(self, monkeypatch):
        from ase.calculators.calculator import Calculator, all_changes
        import numpy as np

        class Simple(Calculator):
            implemented_properties = ["energy", "forces", "stress"]

            def __init__(self, offset=0.0):
                super().__init__()
                self.offset = offset

            def calculate(self, atoms=None, properties=None, system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                pos = self.atoms.get_positions()
                self.results = {"energy": float(pos.sum()),
                                "forces": pos * 0.01 + self.offset,
                                "stress": np.zeros(6)}
        monkeypatch.setattr(calc_mod, "_load_mace", lambda *a, **k: Simple())
        return Simple

    def test_pass_and_recorded_once(self, store, tmp_path, fake_calcs, capsys, monkeypatch):
        json_path = store / "symmetrix-xl-0.2.0" / "m__abc__1024.json"
        json_path.parent.mkdir(parents=True)
        json_path.write_text("{}")
        result = acc.validate(json_path, fake_calcs(), model="mace-mpa-0", model_path=None,
                              elements=["O", "Si"], dtype="float64", tolerance=1.0)
        assert result["passed"] and result["max_dF_meV_per_A"] == 0
        assert "[OK]" in capsys.readouterr().out
        index = json.loads((store / "index.json").read_text())
        assert index["symmetrix-xl-0.2.0/m__abc__1024.json"]["check_float64_O-Si"]["passed"]
        # second call reads the record instead of evaluating MACE again
        def no_mace(*a, **k):
            raise AssertionError("MACE should not be evaluated again")
        monkeypatch.setattr(calc_mod, "_load_mace", no_mace)
        again = acc.validate(json_path, fake_calcs(), model="mace-mpa-0", model_path=None,
                             elements=["O", "Si"], dtype="float64", tolerance=1.0)
        assert again == result

    def test_fail_raises(self, store, fake_calcs):
        json_path = store / "symmetrix-xl-0.2.0" / "m__abc__1024.json"
        json_path.parent.mkdir(parents=True)
        json_path.write_text("{}")
        with pytest.raises(AcceleratorError, match="differs from MACE"):
            acc.validate(json_path, fake_calcs(offset=0.01), model="mace-mpa-0",
                         model_path=None, elements=["O"], dtype="float64", tolerance=1.0)

    def test_failed_check_keeps_refusing(self, store, fake_calcs, monkeypatch):
        """A recorded failure must stop every later run, not only the first."""
        json_path = store / "symmetrix-xl-0.2.0" / "m__abc__1024.json"
        json_path.parent.mkdir(parents=True)
        json_path.write_text("{}")
        kwargs = dict(model="mace-mpa-0", model_path=None, elements=["O"],
                      dtype="float64")
        for _ in range(2):
            with pytest.raises(AcceleratorError, match="differs from MACE"):
                acc.validate(json_path, fake_calcs(offset=0.01), tolerance=1.0, **kwargs)
        # the record is reused (no second MACE evaluation) and judged against
        # the tolerance asked for now
        monkeypatch.setattr(calc_mod, "_load_mace", lambda *a, **k: 1 / 0)
        assert acc.validate(json_path, fake_calcs(offset=0.01), tolerance=20.0,
                            **kwargs)["max_dF_meV_per_A"] == pytest.approx(10.0)
        with pytest.raises(AcceleratorError, match="tolerance 5.0"):
            acc.validate(json_path, fake_calcs(offset=0.01), tolerance=5.0, **kwargs)

    def test_options_get_their_own_check(self, store, fake_calcs):
        json_path = store / "symmetrix-xl-0.2.0" / "m__abc__1024.json"
        json_path.parent.mkdir(parents=True)
        json_path.write_text("{}")
        kwargs = dict(model="mace-mpa-0", model_path=None, elements=["O"],
                      dtype="float32", tolerance=1.0)
        acc.validate(json_path, fake_calcs(), **kwargs)
        # a passing CPU check must not cover the Metal code path
        with pytest.raises(AcceleratorError, match="differs from MACE"):
            acc.validate(json_path, fake_calcs(offset=0.01), options={"metal": True},
                         **kwargs)
        entry = json.loads((store / "index.json").read_text())[
            "symmetrix-xl-0.2.0/m__abc__1024.json"]
        assert len([k for k in entry if k.startswith("check_float32_O")]) == 2

    def test_check_on_first_use_when_elements_unknown(self, store, fake_calcs,
                                                      installed, monkeypatch, capsys):
        """Python API: no elements given, so they come from the first structure."""
        from ase.build import bulk
        installed(symmetrix_xl="0.2.0")
        json_path = store / "symmetrix-xl-0.2.0" / "m__abc__1024.json"
        json_path.parent.mkdir(parents=True)
        json_path.write_text("{}")
        monkeypatch.setattr(acc, "prepare", lambda *a, **k: json_path)
        good, bad = fake_calcs(), fake_calcs(offset=0.01)
        settings = AcceleratorSettings.from_config("symmetrix-xl")

        monkeypatch.setattr(calc_mod, "_load_symmetrix", lambda *a, **k: good)
        calc = acc.load_accelerated(settings, model="mace-mpa-0", model_path=None,
                                    dtype="float64", device="cpu")
        assert "first structure" in capsys.readouterr().out
        assert not (store / "index.json").exists()        # nothing checked yet
        atoms = bulk("NaCl", "rocksalt", a=5.6)
        atoms.calc = calc
        atoms.get_potential_energy()
        entry = json.loads((store / "index.json").read_text())[
            "symmetrix-xl-0.2.0/m__abc__1024.json"]
        assert entry["check_float64_Na-Cl"]["passed"]
        assert "calculate" not in vars(calc)              # the hook is gone
        atoms.rattle(0.01)
        atoms.get_potential_energy()

        (store / "index.json").unlink()
        monkeypatch.setattr(calc_mod, "_load_symmetrix", lambda *a, **k: bad)
        calc = acc.load_accelerated(settings, model="mace-mpa-0", model_path=None,
                                    dtype="float64", device="cpu")
        atoms = bulk("NaCl", "rocksalt", a=5.6)
        atoms.calc = calc
        with pytest.raises(AcceleratorError, match="differs from MACE"):
            atoms.get_potential_energy()

    def test_structure_has_requested_elements(self):
        atoms = acc.test_structure(["O", "Si", "In"])
        assert set(atoms.get_chemical_symbols()) == {"O", "Si", "In"}
        assert atoms.pbc.all() and len(atoms) == 27


class TestGetCalculator:

    def test_routes_mace_through_accelerator(self, monkeypatch):
        seen = {}

        def fake(settings, **kw):
            seen.update(engine=settings.engine, **kw)
            return "ACCELERATED"
        monkeypatch.setattr(acc, "load_accelerated", fake)
        out = calc_mod.get_calculator("mace-mpa-0", device="cpu", accelerator="symmetrix-xl")
        assert out == "ACCELERATED"
        assert seen == {"engine": "symmetrix-xl", "model": "mace-mpa-0", "model_path": None,
                        "dtype": "float64", "device": "cpu"}

    def test_falls_back_to_mace_when_auto_finds_nothing(self, monkeypatch):
        monkeypatch.setattr(acc, "load_accelerated", lambda settings, **kw: None)
        monkeypatch.setattr(calc_mod, "_load_mace", lambda *a, **k: "MACE")
        assert calc_mod.get_calculator("mace-mpa-0", device="cpu", accelerator="auto") == "MACE"

    def test_mps_goes_to_accelerator_not_cpu_warning(self, monkeypatch, recwarn):
        seen = {}
        monkeypatch.setattr(acc, "load_accelerated",
                            lambda settings, **kw: seen.update(kw) or "ACCELERATED")
        calc_mod.get_calculator("mace-mpa-0", device="mps", accelerator="symmetrix-xl")
        assert seen["device"] == "cpu"
        assert not [w for w in recwarn if "mps" in str(w.message)]

    def test_explicit_with_non_mace_refused(self):
        with pytest.raises(AcceleratorError, match="classical backend"):
            calc_mod.get_calculator("lj", device="cpu", accelerator="symmetrix-xl",
                                    classical_params={"params": {"Ar-Ar": {"epsilon": 0.01, "sigma": 3.4}}})

    def test_auto_with_non_mace_runs_unchanged(self, capsys):
        calc = calc_mod.get_calculator(
            "lj", device="cpu", accelerator="auto",
            classical_params={"params": {"Ar-Ar": {"epsilon": 0.01, "sigma": 3.4}}})
        assert type(calc).__name__ == "LennardJonesCalculator"
        assert "not accelerated" in capsys.readouterr().out

    def test_calculator_kwargs_carry_accelerator(self):
        cfg = {"model": "mace-mpa-0", "accelerator": {"engine": "auto"}}
        assert calc_mod.calculator_kwargs(cfg)["accelerator"] == {"engine": "auto"}
        assert calc_mod.potential_kwargs(cfg) == {"accelerator": {"engine": "auto"}}


class TestRequireAccelerator:

    def test_off_and_auto_without_packages_pass(self, installed):
        installed()
        require_accelerator(None, backend="mace", model="mace-mpa-0", model_path=None,
                            default_dtype="auto")
        require_accelerator("auto", backend="mace", model="mace-mpa-0", model_path=None,
                            default_dtype="auto")

    def test_torchsim_engine(self, installed, capsys):
        """torch-sim runs with Symmetrix-XL as its model; auto without it runs
        plain MACE."""
        kwargs = dict(backend="mace", model="mace-mpa-0", model_path=None,
                      default_dtype="auto", engine="torchsim")
        installed(symmetrix_xl="0.2.0")
        require_accelerator("symmetrix-xl", **kwargs)
        require_accelerator("auto", **kwargs)
        assert "runs plain MACE" not in capsys.readouterr().out
        with pytest.raises(AcceleratorError, match="is for the original Symmetrix"):
            require_accelerator({"engine": "symmetrix-xl", "options": {"use_kokkos": False}},
                                **kwargs)
        # a non-MACE model is still refused by name
        with pytest.raises(AcceleratorError, match="accelerates MACE models"):
            require_accelerator("symmetrix-xl", backend="chgnet", model="chgnet",
                                model_path=None, default_dtype="auto", engine="torchsim")
        for have in ({}, {"symmetrix": "0.3.1"}):
            installed(**have)
            require_accelerator("auto", **kwargs)
            assert "runs plain MACE" in capsys.readouterr().out
            with pytest.raises(AcceleratorError, match="was requested, but"):
                require_accelerator("symmetrix-xl", **kwargs)
        installed(symmetrix_xl="0.2.0")
        require_accelerator("symmetrix-xl", backend="mace", model="mace-mpa-0",
                            model_path=None, default_dtype="auto", engine="ase")

    def test_dtype_auto_counts_as_float64_for_metal(self, installed, monkeypatch):
        installed(symmetrix_xl="0.2.0")
        monkeypatch.setattr(sys, "platform", "darwin")
        with pytest.raises(AcceleratorError, match="float32"):
            require_accelerator({"engine": "symmetrix-xl", "options": {"metal": True}},
                                backend="mace", model="mace-mpa-0", model_path=None,
                                default_dtype="auto")


class TestCli:

    def test_parse_key_values(self):
        from amorphgen.cli import _parse_key_values
        assert _parse_key_values(["metal=true", "neighbor_skin=0.4", "execution_profile=speed"],
                                 "--x") == {"metal": True, "neighbor_skin": 0.4,
                                            "execution_profile": "speed"}
        assert _parse_key_values(None, "--x") is None
        with pytest.raises(SystemExit):
            _parse_key_values(["metal"], "--x")

    def test_flags_reach_override(self):
        from amorphgen.cli import _build_override, _get_parser
        parser = _get_parser()
        args = parser.parse_args(["POSCAR", "--accelerator", "symmetrix-xl",
                                  "--accelerator-option", "execution_profile=speed"])
        override = _build_override(args, parser, explicit_only=True)
        assert override["accelerator"] == {"engine": "symmetrix-xl",
                                           "options": {"execution_profile": "speed"}}

    def test_elements_from_composition(self):
        from amorphgen.cli import _fill_accelerator_elements
        args = SimpleNamespace(composition="InGaZnO4*2", input_file=None, input_dir=None)
        override = {"accelerator": {"engine": "symmetrix"}}
        _fill_accelerator_elements(args, override)
        assert override["accelerator"]["elements"] == ["Ga", "In", "O", "Zn"]

    def test_elements_from_input_file(self, tmp_path):
        from ase.build import bulk
        from amorphgen.cli import _fill_accelerator_elements
        path = tmp_path / "in.xyz"
        bulk("NaCl", "rocksalt", a=5.6).write(str(path))
        override = {"accelerator": "auto"}
        _fill_accelerator_elements(
            SimpleNamespace(composition=None, input_file=str(path), input_dir=None), override)
        assert override["accelerator"] == {"engine": "auto", "elements": ["Cl", "Na"]}

    def test_elements_from_snapshots(self, tmp_path):
        """--batch-quench: a snapshot directory or a trajectory file."""
        from ase.build import bulk
        from ase.io import write
        from amorphgen.cli import _fill_accelerator_elements
        folder = tmp_path / "snapshots"
        folder.mkdir()
        bulk("NaCl", "rocksalt", a=5.6).write(str(folder / "snap_0000.xyz"))
        traj = tmp_path / "stage4_eq_traj.xyz"
        write(str(traj), [bulk("MgO", "rocksalt", a=4.2)] * 2)
        for source, expected in ((folder, ["Cl", "Na"]), (traj, ["Mg", "O"])):
            override = {"accelerator": {"engine": "symmetrix"}}
            _fill_accelerator_elements(
                SimpleNamespace(composition=None, input_file=None, input_dir=None,
                                snapshot_dir=str(source)), override)
            assert override["accelerator"]["elements"] == expected

    def test_elements_not_touched_when_off_or_given(self):
        from amorphgen.cli import _fill_accelerator_elements
        args = SimpleNamespace(composition="SiO2", input_file=None, input_dir=None)
        off = {"accelerator": {"engine": "none"}}
        _fill_accelerator_elements(args, off)
        assert "elements" not in off["accelerator"]
        given = {"accelerator": {"engine": "auto", "elements": ["Si"]}}
        _fill_accelerator_elements(args, given)
        assert given["accelerator"]["elements"] == ["Si"]

    def test_gate_stops_before_work(self, tmp_path, monkeypatch, capsys, installed):
        from amorphgen import cli
        installed()
        monkeypatch.setattr(calc_mod, "backend_available", lambda b: True)
        poscar = tmp_path / "POSCAR"
        poscar.write_text("Si\n1.0\n5.43 0 0\n0 5.43 0\n0 0 5.43\nSi\n1\nDirect\n0 0 0\n")
        monkeypatch.setattr("sys.argv", ["amorphgen", str(poscar), "--accelerator", "symmetrix-xl",
                                         "--work-dir", str(tmp_path / "run")])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        assert "symmetrix-xl is not installed" in capsys.readouterr().out
        assert not (tmp_path / "run").exists()


class TestYaml:

    def _load(self, tmp_path, text):
        from amorphgen.configs.yaml_config import load_yaml_config
        path = tmp_path / "c.yaml"
        path.write_text(text)
        return load_yaml_config(str(path))

    def test_shorthand_becomes_block(self, tmp_path):
        assert self._load(tmp_path, "accelerator: symmetrix-xl\n")["accelerator"] \
            == {"engine": "symmetrix-xl"}

    def test_block(self, tmp_path):
        cfg = self._load(tmp_path, "accelerator:\n  engine: auto\n  spline_points: 4096\n"
                                   "  elements: [In, O]\n  options: {metal: true}\n")
        assert cfg["accelerator"]["spline_points"] == 4096

    @pytest.mark.parametrize("text, match", [
        ("accelerator: turbo\n", "Invalid accelerator"),
        ("accelerator:\n  engine: auto\n  bogus: 1\n", "Unknown key"),
        ("accelerator:\n  engine: auto\n  spline_points: many\n", "spline_points"),
    ])
    def test_invalid(self, tmp_path, text, match):
        with pytest.raises(ValueError, match=match):
            self._load(tmp_path, text)

    def test_cli_option_merges_into_yaml_engine(self, tmp_path):
        from amorphgen.utils.common import merge_config
        yaml_cfg = self._load(tmp_path, "accelerator: symmetrix-xl\n")
        merged = merge_config(yaml_cfg, {"accelerator": {"options": {"metal": True}}})
        assert merged["accelerator"] == {"engine": "symmetrix-xl", "options": {"metal": True}}


class TestTestedSeries:
    """The joined evaluation uses private functions of Symmetrix-XL, so a
    release series it was not tested with is announced."""

    def test_tested_series_is_silent(self):
        from amorphgen.utils.accelerator import untested_version
        assert untested_version("symmetrix-xl", "0.2.0") is None
        assert untested_version("symmetrix-xl", "0.2.7") is None

    @pytest.mark.parametrize("version", ["0.3.0", "1.0.0", "0.1.9"])
    def test_other_series_is_announced(self, version):
        from amorphgen.utils.accelerator import untested_version
        message = untested_version("symmetrix-xl", version)
        assert version in message and "0.2.x" in message and "<0.3" in message

    def test_install_hint_names_the_tested_range(self):
        from amorphgen.utils import accelerator
        assert ">=0.2,<0.3" in accelerator._INSTALL["symmetrix-xl"]
