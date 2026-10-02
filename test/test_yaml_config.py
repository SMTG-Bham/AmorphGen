"""
tests/test_yaml_config.py
--------------------------
Tests for YAML configuration loading and validation.
"""

from importlib.resources import as_file, files
import pytest
import yaml

from amorphgen.configs import load_yaml_config
from amorphgen.configs.yaml_config import _validate_config


class TestLoadYamlConfig:

    @pytest.mark.parametrize("name", ["example_config.yaml", "example_classical.yaml"])
    def test_bundled_examples(self, name):
        # CI also runs this suite outside the checkout against the built wheel.
        with as_file(files("amorphgen.configs").joinpath(name)) as path:
            cfg = load_yaml_config(str(path))
        assert "model" in cfg

    def test_basic_load(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("model: chgnet\ndevice: cpu\n")
        cfg = load_yaml_config(str(cfg_file))
        assert cfg["model"] == "chgnet"
        assert cfg["device"] == "cpu"

    def test_nested_config(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "model: mace-mpa-0\n"
            "opt:\n"
            "  fmax: 0.01\n"
            "  max_steps: 500\n"
        )
        cfg = load_yaml_config(str(cfg_file))
        assert cfg["opt"]["fmax"] == 0.01
        assert cfg["opt"]["max_steps"] == 500

    def test_random_gen_block(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "random_gen:\n"
            "  composition:\n"
            "    Si: 16\n"
            "    O: 32\n"
            "  n_structures: 5\n"
            "  target_density: 2.2\n"
            "  target_cn:\n"
            "    Si: 4\n"
            "    O: 2\n"
        )
        cfg = load_yaml_config(str(cfg_file))
        assert cfg["random_gen"]["composition"] == {"Si": 16, "O": 32}
        assert cfg["random_gen"]["target_cn"]["Si"] == 4

    def test_empty_file_raises(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_yaml_config(str(cfg_file))

    def test_non_dict_raises(self, tmp_path):
        cfg_file = tmp_path / "list.yaml"
        cfg_file.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="mapping"):
            load_yaml_config(str(cfg_file))

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_yaml_config("/nonexistent/config.yaml")

    def test_all_stage_keys(self, tmp_path):
        cfg_file = tmp_path / "full.yaml"
        cfg_file.write_text(
            "model: chgnet\n"
            "device: cpu\n"
            "opt:\n  fmax: 0.05\n"
            "eq_premelt:\n  T: 300\n"
            "melt:\n  T_end: 3000\n"
            "eq_high:\n  T: 3000\n"
            "quench:\n  T_start: 3000\n"
            "eq_low:\n  T: 300\n"
        )
        cfg = load_yaml_config(str(cfg_file))
        assert "opt" in cfg
        assert "eq_premelt" in cfg
        assert "melt" in cfg
        assert "eq_high" in cfg
        assert "quench" in cfg
        assert "eq_low" in cfg


class TestConfigValidation:
    """Tests for YAML config schema validation."""

    def test_valid_config_no_warnings(self):
        cfg = {
            "model": "mace-mpa-0",
            "device": "cuda",
            "opt": {"fmax": 0.01, "max_steps": 500},
        }
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert len(warnings) == 0
        assert len(errors) == 0

    def test_unknown_top_key_errors(self):
        cfg = {"model": "chgnet", "banana": 42}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert warnings == []
        assert any("banana" in e for e in errors)

    def test_wrong_type_top_key_errors(self):
        cfg = {"model": 123}  # should be str
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("model" in e and "int" in e for e in errors)

    def test_invalid_device_errors(self):
        cfg = {"device": "gpu"}  # should be cuda/cpu/mps/auto
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("device" in e and "gpu" in e for e in errors)

    @pytest.mark.parametrize("device", ["cuda", "cpu", "mps", "auto"])
    def test_valid_devices(self, device):
        warnings, errors = _validate_config({"device": device}, "test.yaml")
        assert warnings == []
        assert errors == []

    def test_invalid_ensemble_errors(self):
        cfg = {"melt": {"ensemble": "NVE"}}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("ensemble" in e and "NVE" in e for e in errors)

    @pytest.mark.parametrize("ensemble", ["NVT", "NPT", "nvt", "npt"])
    def test_valid_ensembles(self, ensemble):
        warnings, errors = _validate_config({"melt": {"ensemble": ensemble}}, "test.yaml")
        assert warnings == []
        assert errors == []

    def test_negative_temperature_errors(self):
        cfg = {"eq_premelt": {"T": -100}}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("T" in e and "positive" in e for e in errors)

    def test_negative_fmax_errors(self):
        cfg = {"opt": {"fmax": -0.01}}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("fmax" in e and "positive" in e for e in errors)

    def test_wrong_type_in_stage_errors(self):
        cfg = {"opt": {"fmax": "not a number"}}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("fmax" in e and "str" in e for e in errors)

    @pytest.mark.parametrize("block", [
        "opt", "final_opt", "eq_premelt", "melt", "eq_high", "quench",
        "eq_low", "random_gen", "analysis", "classical_params", "convert",
    ])
    def test_unknown_block_key_errors(self, block):
        cfg = {block: {"banana": 42}}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert warnings == []
        assert any(f"{block}.banana" in e for e in errors)

    @pytest.mark.parametrize("cfg, key_path", [
        ({"banana": 42}, "banana"),
        ({"random_gen": {"n_structure": 5}}, "random_gen.n_structure"),
        ({"analysis": {"save_plots": "plots"}}, "analysis.save_plots"),
        ({"classical_params": {"cutof": 10}}, "classical_params.cutof"),
        ({"classical_params": {"params": {"Si-O": {"rhos": 0.2}}}},
         "classical_params.params.Si-O.rhos"),
        ({"final_opt": {"max_step": 500}}, "final_opt.max_step"),
        ({"convert": {"output_format": "cif"}}, "convert.output_format"),
    ])
    def test_unknown_keys_raise_on_load(self, tmp_path, cfg, key_path):
        cfg_file = tmp_path / "unknown.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg))
        with pytest.raises(ValueError) as exc:
            load_yaml_config(str(cfg_file))
        assert key_path in str(exc.value)

    def test_stage_not_dict_errors(self):
        cfg = {"opt": "not a dict"}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("opt" in e and "dict" in e for e in errors)

    def test_classical_params_accepted(self):
        cfg = {"model": "buckingham", "classical_params": {"cutoff": 10.0}}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert not any("classical_params" in w for w in warnings)
        assert not any("classical_params" in e for e in errors)

    def test_random_gen_options_and_dynamic_maps_accepted(self):
        cfg = {"random_gen": {
            "composition": {"Si": 16, "O": 32}, "n_structures": 5,
            "target_density": 2.2, "density_scale": 1.1,
            "output_format": "vasp", "minsep": {"Si-O": 1.5, "O-O": 2.0},
            "target_cn": {"Si": 4, "O": 2}, "dmax": {"Si-O": 2.0},
            "cn_tolerance": 0, "dmax_factor": 1.5, "cell_filter": "cubic",
            "relax": False, "seed": 42,
        }}
        assert _validate_config(cfg, "test.yaml") == ([], [])

    def test_analysis_options_and_dynamic_maps_accepted(self):
        cfg = {"analysis": {
            "cutoff": {"default": "auto-rdf", "Si-O": 2.0},
            "per_structure": True, "check_dimers": True,
            "total_cn": ["Si:O", "O:Si"], "save_report": "report.txt",
            "save_plot": "plots", "rdf_pairs": ["Si-O"],
            "angle_triplets": ["O-Si-O"], "angle_style": "line",
            "rmax": 6.0, "smearing": 0.0, "total_rdf": True,
            "save_pdf": True, "dpi": 300, "show_title": False,
            "pair_panels": True, "sq": True, "sq_weighting": "neutron",
            "sq_method": "direct", "sq_smooth": 0.1, "sq_partials": True,
            "tr": True, "tr_qrange": [0.5, 25.0], "tr_window": "lorch",
            "tr_scan": True, "rings": ["Si", "O"],
            "ring_bond_pair": "Si-O", "connectivity": True,
            "voronoi": True, "voronoi_element": "Si",
            "reference": "reference.yaml", "energy_ranking": True,
            "voids": True, "void_samples": 1000, "void_probe_radius": 0.0,
            "void_bins": 50, "void_seed": 0, "void_radii": {"Si": 1.1},
            "oxygen_speciation": True, "network_formers": ["Si"],
            "elastic": True, "elastic_strain": 0.005, "elastic_relax": False,
            "vdos": True, "vdos_displacement": 0.01, "vdos_sigma": 0.1,
            "vdos_npoints": 400,
        }}
        assert _validate_config(cfg, "test.yaml") == ([], [])

    @pytest.mark.parametrize("model, params", [
        ("lennard-jones", {"Ar-Ar": {"epsilon": 0.0104, "sigma": 3.4}}),
        ("buckingham", {"Si-O": {"A": 18003.76, "rho": 0.2052, "C": 133.54}}),
    ])
    def test_classical_pair_parameters_accepted(self, model, params):
        cfg = {"model": model, "classical_params": {
            "params": params, "charges": {"Si": 2.4, "O": -1.2},
            "cutoff": 10.0, "coulomb": True, "alpha": None,
            "coulomb_method": "ewald",
        }}
        assert _validate_config(cfg, "test.yaml") == ([], [])

    def test_final_opt_and_convert_load_without_warnings(self, tmp_path, capsys):
        cfg = {
            "final_opt": {"fmax": 0.01, "max_steps": 500,
                          "optimizer": "LBFGS", "cell_filter": "none"},
            "convert": {"input": "input.xyz", "format": "cif",
                        "output_dir": "converted"},
        }
        cfg_file = tmp_path / "supported.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg))
        assert load_yaml_config(str(cfg_file)) == cfg
        assert "warning" not in capsys.readouterr().out.lower()
        assert _validate_config(cfg, "test.yaml") == ([], [])

    @pytest.mark.parametrize("block", ["opt", "final_opt"])
    def test_supported_optimization_output_options(self, block):
        cfg = {block: {
            "logfile": "opt.log", "traj_file": "opt.traj",
            "output_cif": "opt.cif", "output_xyz": "opt.xyz",
            "output_format": "vasp", "pressure_tol_gpa": 0.02,
            "batch_size": "auto",
        }}
        assert _validate_config(cfg, "test.yaml") == ([], [])

    @pytest.mark.parametrize("block", [
        "eq_premelt", "eq_high", "eq_low", "melt", "quench",
    ])
    def test_supported_md_output_options(self, block):
        cfg = {block: {"log_file": "md.log", "traj_file": "md.traj",
                       "output_xyz": "md.xyz"}}
        if block != "quench":
            cfg[block]["make_cubic"] = False
        assert _validate_config(cfg, "test.yaml") == ([], [])

    @pytest.mark.parametrize("cfg, key_path", [
        ({"random_gen": {"composition": {"Silicon": 16}}},
         "random_gen.composition.Silicon"),
        ({"random_gen": {"minsep": {"Si-oxygen": 2.0}}},
         "random_gen.minsep.Si-oxygen"),
        ({"analysis": {"cutoff": {"defaults": 2.0}}}, "analysis.cutoff.defaults"),
        ({"classical_params": {"charges": {"oxygen": -1.2}}},
         "classical_params.charges.oxygen"),
        ({"classical_params": {"params": {"Si-oxygen": {"A": 1.0}}}},
         "classical_params.params.Si-oxygen"),
    ])
    def test_dynamic_map_keys_are_checked(self, cfg, key_path):
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert warnings == []
        assert any(key_path in e for e in errors)

    @pytest.mark.parametrize("cfg", [
        {"random_gen": {"target_density": None, "minsep": None,
                        "target_cn": None, "dmax": None,
                        "cn_tolerance": None, "seed": None}},
        {"analysis": {"network_formers": None, "void_radii": None,
                      "rdf_pairs": None, "angle_triplets": None}},
        {"analysis": {"total_cn": "Si:O", "network_formers": "Si,Ge",
                      "rings": True, "voronoi": "Si", "cutoff": 2.4}},
        {"analysis": {"rings": "Si-O", "ring_bond_pair": ["Si", "O"]}},
        {"convert": {"input": "input.xyz", "output_dir": None}},
    ])
    def test_supported_optional_forms(self, cfg):
        assert _validate_config(cfg, "test.yaml") == ([], [])

    @pytest.mark.parametrize("cfg, key_path", [
        ({"seed": True}, "seed"),
        ({"opt": {"max_steps": True}}, "opt.max_steps"),
        ({"final_opt": {"fmax": "small"}}, "final_opt.fmax"),
        ({"random_gen": {"n_structures": True}}, "random_gen.n_structures"),
        ({"random_gen": {"relax": "yes"}}, "random_gen.relax"),
        ({"random_gen": {"composition": {"Si": "sixteen"}}},
         "random_gen.composition.Si"),
        ({"random_gen": {"target_cn": {"Si": True}}}, "random_gen.target_cn.Si"),
        ({"random_gen": {"minsep": {"Si-O": []}}}, "random_gen.minsep.Si-O"),
        ({"random_gen": {"dmax": {"Si-O": "large"}}}, "random_gen.dmax.Si-O"),
        ({"analysis": {"dpi": True}}, "analysis.dpi"),
        ({"analysis": {"cutoff": {"Si-O": []}}}, "analysis.cutoff.Si-O"),
        ({"analysis": {"void_radii": {"Si": True}}}, "analysis.void_radii.Si"),
        ({"analysis": {"rdf_pairs": [1]}}, "analysis.rdf_pairs"),
        ({"analysis": {"tr_qrange": [0.5, "high"]}}, "analysis.tr_qrange"),
        ({"classical_params": {"charges": {"O": "negative"}}},
         "classical_params.charges.O"),
        ({"classical_params": {"params": {"Si-O": {"A": False}}}},
         "classical_params.params.Si-O.A"),
        ({"classical_params": {"params": {"Si-O": []}}},
         "classical_params.params.Si-O"),
        ({"convert": {"input": []}}, "convert.input"),
    ])
    def test_nested_values_are_type_checked(self, cfg, key_path):
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert warnings == []
        assert any(key_path in e for e in errors)

    @pytest.mark.parametrize("cfg, key_path", [
        ({"device": []}, "device"),
        ({"melt": {"ensemble": []}}, "melt.ensemble"),
        ({"quench": {"npt_method": {}}}, "quench.npt_method"),
        ({"random_gen": []}, "random_gen"),
        ({"analysis": []}, "analysis"),
        ({"classical_params": []}, "classical_params"),
        ({"final_opt": []}, "final_opt"),
        ({"convert": []}, "convert"),
    ])
    def test_malformed_collection_values_report_errors(self, tmp_path, cfg, key_path):
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert warnings == []
        assert any(key_path in e for e in errors)
        cfg_file = tmp_path / "bad_type.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg))
        with pytest.raises(ValueError) as exc:
            load_yaml_config(str(cfg_file))
        assert key_path in str(exc.value)

    def test_negative_steps_errors(self):
        cfg = {"eq_premelt": {"steps": -100}}
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert any("steps" in e and "positive" in e for e in errors)

    def test_full_valid_config(self):
        cfg = {
            "model": "mace-mpa-0",
            "device": "cuda",
            "default_dtype": "float64",
            "opt": {"fmax": 0.01, "max_steps": 1000, "optimizer": "LBFGS"},
            "eq_premelt": {"ensemble": "NVT", "T": 300, "steps": 50000,
                           "timestep": 1.0, "friction": 0.01},
            "melt": {"ensemble": "NPT", "T_start": 300, "T_end": 3000,
                     "T_step": 100, "steps_per_T": 1000, "timestep": 1.0},
            "quench": {"ensemble": "NVT", "T_start": 3000, "T_end": 300,
                       "T_step": -100, "steps_per_T": 1000},
        }
        warnings, errors = _validate_config(cfg, "test.yaml")
        assert len(warnings) == 0
        assert len(errors) == 0

    def test_invalid_yaml_raises(self, tmp_path):
        """YAML with errors should raise ValueError."""
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("model: 123\ndevice: gpu\n")
        with pytest.raises(ValueError, match="error"):
            load_yaml_config(str(cfg_file))
