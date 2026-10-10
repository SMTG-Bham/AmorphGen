"""Malformed inputs, degenerate structures and fallback branches across the analysis modules."""

import json

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import (Calculator, PropertyNotImplementedError,
                                        all_changes)
from ase.io import write

from amorphgen.analysis._serialization import numpy_json_default
from amorphgen.analysis.cutoff import (auto_cutoff_minsep, auto_cutoff_rdf,
                                       parse_cutoff_spec, resolve_cutoffs)
from amorphgen.analysis.elasticity import compute_elastic_moduli
from amorphgen.analysis.energy import format_log_ranking, rank_from_log
from amorphgen.analysis.experiment import compare_experiment, load_experiment
from amorphgen.analysis.melt_memory import report_melt_memory
from amorphgen.analysis.robustness import compute_cutoff_robustness
from amorphgen.analysis.screening import screen_structures, validate_screening_config
from amorphgen.analysis.uncertainty import summarize_site_groups, summarize_structures
from amorphgen.analysis.vibrations import compute_vibrational_dos
from amorphgen.analysis.voronoi import compute_voronoi


@pytest.fixture
def nacl():
    return bulk("NaCl", "rocksalt", a=5.64, cubic=True)


# ─── Experimental comparison ──────────────────────────────────────────────────

@pytest.mark.parametrize("model,measured,match", [
    ({"q": [1, 2], "s_q": [1, 2]}, [[1, 1], [2, 2]], "experiment must be a mapping"),
    ({"q": [1, 2], "s_q": [1, 2]}, {"x": [1, 2]}, "must contain x and observed"),
    ({"q": [1, 2], "s_q": [1, 2]}, {"x": [1, 2], "observed": [[1, 2]]},
     "observations must be a one-dimensional"),
    ([[1, 1], [2, 2]], {"x": [1], "observed": [1]}, "scattering result mapping"),
    ({"r": [1, 2], "s_q": [1, 2]}, {"x": [1], "observed": [1]}, "sq result must contain q"),
    ({"q": [1, 2]}, {"x": [1], "observed": [1]}, "must contain s_q or per_structure"),
    ({"q": ["a", "b"], "s_q": [1, 2]}, {"x": [1], "observed": [1]},
     "calculated x must contain numeric"),
    ({"q": [1, 2], "per_structure": [[1, 2], [1]]}, {"x": [1], "observed": [1]},
     "rectangular structure-by-coordinate"),
])
def test_comparison_rejects_malformed_inputs_by_name(model, measured, match):
    with pytest.raises(ValueError, match=match):
        compare_experiment(model, measured)


def test_loader_rejects_a_bare_column_index(tmp_path):
    path = tmp_path / "measured.dat"
    path.write_text("1 2\n2 3\n")
    with pytest.raises(ValueError, match="columns must select two or three"):
        load_experiment(path, columns=1)


# ─── Voronoi ──────────────────────────────────────────────────────────────────

def test_failed_tessellation_warns_and_keeps_other_structures():
    flat = Atoms("Cu4", positions=[[0, 0, 0], [1.8, 0, 0], [0, 1.8, 0], [1.8, 1.8, 0]],
                 cell=[3.6, 3.6, 0], pbc=[True, True, False])
    fcc = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
    with pytest.warns(UserWarning, match="Voronoi tessellation failed .*4 atoms"):
        result = compute_voronoi([flat, fcc])
    assert [s["tessellation_succeeded"] for s in result["per_structure"]] == [False, True]
    assert result["distribution"] == {(0, 12, 0, 0): 32}     # rhombic dodecahedra
    prevalence = result["uncertainty"]["fraction_of_structures"][(0, 12, 0, 0)]
    assert prevalence["per_structure"] == [None, 1.0]
    assert prevalence["n_structures"] == 1


def test_unbounded_cells_of_a_cell_free_molecule_are_not_counted():
    molecule = Atoms("Cu4", positions=[[0, 0, 0], [1.8, 0, 0], [0, 1.8, 0], [1, 1, 1.8]])
    result = compute_voronoi([molecule])
    assert result["per_structure"][0]["tessellation_succeeded"] is True
    assert result["total_atoms"] == 0 and result["distribution"] == {}
    assert result["per_structure"][0]["mean_faces"] is None


# ─── Cutoff robustness ────────────────────────────────────────────────────────

def _silica_fragment():
    return Atoms("SiO2", positions=[[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]],
                 cell=[20] * 3, pbc=True)


@pytest.mark.parametrize("kwargs,match", [
    ({"atoms_list": []}, "at least one structure"),
    ({"window": "wide"}, "window must be finite"),
    ({"window": None}, "window must be finite"),
    ({"max_cutoff": np.inf}, "finite and nonnegative"),
    ({"get_cutoff_fn": lambda a, b: -1.0}, "finite and nonnegative"),
    ({"get_cutoff_fn": lambda a, b: np.nan}, "finite and nonnegative"),
    ({"max_cutoff": 1.7e308, "window": 1e308}, "nonfinite radii"),
])
def test_robustness_rejects_unusable_settings(kwargs, match):
    arguments = {"atoms_list": [_silica_fragment()], "max_cutoff": 2.0,
                 "get_cutoff_fn": lambda a, b: 2.0, **kwargs}
    with np.errstate(over="ignore", invalid="ignore"), \
            pytest.raises(ValueError, match=match):
        compute_cutoff_robustness(**arguments)


def test_robustness_empty_structure_has_no_contacts_or_centres():
    report = compute_cutoff_robustness([_silica_fragment(), Atoms(cell=[20] * 3, pbc=True)],
                                       2.0, lambda a, b: 2.0)
    pair = report["pairs"]["O-Si"]
    assert [row["n_pairs"] for row in pair["per_structure"]] == [2, 0]
    assert pair["per_structure"][1]["near_fraction"] is None
    cn = pair["coordination"]["Si-O"]
    assert cn["per_structure"] == [[2.0] * 5, None]
    assert cn["mean"] == cn["ensemble_mean"] == [2.0] * 5
    assert cn["total_atoms"] == 1


def test_robustness_zero_global_radius_admits_no_contact():
    report = compute_cutoff_robustness([_silica_fragment()], 0.0, lambda a, b: 2.0)
    assert report["global_cutoffs"] == [0.0] * 5
    assert report["pairs"]["O-Si"]["cutoffs"] == pytest.approx([1.9, 1.95, 2.0, 2.05, 2.1])
    assert report["pairs"]["O-Si"]["n_pairs"] == 0
    assert report["pairs"]["O-Si"]["coordination"]["Si-O"]["mean"] == [0.0] * 5


# ─── Vibrational DOS ──────────────────────────────────────────────────────────

class _Harmonic(Calculator):
    implemented_properties = ["forces"]

    def __init__(self, hessian):
        super().__init__()
        self.hessian = np.asarray(hessian, dtype=float)

    def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["forces"] = -(self.hessian @ atoms.positions.ravel()).reshape(-1, 3)


class _Failing(Calculator):
    implemented_properties = ["forces"]

    def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
        raise RuntimeError("model server unreachable")


class _HardWall(Calculator):
    """Finite forces of +/-1e308 eV/A whose central difference overflows."""

    implemented_properties = ["forces"]

    def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["forces"] = -1e308 * np.sign(atoms.positions)


@pytest.mark.parametrize("parameter,value", [("displacement", "small"), ("sigma", None)])
def test_vdos_rejects_non_numeric_settings(parameter, value):
    with pytest.raises(ValueError, match=f"{parameter} must be a finite positive number"):
        compute_vibrational_dos([Atoms("H")], calculator=_Harmonic(np.eye(3)),
                                **{parameter: value})


def test_vdos_reports_a_failing_calculator_with_the_structure_index():
    working, failing = Atoms("H"), Atoms("He")
    working.calc, failing.calc = _Harmonic(np.eye(3)), _Failing()
    with pytest.raises(ValueError, match="structure 1: model server unreachable") as info:
        compute_vibrational_dos([working, failing])
    assert isinstance(info.value.__cause__, RuntimeError)


@pytest.mark.parametrize("atoms,calculator,kwargs,match", [
    (Atoms("H"), _HardWall(), {}, "Finite-difference Hessian .*not finite"),
    (Atoms("H", masses=[1e-200]), _Harmonic(1e200 * np.eye(3)), {}, "Mass-weighted Hessian"),
    (Atoms("H"), _Harmonic(np.eye(3)), {"sigma": 1e308}, "supported grid range"),
    # A 1e-200 THz Gaussian centred between grid points is zero at every one
    (Atoms("H"), _Harmonic(np.diag([1.0, 2.0, 3.0])), {"sigma": 1e-200},
     "sigma is too small"),
])
def test_vdos_overflow_is_an_error_not_an_infinite_spectrum(atoms, calculator, kwargs, match):
    with np.errstate(over="ignore"), pytest.raises(ValueError, match=match):
        compute_vibrational_dos([atoms], calculator=calculator, **kwargs)


# ─── JSON conversion ──────────────────────────────────────────────────────────

def test_numpy_json_default_converts_numpy_values_without_hiding_nan():
    text = json.dumps({"f": np.float32(1.5), "a": np.arange(3), "i": np.int64(7),
                       "b": np.bool_(True)}, default=numpy_json_default)
    assert json.loads(text) == {"f": 1.5, "a": [0, 1, 2], "i": 7, "b": True}
    with pytest.raises(ValueError):
        json.dumps(np.float32("nan"), default=numpy_json_default, allow_nan=False)
    with pytest.raises(TypeError, match="Cannot serialize object"):
        json.dumps(object(), default=numpy_json_default)


def test_melt_memory_writes_numpy_typed_settings_as_native_json(tmp_path):
    octahedron = np.vstack(([0, 0, 0], np.eye(3), -np.eye(3)))
    crystal = Atoms("Cu7", positions=octahedron)
    write(tmp_path / "crystal.xyz", crystal)
    write(tmp_path / "stage3_melted.xyz", crystal)
    report_melt_memory(tmp_path / "crystal.xyz", tmp_path, [], tmp_path / "out",
                       cutoff=1.1, qbar6_threshold=np.float32(0.25),
                       min_neighbors=np.int64(4))
    saved = json.loads((tmp_path / "out" / "melt_memory.json").read_text())
    assert saved["requested_parameters"] == {"cutoff": 1.1, "qbar6_threshold": 0.25,
                                             "min_neighbors": 4}
    assert saved["comparisons"][0]["survival_fraction"] == 1.0


# ─── Uncertainty helpers ──────────────────────────────────────────────────────

def test_structure_summary_rejects_higher_rank_curves():
    with pytest.raises(ValueError, match="structure-by-bin matrix"):
        summarize_structures([np.zeros((2, 2)), np.zeros((2, 2))])


def test_empty_curve_matrix_keeps_its_bin_count():
    summary = summarize_structures(np.empty((0, 5)))
    assert summary["per_structure"] == [] and summary["n_structures"] == 0
    assert summary["n_per_point"] == [0] * 5
    assert summary["mean"] == summary["sem"] == [None] * 5


def test_site_groups_without_any_site_have_no_summary():
    assert summarize_site_groups([[], None]) == {}


# ─── Cutoff specifications ────────────────────────────────────────────────────

@pytest.mark.parametrize("spec,expected", [
    ("Na-Cl=3.0,", {"Na-Cl": 3.0}),
    ("auto,,Na-Cl=3.0", {"default": "auto", "Na-Cl": 3.0}),
])
def test_cutoff_spec_ignores_empty_tokens(spec, expected):
    assert parse_cutoff_spec(spec) == expected


@pytest.mark.parametrize("spec", [",", " , ", "  "])
def test_cutoff_spec_without_any_rule_is_rejected(spec):
    with pytest.raises(ValueError, match="invalid cutoff"):
        parse_cutoff_spec(spec)


def test_radii_base_with_a_reversed_override(nacl):
    table = auto_cutoff_minsep([nacl])
    cutoffs, label = resolve_cutoffs([nacl], "auto,Na-Cl=3.1")
    assert cutoffs == {"Cl-Cl": table["Cl-Cl"], "Cl-Na": 3.1, "Na-Na": table["Na-Na"]}
    assert label == "auto (minsep) + overrides Na-Cl=3.10"


def test_override_for_an_absent_pair_warns_and_changes_nothing(nacl):
    with pytest.warns(UserWarning, match="not in the structure: Si-O"):
        cutoffs, _ = resolve_cutoffs([nacl], "2.5,Si-O=1.8")
    assert cutoffs == {"Cl-Cl": 2.5, "Cl-Na": 2.5, "Na-Na": 2.5}


def test_too_coarse_rdf_falls_back_to_radii_cutoffs(nacl):
    crystal = nacl.repeat(2)
    with pytest.warns(UserWarning, match="no clear first minimum") as record:
        cutoffs = auto_cutoff_rdf([crystal], nbins=4)        # 1.5 A bins
    assert cutoffs == pytest.approx(auto_cutoff_minsep([crystal]))
    assert sum("no clear first minimum" in str(w.message) for w in record) == 3


# ─── Energy ranking from the random-gen log ───────────────────────────────────

def _relaxation(formula, steps, outcome, index, total):
    rows = "".join(f"    {step:5d}  {energy:14.6f}  {fmax:11.6f}      5.0000      5.0000"
                   f"      5.0000       125.0\n" for step, energy, fmax in steps)
    return (f"\n    Composition: {formula} (8 atoms)\n"
            f"    Optimizer: FIRE  fmax=0.05  max_steps=2\n{rows}"
            f"    {'-' * 85}\n\n    {outcome}\n"
            f"  [{index + 1}/{total}] {formula} -> out/random_opt/random_{index:04d}_opt.vasp"
            f" (seed={index})\n")


def test_log_ranking_keeps_unconverged_status_and_ignores_text_between_blocks(tmp_path):
    log = tmp_path / "random_gen.log"
    log.write_text(
        "  AmorphGen - Random Structure Generation\n  Composition: Si8 (8 atoms)\n"
        + _relaxation("Si8", [(1, -30.0, 0.5), (2, -32.0, 0.04)],
                      "Converged after 2 steps!  Fmax = 0.040000 eV/A", 0, 2)
        + "    1      -99.000000     0.010000\n"     # not inside a relaxation block
        + _relaxation("Si8", [(1, -31.0, 0.9), (2, -33.6, 0.3)],
                      "WARNING: did not converge in 2 steps.", 1, 2)
        + "    2      -99.000000     0.010000\n")
    result = rank_from_log(log)
    assert result["n_atoms"] == 8
    lowest, highest = result["rows"]
    assert lowest[0] == 1 and lowest[4:] == (None, "not converged")
    assert lowest[1:4] == pytest.approx((-33.6, -4.2, 0.3))
    assert highest[0] == 0 and highest[4:] == (2, "converged")
    assert highest[1:4] == pytest.approx((-32.0, -4.0, 0.04))
    assert result["spread_meV_per_atom"] == pytest.approx(200.0)
    text = format_log_ranking(result)
    assert any(line.split()[-3:] == ["N/A", "not", "converged"] for line in text.splitlines())


def test_log_without_relaxations_has_no_ranking(tmp_path):
    log = tmp_path / "random_gen.log"
    log.write_text("  [1/1] Si8 -> out/random_0000.vasp (seed=1)\n")
    result = rank_from_log(log)
    assert result == {"rows": [], "n_atoms": None, "best": None, "worst": None,
                      "spread_meV_per_atom": 0.0}
    assert format_log_ranking(result) == "No converged structures found in log."


# ─── Screening ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("config,match", [
    ({"close_contacts": {"min_distance": {}}}, "pair table must not be empty"),
    ({"coordination": 3}, "screening.coordination must be a boolean or mapping"),
    ({"density": {"min": 1, "maxx": 2}}, "unknown screening.density option.*maxx"),
    ({"coordination": {"allowed": "manual"}}, "'auto' or a nonempty element mapping"),
    ({"crystal_like": {"cutoff": {"Na-Cl": 2.0, "Cl-Na": 2.1}}}, "repeats pair Cl-Na"),
])
def test_screening_config_errors_name_the_setting(config, match):
    with pytest.raises(ValueError, match=match):
        validate_screening_config(config)


def test_crystal_cutoff_pair_table_is_kept_after_validation():
    config = validate_screening_config({"crystal_like": {"cutoff": {"Na-Cl": 3.0,
                                                                    "default": 3.5}}})
    assert config["crystal_like"]["cutoff"] == {"Na-Cl": 3.0, "default": 3.5}


def test_unshareable_cutoffs_make_neighbour_screens_unavailable(nacl):
    # "auto" pair tables come from one composition and cannot cover the other.
    kcl = bulk("KCl", "rocksalt", a=6.29, cubic=True)
    result = screen_structures([nacl, kcl], {
        "coordination": {"exclude": True}, "crystal_like": {"cutoff": "auto"},
        "close_contacts": True}, cutoff="auto")
    for row in result["per_structure"]:
        assert row["labels"] == ["coordination_unavailable", "crystal_like_unavailable"]
        assert row["excluded"]
        for name in ("coordination", "crystal_like"):
            assert "same element types" in row["screens"][name]["metrics"]["reason"]
        assert row["screens"]["close_contacts"]["status"] == "passed"
    assert result["summary"]["excluded"] == 2


# ─── Elastic moduli ───────────────────────────────────────────────────────────

class _ForcesOnly(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}


class _NonFiniteForces(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.full((len(atoms), 3), np.nan),
                        "stress": np.zeros(6)}


class _SaturatedStress(Calculator):
    """+/-1e308 eV/A^3 under +/- xx strain: each stress finite, the slope not."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, reference):
        super().__init__()
        self.length = reference.cell[0, 0]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        stress = np.zeros(6)
        stress[0] = 1e308 * np.sign(round(atoms.cell[0, 0] - self.length, 9))
        self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3)),
                        "stress": stress}


def test_stress_free_calculator_is_reported_with_its_cause():
    with pytest.raises(RuntimeError, match="could not calculate stress at reference") as info:
        compute_elastic_moduli([bulk("Cu", "fcc", a=3.6, cubic=True)],
                               calculator=_ForcesOnly())
    assert isinstance(info.value.__cause__, PropertyNotImplementedError)


def test_relaxation_stops_on_nonfinite_forces():
    with pytest.raises(ValueError, match="non-finite or invalid forces at reference"):
        compute_elastic_moduli([bulk("Cu", "fcc", a=3.6, cubic=True)],
                               calculator=_NonFiniteForces(), relax=True)


def test_overflowing_strain_derivative_is_rejected():
    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    with np.errstate(over="ignore"), \
            pytest.raises(ValueError, match="strain derivatives must be finite"):
        compute_elastic_moduli([cu], calculator=_SaturatedStress(cu))


def _empty_cell():
    return Atoms(cell=[3.6] * 3, pbc=True)


def _nonfinite_position():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.positions[0, 0] = np.nan
    return atoms


@pytest.mark.parametrize("make,match", [
    (_empty_cell, "must contain atoms"),
    (_nonfinite_position, "positions must be finite"),
])
def test_empty_or_nonfinite_structures_rejected_before_any_stress(make, match):
    # _ForcesOnly would fail on the first stress, so a ValueError here comes
    # from validation.
    with pytest.raises(ValueError, match=match):
        compute_elastic_moduli([make()], calculator=_ForcesOnly())
