"""
tests/test_mace_integration.py
-------------------------------
Tier 3 — MACE integration tests.

These tests use a real MACE calculator and require:
  - mace-torch installed
  - Internet access (first run downloads the model)
  - Ideally a GPU (falls back to CPU — slow)

Run with:
    pytest tests/test_mace_integration.py --run-mace -v

These are skipped in CI by default.
"""

import os
import pytest
from ase.build import bulk


@pytest.mark.mace
class TestMaceCalculatorFactory:

    def test_get_mace_calculator_cpu(self):
        from amorphgen.utils import get_mace_calculator
        calc = get_mace_calculator(model="mace-mpa-0", device="cpu")
        assert calc is not None

    def test_invalid_model_path_raises(self, tmp_path):
        from amorphgen.utils import get_mace_calculator
        with pytest.raises(FileNotFoundError):
            get_mace_calculator(model_path="/nonexistent/path/model.model",
                                device="cpu")

    def test_mace_calculator_can_compute_energy(self):
        from amorphgen.utils import get_mace_calculator
        atoms = bulk("Al", "fcc", a=4.05, cubic=True)
        calc  = get_mace_calculator(model="mace-mpa-0", device="cpu")
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        assert energy < 0    # Al should have negative cohesive energy


@pytest.mark.mace
class TestFullPipelineWithMACE:
    """
    End-to-end pipeline test with real MACE on a tiny Al cell.
    Not a physically meaningful amorphisation — just checks no crashes.
    """

    def test_stages_1_to_7_al(self, tmp_path):
        from amorphgen import MeltQuenchPipeline
        from ase.build import bulk
        from ase.io import write
        from ase import Atoms

        atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat(2)
        poscar = os.path.join(tmp_path, "Al_test.cif")
        write(poscar, atoms)
        n_atoms = len(atoms)

        cfg_override = {
            "mace_model": "mace-mpa-0",
            "device":     "cpu",
            "opt": {
                "fmax": 0.1, "max_steps": 10,
                "fix_symmetry": False,
            },
            "melt": {
                # NVT avoids NPT barostat instability on tiny test cells
                "ensemble":    "NVT",
                "T_start":     300, "T_end": 500,
                "T_step":      100, "steps_per_T": 5,
                "make_cubic":  False,
            },
            "eq_premelt": {
                "ensemble": "NVT", "steps": 5,
            },
            "eq_high": {
                "ensemble":           "NVT",
                "temperature_K":      500,
                "steps":              5,
                "sample_interval_ps": None,   # disable snapshots
            },
            "eq_low": {
                "ensemble": "NVT", "steps": 5,
            },
            "quench": {
                "ensemble":    "NVT",
                "T_start":     500, "T_end": 300,
                "T_step":      -100, "steps_per_T": 5,
            },
            "final_opt": {
                "fmax": 0.1, "max_steps": 10,
                "fix_symmetry": False,
            },
        }

        work_dir = os.path.join(tmp_path, "al_pipeline")
        pipeline = MeltQuenchPipeline(
            input_file=poscar,
            work_dir=work_dir,
            cfg_override=cfg_override,
        )
        result = pipeline.run()

        # pipeline.run() returns Atoms or (Atoms, snapshot_paths)
        if isinstance(result, tuple):
            result, _ = result

        assert isinstance(result, Atoms), (
            f"Expected Atoms, got {type(result)}")
        assert len(result) == n_atoms, (
            f"Atom count changed: {n_atoms} → {len(result)}")
