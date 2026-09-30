"""Real MACE integration tests, enabled with ``pytest test --run-mace``.

Requires mace-torch. The first run may download the foundation model. These
small checks use CPU so the opt-in suite also works without a GPU.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
from ase.io import write

pytestmark = pytest.mark.mace


@pytest.fixture(scope="module")
def mace_calc():
    pytest.importorskip("mace")
    import torch
    from amorphgen.utils import get_calculator

    original_dtype = torch.get_default_dtype()
    try:
        yield get_calculator(model="mace-mpa-0", device="cpu")
    finally:
        torch.set_default_dtype(original_dtype)


def test_mace_calculator_computes_energy_forces_and_stress(mace_calc):
    assert isinstance(mace_calc, Calculator)
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms.calc = mace_calc
    assert np.isfinite(atoms.get_potential_energy())
    forces = atoms.get_forces()
    assert forces.shape == (len(atoms), 3)
    assert np.isfinite(forces).all()
    stress = atoms.get_stress()
    assert stress.shape == (6,)
    assert np.isfinite(stress).all()


def test_stages_1_to_7_al(tmp_path, mace_calc):
    """A short real-model pipeline preserves atoms and produces finite geometry."""
    from amorphgen import MeltQuenchPipeline

    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat(2)
    source = tmp_path / "Al_test.cif"
    write(source, atoms)
    cfg_override = {
        "model": "mace-mpa-0",
        "device": "cpu",
        "seed": 17,
        "opt": {"fmax": 0.1, "max_steps": 10, "fix_symmetry": False},
        "melt": {
            # NVT avoids barostat instability on this small test cell.
            "ensemble": "NVT", "T_start": 300, "T_end": 500,
            "T_step": 100, "steps_per_T": 5, "make_cubic": False,
        },
        "eq_premelt": {"ensemble": "NVT", "T": 300, "steps": 5},
        "eq_high": {
            "ensemble": "NVT", "T": 500, "steps": 5,
            "sample_interval_ps": None,
        },
        "eq_low": {"ensemble": "NVT", "T": 300, "steps": 5},
        "quench": {
            "ensemble": "NVT", "T_start": 500, "T_end": 300,
            "T_step": -100, "steps_per_T": 5,
        },
        "final_opt": {"fmax": 0.1, "max_steps": 10, "fix_symmetry": False},
    }
    pipeline = MeltQuenchPipeline(
        input_file=str(source), work_dir=str(tmp_path / "al_pipeline"),
        cfg_override=cfg_override, calc=mace_calc,
    )
    result = pipeline.run()
    if isinstance(result, tuple):
        result, _ = result
    assert isinstance(result, Atoms)
    assert result.get_chemical_symbols() == atoms.get_chemical_symbols()
    assert np.isfinite(result.positions).all()
    assert np.isfinite(result.cell.array).all()
    assert result.get_volume() > 0
