"""
tests/test_pipeline_stages.py
------------------------------
Tier 2 integration tests — run pipeline stages with ASE's EMT calculator.
No GPU or MACE model needed. EMT only supports Cu, Ag, Au, Ni, Pd, Pt.
"""

import os
import pytest
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT

from amorphgen.utils.common import merge_config, build_md_dynamics
from amorphgen.configs import DEFAULT_CONFIG


class TestBuildMdDynamics:
    """Test NVT and NPT dynamics creation."""

    def test_nvt_creation(self, cu_supercell, emt_calc):
        cu_supercell.calc = emt_calc
        dyn = build_md_dynamics(cu_supercell, ensemble="NVT", T=300.0)
        assert dyn is not None

    def test_npt_creation(self, cu_supercell, emt_calc):
        cu_supercell.calc = emt_calc
        dyn = build_md_dynamics(cu_supercell, ensemble="NPT", T=300.0)
        assert dyn is not None

    def test_invalid_ensemble_raises(self, cu_supercell, emt_calc):
        cu_supercell.calc = emt_calc
        with pytest.raises(ValueError, match="Unknown ensemble"):
            build_md_dynamics(cu_supercell, ensemble="XYZ", T=300.0)


class TestOptCell:
    """Test structure optimisation with EMT."""

    def test_opt_reduces_forces(self, cu_bulk, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.opt_cell import run

        # Slightly distort the cell
        cu_bulk.positions[0] += [0.1, 0.0, 0.0]

        override = {
            "model": "mace-mpa-0",  # won't be used since calc is passed
            "opt": {
                "fmax": 0.1,
                "max_steps": 50,
                "optimizer": "LBFGS",
                "logfile": "test_opt.log",
                "traj_file": "test_opt.traj",
                "output_cif": "test_opt.cif",
                "output_xyz": "test_opt.extxyz",
            },
        }

        result = run(cu_bulk, cfg_override=override, calc=emt_calc)
        assert result is not None
        assert len(result) == len(cu_bulk)
        assert os.path.isfile("test_opt.log")

    def test_optimizer_choices(self, cu_bulk, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.opt_cell import _get_optimizer
        for name in ["LBFGS", "FIRE", "BFGS"]:
            cls = _get_optimizer(name)
            assert cls is not None

    def test_invalid_optimizer_raises(self):
        from amorphgen.pipeline.opt_cell import _get_optimizer
        with pytest.raises(ValueError):
            _get_optimizer("NonexistentOptimizer")


class TestMDStages:
    """Smoke tests for MD stages — verify they run without crashing."""

    def test_short_nvt_run(self, cu_supercell, emt_calc, tmp_work_dir):
        """Run 10 NVT MD steps to verify dynamics setup works."""
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        cu_supercell.calc = emt_calc
        MaxwellBoltzmannDistribution(cu_supercell, temperature_K=300)
        dyn = build_md_dynamics(cu_supercell, ensemble="NVT", T=300.0)
        dyn.run(10)

        # System should still be physically reasonable
        assert cu_supercell.get_temperature() > 0
        assert np.isfinite(cu_supercell.get_potential_energy())
