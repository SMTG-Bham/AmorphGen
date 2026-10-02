"""Compare a live EMT calculation with low-temperature copper measurements.

Reference: W. C. Overton, Jr. and J. Gaffney, Phys. Rev. 98, 969-977 (1955),
https://doi.org/10.1103/PhysRev.98.969, Tables I-II on p. 975.
The ultrasonic single-crystal measurements span 4.2-300 K; we use their
extrapolated 0 K row, where the adiabatic and isothermal constants coincide.
The tabulated values (10**11 dyne/cm**2) are multiplied by 10 to obtain GPa.

EMT is an approximate empirical model, and its static lattice calculation
omits quantum nuclear motion. The acceptance bounds below accommodate model
discrepancy; they are NOT experimental error bars or confidence intervals.
These tests exercise the numerical pipeline against a physical scale, without
claiming independent validation of EMT or of amorphous copper. No download,
optional ML backend, or finite-temperature simulation is required.
"""

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

from amorphgen.analysis.elasticity import compute_elastic_moduli


# The published 0 K values, not fitted to this test's computed output.
CU_EXPERIMENT_GPA = {"C11": 176.20, "C12": 124.94, "C44": 81.77, "bulk": 142.03}

# Rounded model-error budgets: EMT differs from experiment by approximately
# 2.1%, 7.6%, 9.9%, and 5.3%, respectively, at its relaxed zero-pressure cell.
# Keep these separate from the much tighter numerical convergence checks.
EMT_MODEL_RTOL = {"C11": 0.05, "C12": 0.10, "C44": 0.12, "bulk": 0.08}


@pytest.fixture(scope="module")
def relaxed_copper_elasticity():
    """Relax volume first: compute_elastic_moduli only relaxes atomic positions."""
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.calc = EMT()
    optimizer = BFGS(FrechetCellFilter(atoms, hydrostatic_strain=True), logfile=None)
    assert optimizer.run(fmax=1e-6, steps=100), "EMT reference cell did not converge"
    assert np.max(np.abs(atoms.get_stress() / units.GPa)) < 1e-4

    return {
        strain: compute_elastic_moduli(
            [atoms], strain=strain, relax=True, fmax=1e-6, steps=100,
        )["per_structure"][0]
        for strain in (0.001, 0.002)
    }


@pytest.mark.parametrize("component,indices", [
    ("C11", ([0, 1, 2], [0, 1, 2])),
    ("C12", ([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1])),
    ("C44", ([3, 4, 5], [3, 4, 5])),
])
def test_emt_copper_stiffness_compared_with_experiment(
    relaxed_copper_elasticity, component, indices,
):
    tensor = np.asarray(relaxed_copper_elasticity[0.001]["stiffness_tensor_gpa"])
    np.testing.assert_allclose(
        tensor[indices], CU_EXPERIMENT_GPA[component],
        rtol=EMT_MODEL_RTOL[component], atol=0,
        err_msg=f"{component}: EMT exceeds model-error budget versus 0 K experiment",
    )


@pytest.mark.parametrize("average", ["voigt", "reuss", "hill"])
def test_emt_copper_bulk_modulus_compared_with_experiment(
    relaxed_copper_elasticity, average,
):
    # The reference is the source's tabulated bulk modulus, not a value
    # reconstructed from our calculated tensor or an isotropic shear estimate.
    calculated = relaxed_copper_elasticity[0.001]["moduli"][average]["bulk_modulus_gpa"]
    assert calculated == pytest.approx(
        CU_EXPERIMENT_GPA["bulk"], rel=EMT_MODEL_RTOL["bulk"], abs=0,
    )


def test_experimental_comparison_has_converged_strain_and_cubic_symmetry(
    relaxed_copper_elasticity,
):
    result = relaxed_copper_elasticity[0.001]
    tensor = np.asarray(result["stiffness_tensor_gpa"])
    coarser = relaxed_copper_elasticity[0.002]["stiffness_tensor_gpa"]
    # This numerical budget is 0.005 GPa, far below the physical tolerances.
    np.testing.assert_allclose(tensor, coarser, rtol=0, atol=0.005)
    assert result["mechanically_stable"] is True
    assert result["compliance_valid"] is True
    assert result["max_residual_stress_gpa"] < 1e-4
    np.testing.assert_allclose(tensor[:3, 3:], 0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(tensor[3:, :3], 0, rtol=0, atol=1e-6)
    shear = tensor[3:, 3:]
    np.testing.assert_allclose(shear - np.diag(np.diag(shear)), 0, rtol=0, atol=1e-6)
