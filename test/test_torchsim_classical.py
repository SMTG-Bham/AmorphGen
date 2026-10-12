"""Buckingham + Coulomb on the torch-sim engine (Tier 2, CPU).

The batched model is built from torch-sim's pair-potential and electrostatics
models; it must reproduce the NumPy calculator in ``amorphgen.utils.classical``.
Skipped when torch-sim is not installed (pip install "amorphgen[torchsim]").
"""
import numpy as np
import pytest
from ase import Atoms

ts = pytest.importorskip("torch_sim")

from amorphgen.utils.calculators import get_calculator  # noqa: E402
from amorphgen.utils.torchsim_engine import build_model, batch_relax  # noqa: E402
from amorphgen.pipeline.random_gen import generate_random  # noqa: E402

# BKS silica (van Beest, Kramer & van Santen, PRL 64, 1955 (1990)).
BKS = {
    "params": {"Si-O": {"A": 18003.7572, "rho": 0.205205, "C": 133.5381},
               "O-O": {"A": 1388.7730, "rho": 0.362319, "C": 175.0}},
    "charges": {"Si": 2.4, "O": -1.2},
    "cutoff": 10.0,
}


def _silica(seed, n_formula=8, a=10.6):
    """A random SiO2 cell at 2.2 g/cm3 with no pair closer than 1.5 A."""
    rng = np.random.default_rng(seed)
    pos = []
    while len(pos) < 3 * n_formula:
        p = rng.uniform(0, a, 3)
        if all(np.linalg.norm((p - q + a / 2) % a - a / 2) > 1.5 for q in pos):
            pos.append(p)
    return Atoms(["Si"] * n_formula + ["O"] * 2 * n_formula, positions=pos,
                 cell=[a, a, a], pbc=True)


def _placed_silica(seed):
    """AmorphGen's own placement (SiO2*24 at 2.2 g/cm3, physical minimum separations):
    a cell the BKS potential relaxes without the pair collapse a crude random cell can
    trigger (BKS has no short-range wall)."""
    return generate_random({"Si": 24, "O": 48}, target_density=2.2, seed=seed,
                           retry_mode="none")


def _torchsim_eval(model, atoms_list):
    state = ts.initialize_state(atoms_list, model.device, model.dtype)
    out = model(state)
    n = [len(a) for a in atoms_list]
    forces = np.split(out["forces"].cpu().numpy(), np.cumsum(n)[:-1])
    return out["energy"].cpu().numpy(), forces


@pytest.mark.parametrize("coulomb_method, e_tol, f_tol", [
    ("ewald", 5e-4, 0.01),   # two Ewald implementations (torch-sim's auto-estimated parameters vs
                             # alpha = 3.5/rc in classical.py); they agree to ~2e-4 eV/atom at 10 A
    (None, 1e-10, 1e-8),     # no Coulomb: the short-range term alone is exact
])
def test_buckingham_matches_numpy_calculator(coulomb_method, e_tol, f_tol):
    cp = dict(BKS)
    if coulomb_method is None:
        cp["coulomb"] = False
    else:
        cp["coulomb_method"] = coulomb_method
    atoms_list = [_silica(k) for k in range(3)]
    model = build_model("buckingham", device="cpu", classical_params=cp)
    energies, forces = _torchsim_eval(model, atoms_list)
    for atoms, e, f in zip(atoms_list, energies, forces):
        atoms.calc = get_calculator("buckingham", device="cpu", classical_params=cp)
        e_ref, f_ref = atoms.get_potential_energy(), atoms.get_forces()
        assert abs(e - e_ref) / len(atoms) < e_tol
        assert np.abs(f - f_ref).max() < f_tol * max(1.0, np.abs(f_ref).max())


def test_wolf_uses_damped_shifted_force_and_stays_close():
    # torch-sim's DSF kernel is the force-shifted relative of the NumPy Wolf sum:
    # same alpha, different kernel and self-term, so the energies carry a
    # structure-dependent offset and only the forces are compared (within ~5 %).
    cp = dict(BKS, coulomb_method="wolf", alpha=0.2)
    atoms = _silica(5)
    model = build_model("buckingham", device="cpu", classical_params=cp)
    (e,), (f,) = _torchsim_eval(model, [atoms])
    atoms.calc = get_calculator("buckingham", device="cpu", classical_params=cp)
    assert np.isfinite(e)
    assert np.abs(f - atoms.get_forces()).max() < 0.1 * np.abs(atoms.get_forces()).max()


def test_model_exposes_what_the_engine_reads():
    model = build_model("buckingham", device="cpu", classical_params=BKS)
    assert str(model.device) == "cpu" and model.compute_forces and model.compute_stress
    assert model.memory_scales_with  # batch-size estimate needs it


def test_charges_follow_a_resliced_batch():
    # The electrostatics read per-atom charges from the state; they are set from
    # the atomic numbers on every call, so a sub-batch with a different atom
    # order still gets Si = +2.4 and O = -1.2.
    model = build_model("buckingham", device="cpu", classical_params=BKS)
    a = _silica(1)
    b = a[::-1]  # reversed order: O first
    (ea,), _ = _torchsim_eval(model, [a])
    (eb,), _ = _torchsim_eval(model, [b])
    assert abs(ea - eb) < 1e-8


def test_batch_relax_fixed_and_cubic_cell():
    model = build_model("buckingham", device="cpu", classical_params=BKS)
    seeds = [_placed_silica(k) for k in range(2)]
    cell0 = seeds[0].cell.lengths()[0]
    out = batch_relax([s.copy() for s in seeds], model, fmax=0.1, max_steps=1500,
                      cell_filter="none", log=lambda *a: None)
    assert all(o.info["max_force"] < 0.1 for o in out)
    assert all(np.allclose(o.cell.lengths(), cell0) for o in out)
    out = batch_relax([s.copy() for s in seeds], model, fmax=0.1, max_steps=1500,
                      cell_filter="cubic", log=lambda *a: None)
    for o in out:  # stress from the pair term + Ewald drives the cell; it stays cubic
        assert o.info["max_force"] < 0.1
        assert np.allclose(o.cell.lengths(), o.cell.lengths()[0], atol=1e-6)
        assert abs(o.cell.lengths()[0] - cell0) > 1e-3


def test_errors_are_explicit():
    with pytest.raises(ValueError, match="pair"):
        build_model("buckingham", device="cpu", classical_params={"params": {}})
    with pytest.raises(ValueError, match="charges"):
        build_model("buckingham", device="cpu",
                    classical_params={"params": BKS["params"], "coulomb": True})
    with pytest.raises(ValueError, match="coulomb_method"):
        build_model("buckingham", device="cpu",
                    classical_params=dict(BKS, coulomb_method="pme"))
    with pytest.raises(ValueError, match="chgnet"):
        build_model("chgnet", device="cpu")
