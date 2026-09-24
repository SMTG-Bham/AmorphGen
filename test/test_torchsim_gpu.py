"""GPU tests for the torch-sim engines (Tier 3: need a CUDA device and torch-sim).

Skipped automatically without CUDA. On BlueBEAR:
    sbatch examples/run_gpu_tests_bluebear.slurm
or, in an interactive GPU session:
    python -m pytest test/test_torchsim_gpu.py -v
The MACE tests also need `pip install "amorphgen[mace]"` (mace-torch).
"""
import os
import sys
import time

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.io import write, read

ts = pytest.importorskip("torch_sim")
torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

from amorphgen.utils.torchsim_engine import build_model, batch_relax  # noqa: E402
from amorphgen.utils.torchsim_md import batch_nvt  # noqa: E402

LJ = {"params": {"Cu-Cu": {"sigma": 2.3, "epsilon": 0.1}}, "cutoff": 6.0}


def _cu(k, scale=1.05):
    a = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    a.rattle(0.15, seed=k); a.set_cell(a.cell * scale, scale_atoms=True)
    return a


def _sio2(seed):
    from amorphgen.pipeline.random_gen import generate_random
    return generate_random({"Si": 16, "O": 32}, seed=seed)          # 48 atoms, ~9 A cell


class TestLJOnCuda:
    def test_batch_relax_on_cuda(self):
        model = build_model("lennard-jones", device="cuda", classical_params=LJ)
        assert str(model.device).startswith("cuda")
        out = batch_relax([_cu(k) for k in range(8)], model, fmax=0.02, max_steps=800,
                          cell_filter="cubic", log=lambda *a: None)
        st = ts.io.atoms_to_state(out, device=model.device, dtype=model.dtype)
        stress = model(st)["stress"].detach().cpu().numpy()
        for o, s in zip(out, stress):
            assert o.info["max_force"] < 0.02
            assert abs(np.trace(s) / 3) * 160.21766 < 0.02              # |P| < 0.02 GPa
            assert np.allclose(o.cell.lengths(), o.cell.lengths()[0], atol=1e-6)

    def test_batch_nvt_on_cuda(self):
        model = build_model("lennard-jones", device="cuda", classical_params=LJ)
        out = batch_nvt([_cu(k, 1.0) for k in range(4)], model, 300.0, n_steps=500,
                        timestep_fs=1.0, seed=1, log=lambda *a: None)
        assert all(np.abs(a.get_momenta()).sum() > 0 for a in out)
        assert all(120 < a.get_temperature() < 500 for a in out)


class TestMaceOnCuda:
    @pytest.fixture(scope="class")
    def model(self):
        pytest.importorskip("mace")
        return build_model("mace-mpa-0", device="cuda", dtype="float64")

    def test_relax_matches_ase_mace_energy(self, model):
        """torch-sim relaxation reaches the force/pressure tolerance, and ASE's
        MACE calculator agrees on the energy of the relaxed structure."""
        from mace.calculators import mace_mp
        ats = [_sio2(s) for s in (1, 2)]
        out = batch_relax(ats, model, fmax=0.1, max_steps=300, cell_filter="cubic",
                          log=lambda *a: None)
        calc = mace_mp(model="medium-mpa-0", default_dtype="float64", device="cuda")
        for a in out:
            b = a.copy(); b.calc = calc
            assert abs(b.get_potential_energy() - a.get_potential_energy()) / len(a) < 1e-3
            assert np.abs(b.get_forces()).max() < 0.1 + 0.02
            P = -np.trace(b.get_stress(voigt=False)) / 3 / units.GPa
            assert abs(P) < 0.05

    def test_batched_md_temperature_and_speed(self, model):
        ats = [_sio2(s) for s in range(1, 5)]
        ats = batch_relax(ats, model, fmax=0.5, max_steps=50, cell_filter="none", log=lambda *a: None)
        t = time.time()
        out = batch_nvt(ats, model, 1500.0, n_steps=200, timestep_fs=0.5, seed=3, log=lambda *a: None)
        dt = time.time() - t
        assert all(np.abs(a.get_momenta()).sum() > 0 for a in out)
        assert 600 < np.mean([a.get_temperature() for a in out]) < 2400
        print(f"\n[gpu] batched MD: 4 x 48 atoms, 200 steps in {dt:.1f} s "
              f"({1000 * dt / 200 / 4:.1f} ms per step per structure)")

    def test_hybrid_cli_end_to_end(self, model, tmp_path, monkeypatch):
        from amorphgen.cli import main
        src = tmp_path / "in"; src.mkdir()
        for s in (1, 2, 3):
            write(str(src / f"s{s}.xyz"), _sio2(s), format="extxyz")
        argv = ["amorphgen", "--hybrid-ensemble", "--input-dir", str(src), "-o", str(tmp_path / "out"),
                "-m", "mace-mpa-0", "-d", "cuda", "--default-dtype", "float32", "--seed", "7",
                "--engine", "torchsim", "--batch-size", "3", "--timestep", "0.5",
                "--eq-high-ensemble", "NVT", "--eq-high-T", "1500", "--eq-high-steps", "200",
                "--quench-ensemble", "NVT", "--quench-T-start", "1500", "--quench-T-end", "300",
                "--quench-T-step", "-400", "--quench-steps-per-T", "50",
                "--eq-low-ensemble", "NVT", "--eq-low-T", "300", "--eq-low-steps", "100",
                "-C", "cubic", "-f", "0.2", "--opt-steps", "100"]
        monkeypatch.setattr(sys, "argv", argv); main()
        final = sorted((tmp_path / "out" / "final").glob("*.xyz"))
        assert len(final) == 3
        assert (tmp_path / "out" / "quench_runs" / "run_0000" / "stage4_eq_traj.xyz").exists()
        assert len(read(str(final[0]))) == 48
