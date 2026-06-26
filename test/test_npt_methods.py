"""Tests for the NPT integrator selector in `build_md_dynamics`.

Covers the three options exposed by ``npt_method``:

  * ``"berendsen"`` (default) -> ``ase.md.nptberendsen.NPTBerendsen``
  * ``"mtk"``                  -> ``ase.md.nose_hoover_chain.IsotropicMTKNPT``
  * ``"parrinello-rahman"``    -> ``ase.md.npt.NPT``

Plus YAML-schema validation for the new ``npt_method`` key.

Tier 2 — uses EMT calculator (GPU-free, runs in CI).
"""
from __future__ import annotations

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.nptberendsen import NPTBerendsen

# Parrinello-Rahman class was renamed in newer ASE.  Accept either.
try:
    from ase.md.melchionna import MelchionnaNPT as _PR_NPT
except ImportError:  # pragma: no cover
    from ase.md.npt import NPT as _PR_NPT

from amorphgen.utils.common import build_md_dynamics


# ─── Helpers ─────────────────────────────────────────────────────────────


def _make_atoms():
    """Cu bulk supercell with EMT — cheap, GPU-free, NPT-stable."""
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    atoms.calc = EMT()
    return atoms


# ─── Method dispatch ─────────────────────────────────────────────────────


def test_npt_default_is_berendsen():
    """No npt_method -> NPTBerendsen (preserves prior behaviour)."""
    atoms = _make_atoms()
    dyn = build_md_dynamics(atoms, ensemble="NPT", T=300.0, timestep=0.5)
    assert isinstance(dyn, NPTBerendsen)


def test_npt_berendsen_explicit():
    atoms = _make_atoms()
    dyn = build_md_dynamics(
        atoms, ensemble="NPT", T=300.0, timestep=0.5, npt_method="berendsen"
    )
    assert isinstance(dyn, NPTBerendsen)


def test_npt_mtk():
    """MTK opt-in returns ASE's Nose-Hoover-chain isotropic NPT."""
    atoms = _make_atoms()
    dyn = build_md_dynamics(
        atoms, ensemble="NPT", T=300.0, timestep=0.5, npt_method="mtk"
    )
    assert isinstance(dyn, IsotropicMTKNPT)


def test_npt_parrinello_rahman():
    """parrinello-rahman opt-in returns ASE's flexible-cell NPT."""
    atoms = _make_atoms()
    dyn = build_md_dynamics(
        atoms, ensemble="NPT", T=300.0, timestep=0.5,
        npt_method="parrinello-rahman",
    )
    assert isinstance(dyn, _PR_NPT)


def test_npt_method_case_insensitive():
    atoms = _make_atoms()
    dyn = build_md_dynamics(
        atoms, ensemble="NPT", T=300.0, timestep=0.5, npt_method="MTK"
    )
    assert isinstance(dyn, IsotropicMTKNPT)


def test_npt_method_invalid_raises():
    atoms = _make_atoms()
    with pytest.raises(ValueError, match="Unknown npt_method"):
        build_md_dynamics(
            atoms, ensemble="NPT", T=300.0, timestep=0.5, npt_method="bogus"
        )


def test_npt_method_ignored_when_nvt():
    """NVT path should not touch npt_method, even if invalid."""
    atoms = _make_atoms()
    dyn = build_md_dynamics(
        atoms, ensemble="NVT", T=300.0, timestep=0.5, npt_method="bogus"
    )
    assert isinstance(dyn, Langevin)


# ─── 5-step smoke runs (real MD, not just object construction) ──────────


@pytest.mark.parametrize("method", ["berendsen", "mtk", "parrinello-rahman"])
def test_npt_method_runs_5_steps(method):
    """Each NPT method should integrate 5 MD steps without raising."""
    atoms = _make_atoms()
    dyn = build_md_dynamics(
        atoms, ensemble="NPT", T=300.0, timestep=0.5, ttime=25.0,
        npt_method=method,
    )
    dyn.run(5)
    # Sanity: positions are finite and atoms haven't exploded.
    pos = atoms.get_positions()
    assert pos.shape == (len(atoms), 3)
    assert (abs(pos) < 100.0).all(), f"{method}: atoms wandered far"


# ─── YAML schema validation ──────────────────────────────────────────────


def test_yaml_npt_method_accepted(tmp_path):
    """A YAML config with a valid npt_method loads without error."""
    from amorphgen.configs.yaml_config import load_yaml_config

    yaml_text = """
melt:
  ensemble: NPT
  npt_method: mtk
  T_start: 300
  T_end: 3000
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text)
    cfg = load_yaml_config(str(p))
    assert cfg["melt"]["npt_method"] == "mtk"


def test_yaml_npt_method_invalid_rejected(tmp_path):
    """A YAML config with a bogus npt_method raises ValueError."""
    from amorphgen.configs.yaml_config import load_yaml_config

    yaml_text = """
melt:
  ensemble: NPT
  npt_method: bogus
  T_start: 300
  T_end: 3000
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError, match="Invalid YAML config"):
        load_yaml_config(str(p))


def test_default_config_has_berendsen_for_melt():
    """Default config preserves the prior NPT integrator for stage 3."""
    from amorphgen.configs.default_config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["melt"]["ensemble"] == "NPT"
    assert DEFAULT_CONFIG["melt"]["npt_method"] == "berendsen"


def test_default_config_eq_high_uses_mtk():
    """eq_high now defaults to NPT with the MTK Nose-Hoover-chain
    integrator for true canonical fluctuations at the equilibration
    plateau.  This is a deliberate behaviour change."""
    from amorphgen.configs.default_config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["eq_high"]["ensemble"] == "NPT"
    assert DEFAULT_CONFIG["eq_high"]["npt_method"] == "mtk"


# ─── New stability knobs: taup_factor and compressibility_GPa ──────────


def test_taup_factor_propagates_to_berendsen():
    """A larger taup_factor should produce a longer Berendsen taup."""
    atoms = _make_atoms()
    dyn_default = build_md_dynamics(
        atoms, ensemble="NPT", T=300.0, timestep=0.5, ttime=25.0,
        npt_method="berendsen",
    )
    atoms2 = _make_atoms()
    dyn_slow = build_md_dynamics(
        atoms2, ensemble="NPT", T=300.0, timestep=0.5, ttime=25.0,
        npt_method="berendsen", taup_factor=30.0,
    )
    # NPTBerendsen exposes self.taup; the slow one should be 3x larger.
    assert dyn_slow.taup == pytest.approx(3.0 * dyn_default.taup)


def test_compressibility_propagates_to_berendsen():
    """A stiffer compressibility (smaller 1/B) should reduce the
    Berendsen compressibility_au by the corresponding factor."""
    atoms = _make_atoms()
    dyn_soft = build_md_dynamics(
        atoms, ensemble="NPT", T=300.0, timestep=0.5,
        npt_method="berendsen", compressibility_GPa=100.0,
    )
    atoms2 = _make_atoms()
    dyn_stiff = build_md_dynamics(
        atoms2, ensemble="NPT", T=300.0, timestep=0.5,
        npt_method="berendsen", compressibility_GPa=200.0,
    )
    # compressibility_au scales as 1/compressibility_GPa, so stiff = soft/2.
    assert dyn_stiff.compressibility == pytest.approx(0.5 * dyn_soft.compressibility)


def test_taup_factor_propagates_to_mtk():
    """taup_factor should also drive MTK's barostat damping time."""
    atoms = _make_atoms()
    dyn_default = build_md_dynamics(
        atoms, ensemble="NPT", T=300.0, timestep=0.5, ttime=25.0,
        npt_method="mtk",
    )
    atoms2 = _make_atoms()
    dyn_slow = build_md_dynamics(
        atoms2, ensemble="NPT", T=300.0, timestep=0.5, ttime=25.0,
        npt_method="mtk", taup_factor=30.0,
    )
    # IsotropicMTKBarostat stores _pdamp on the internal _barostat
    # object (no public accessor).  Slow should be 3x larger.
    assert dyn_slow._barostat._pdamp == pytest.approx(
        3.0 * dyn_default._barostat._pdamp
    )


@pytest.mark.parametrize("method", ["berendsen", "mtk", "parrinello-rahman"])
def test_taup_factor_5_step_run(method):
    """Each method should still integrate stably with a non-default
    taup_factor."""
    atoms = _make_atoms()
    dyn = build_md_dynamics(
        atoms, ensemble="NPT", T=300.0, timestep=0.5, ttime=25.0,
        npt_method=method, taup_factor=30.0, compressibility_GPa=200.0,
    )
    dyn.run(5)
    pos = atoms.get_positions()
    assert pos.shape == (len(atoms), 3)
    assert (abs(pos) < 100.0).all(), f"{method}: atoms wandered far"


def test_yaml_taup_and_compressibility_accepted(tmp_path):
    """YAML config with valid taup_factor and compressibility_GPa loads."""
    from amorphgen.configs.yaml_config import load_yaml_config

    yaml_text = """
melt:
  ensemble: NPT
  npt_method: berendsen
  T_start: 300
  T_end: 3000
  taup_factor: 30.0
  compressibility_GPa: 200.0
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text)
    cfg = load_yaml_config(str(p))
    assert cfg["melt"]["taup_factor"] == 30.0
    assert cfg["melt"]["compressibility_GPa"] == 200.0


def test_yaml_negative_taup_rejected(tmp_path):
    """taup_factor must be positive."""
    from amorphgen.configs.yaml_config import load_yaml_config

    yaml_text = """
melt:
  ensemble: NPT
  npt_method: berendsen
  T_start: 300
  T_end: 3000
  taup_factor: -5.0
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError, match="Invalid YAML config"):
        load_yaml_config(str(p))
