"""
amorphgen.pipeline.quench
--------------------------
Stage 5 – Cool the melt from T_start down to T_end via a temperature ramp.

Ensemble is configurable: NVT (default) or NPT.
"""

from __future__ import annotations

from ..utils import get_calculator
from .md_ramp import run_ramp


def run(atoms_or_file, cfg_override=None, calc=None, work_dir=None, **kwargs):
    """
    Cool the structure from T_start -> T_end.

    Parameters
    ----------
    atoms_or_file : str or ase.Atoms
    cfg_override : dict, optional
    calc : ASE calculator, optional
    work_dir : str or path-like, optional
        Directory for the log, trajectory and output structure, created if
        missing. Default: the current directory.

    Returns
    -------
    ase.Atoms — quenched structure at T_end
    """
    return run_ramp(atoms_or_file, cfg_override, calc, work_dir, stage=5,
                    calculator_factory=get_calculator, resume=kwargs.get("resume"))
