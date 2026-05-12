"""
amorphgen.pipeline.final_opt
------------------------------
Stage 7 – Final structural optimisation of the quenched amorphous structure.

Delegates to opt_cell.run() with stage_key="final_opt".
"""

from __future__ import annotations

from .opt_cell import run as _opt_run


def run(atoms_or_file, cfg_override=None, calc=None, **kwargs):
    """
    Final optimisation of the amorphous structure.

    Wraps opt_cell.run() — see that function for full parameter docs.
    """
    print("[Stage 7] Final optimisation (amorphous)")
    return _opt_run(atoms_or_file, cfg_override=cfg_override,
                    calc=calc, stage_key="final_opt", **kwargs)
