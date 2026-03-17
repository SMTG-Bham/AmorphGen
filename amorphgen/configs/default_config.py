"""
amorphgen.configs.default_config
---------------------------------
Central configuration for the melt-and-quench pipeline.

Override any value by passing ``cfg_override`` to
:class:`~amorphgen.pipeline.run_pipeline.MeltQuenchPipeline` or to
individual stage functions.
"""

DEFAULT_CONFIG = {

    # ── Calculator ────────────────────────────────────────────────────────────
    # Foundation model short name — works across all backends:
    #   MACE:     "mace-mpa-0", "mace-mh-1", "mace-omat-0-medium", ...
    #   CHGNet:   "chgnet"
    #   M3GNet:   "m3gnet"
    "model": "mace-mpa-0",

    # Legacy alias — reads are redirected to "model" in the pipeline
    "mace_model": None,

    # Path to a local .model file (overrides 'model' if set)
    "model_path": None,

    # Device: "cuda", "cpu", or "auto"
    "device": "auto",

    # ── Trajectory output format ──────────────────────────────────────────────
    # One of: "extxyz", "xyz", "traj", "lammps-dump"
    "traj_format": "extxyz",

    # ── Stage 1 & 7: structure optimisation ───────────────────────────────────
    "opt": {
        "fmax":      0.01,   # eV/Å  force convergence
        "max_steps": 1000,
    },

    # ── Stage 2: pre-melt equilibration ────────────────────────────────────
    "eq_premelt": {
        "ensemble":  "NVT",
        "T":         300,      # K
        "steps":     50000,    # 50 ps at 1 fs timestep
        "timestep":  1.0,
        "friction":  0.01,
    },

    # ── Stage 3: melt (heating ramp) ─────────────────────────────────────────
    "melt": {
        "ensemble":     "NPT",
        "T_start":      300,     # K
        "T_end":        3000,    # K
        "T_step":       100,     # K  per ramp segment
        "steps_per_T":  1000,
        "timestep":     1.0,     # fs
        "friction":     0.01,    # 1/fs  (NVT Langevin)
        "ttime":        25.0,    # fs    (NPT thermostat)
    },

    # ── Stage 4: high-temperature equilibration ──────────────────────────────
    "eq_high": {
        "ensemble":  "NVT",
        "T":         3000,     # K  (should match melt T_end)
        "steps":     10000,
        "timestep":  1.0,
        "friction":  0.01,
    },

    # ── Stage 5: quench (cooling ramp) ───────────────────────────────────────
    "quench": {
        "ensemble":     "NVT",
        "T_start":      3000,    # K  (should match eq_high T)
        "T_end":        300,     # K
        "T_step":       -100,    # K  (negative = cooling)
        "steps_per_T":  1000,
        "timestep":     1.0,
        "friction":     0.01,
        "ttime":        25.0,
    },

    # ── Stage 6: low-temperature equilibration ───────────────────────────────
    "eq_low": {
        "ensemble":  "NVT",
        "T":         300,      # K  (should match quench T_end)
        "steps":     10000,
        "timestep":  1.0,
        "friction":  0.01,
    },
}
