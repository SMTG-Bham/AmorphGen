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
        "steps":     100000,   # 50 ps at 0.5 fs timestep
        "timestep":  0.5,
        "friction":  0.01,
    },

    # ── Stage 3: melt (heating ramp) ─────────────────────────────────────────
    "melt": {
        "ensemble":     "NPT",
        # NPT integrator. "berendsen" is the default for ramp stability.
        # Switch to "mtk" or "parrinello-rahman" if you need true canonical
        # fluctuations (NB: these can become unstable during rapid ramps).
        "npt_method":   "berendsen",
        "T_start":      300,     # K
        "T_end":        3000,    # K
        "T_step":       100,     # K  per ramp segment
        "steps_per_T":  2000,
        "rate":         None,    # K/ps  (overrides steps_per_T if set)
        "timestep":     0.5,     # fs
        "friction":     0.01,    # 1/fs  (NVT Langevin)
        "ttime":        25.0,    # fs    (NPT thermostat)
        # Barostat coupling: taup = taup_factor * ttime.  Larger ->
        # slower, more stable barostat (recommended for the ramp).
        "taup_factor":  10.0,
        # Reference compressibility for Berendsen.  100 GPa is soft /
        # liquid-like; oxides with bulk modulus 150-300 GPa benefit
        # from 200 GPa here for less aggressive volume control.
        "compressibility_GPa": 100.0,
    },

    # ── Stage 4: high-temperature equilibration ──────────────────────────────
    "eq_high": {
        # NPT with Nose-Hoover-chain (MTK) for true canonical
        # fluctuations at the equilibration plateau.  Stage 3 has
        # already expanded the cell to the melt density; this stage
        # lets it fluctuate physically around the equilibrium volume.
        # Users who want the previous behaviour can set ensemble: NVT.
        "ensemble":     "NPT",
        "npt_method":   "mtk",
        "T":            3000,    # K  (should match melt T_end)
        "steps":        20000,
        "timestep":     0.5,
        "friction":     0.01,
        "ttime":        25.0,
        "taup_factor":  10.0,
        "compressibility_GPa": 100.0,
    },

    # ── Stage 5: quench (cooling ramp) ───────────────────────────────────────
    "quench": {
        "ensemble":     "NVT",
        "T_start":      3000,    # K  (should match eq_high T)
        "T_end":        300,     # K
        "T_step":       -100,    # K  (negative = cooling)
        "steps_per_T":  2000,
        "rate":         None,    # K/ps  (overrides steps_per_T if set)
        "timestep":     0.5,
        "friction":     0.01,
        "ttime":        25.0,
    },

    # ── Stage 6: low-temperature equilibration ───────────────────────────────
    "eq_low": {
        "ensemble":  "NVT",
        "T":         300,      # K  (should match quench T_end)
        "steps":     20000,
        "timestep":  0.5,
        "friction":  0.01,
    },
}
