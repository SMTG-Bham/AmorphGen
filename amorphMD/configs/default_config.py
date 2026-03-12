"""
default_config.py
-----------------
Central configuration for the 7-stage melt-and-quench pipeline.

Stage 1 : Optimise crystalline input cell (LBFGS + UnitCellFilter)
Stage 2 : Pre-melt equilibration at 300 K  (NVT, 50 ps)
Stage 3 : Melt  –  NPT heat ramp  300 K → T_melt  (default 2500 K, 100 K/ps)
Stage 4 : High-T equilibration at T_melt  (NVT, 100 ps)
Stage 5 : Quench  –  NVT cooling ramp  T_melt → 300 K  (100 K/ps)
Stage 6 : Low-T equilibration at 300 K  (NVT, 50 ps)
Stage 7 : Final optimisation  →  amorphous structure output

Override any value by passing cfg_override to MeltQuenchPipeline
or directly to individual stage run() functions.
"""

DEFAULT_CONFIG = {

    # ── MACE calculator ───────────────────────────────────────────────────────
    # Foundation model short-name (see utils/common.py → MACE_FOUNDATION_MODELS)
    "mace_model": "mace-mpa-0",

    # Set model_path to a local .model file to use a fine-tuned model.
    # model_path always takes priority over mace_model.
    "model_path": None,

    # Compute device: "cuda", "cpu", or "auto" (detects GPU automatically)
    "device": "auto",

    # ── Trajectory format (all MD stages) ────────────────────────────────────
    # "extxyz"      : Extended XYZ (default) – stores cell + per-atom properties
    #                 in the comment line; readable by OVITO, VESTA, ASE.
    # "xyz"         : Plain XYZ (no cell info)
    # "traj"        : ASE binary .traj
    # "lammps-dump" : LAMMPS text dump
    "traj_format": "extxyz",

    # ── Stage 1 : Crystalline structure optimisation ──────────────────────────
    "opt": {
        "fmax":         0.01,        # eV/Å convergence criterion
        "max_steps":    1000,
        "fix_symmetry": True,        # preserve crystal space group during relaxation
        # Optimizer: "LBFGS" (default), "FIRE", "BFGSLineSearch", "BFGS", "MDMin"
        # LBFGS  – fastest for well-behaved systems
        # FIRE   – more robust for disordered/amorphous structures
        "optimizer":    "LBFGS",
        # Cell filter: "UnitCellFilter" (default), "ExpCellFilter",
        #              "StrainFilter", "cubic" (reshape + fix angles)
        "cell_filter":  "UnitCellFilter",
        "logfile":      "opt_stage1.log",
        "traj_file":    "opt_stage1.traj",
        "output_cif":   "stage1_optimised.cif",
        "output_xyz":   "stage1_optimised.xyz",
    },

    # ── Stage 2 : Pre-melt equilibration at 300 K ────────────────────────────
    # Equilibrate the crystalline cell at room temperature before heating.
    # Allows the cell to relax to the MLIP potential surface before melting.
    "eq_premelt": {
        "ensemble":        "NVT",    # "NVT" or "NPT"
        "temperature_K":   300,
        "steps":           50_000,   # 50 ps at 1 fs timestep
        "timestep_fs":     1.0,
        "taut_fs":         100,
        "pressure_bar":    1.0,      # NPT only
        "taup_fs":         1000,     # NPT only
        "compressibility": 4.5e-5,   # NPT only
        "log_interval":    10,
        "traj_file":       "stage2_eq_premelt",
        "log_file":        "stage2_eq_premelt_log.txt",
        "output_cif":      "stage2_eq_premelt.cif",
        "output_xyz":      "stage2_eq_premelt.xyz",
    },

    # ── Stage 3 : Melt (heat ramp) ────────────────────────────────────────────
    # NPT ensemble recommended — the cell volume expands freely to the melt
    # density (e.g. In2O3: 4.2 g/cm³ → ~3 g/cm³ at 2500 K).
    #
    # Heating rate = T_step / (steps_per_T * timestep_fs * 1e-3) K/ps
    # Default: 100 K / (1000 steps × 0.001 ps) = 100 K/ps
    #
    # Alternative: set "rate_K_per_ps" to override T_step + steps_per_T.
    "melt": {
        "ensemble":        "NPT",
        "T_start":         300,      # K
        "T_end":           2500,     # K  —  match your system's melting point
        "T_step":          100,      # K per ramp step
        "steps_per_T":     1000,     # MD steps at each T (1 ps at 1 fs dt)
        "timestep_fs":     1.0,
        # rate_K_per_ps: None,       # uncomment to use rate-based API instead
        "taut_fs":         100,
        "pressure_bar":    1.0,
        "taup_fs":         1000,
        "compressibility": 4.5e-5,
        "make_cubic":      True,     # reshape supercell to cube before melting
        "log_interval":    10,
        "traj_file":       "stage3_melt",
        "log_file":        "stage3_melt_log.txt",
        "output_cif":      "stage3_melted.cif",
        "output_xyz":      "stage3_melted.xyz",
    },

    # ── Stage 4 : High-T equilibration at T_melt ─────────────────────────────
    # NVT run at T_melt to fully disorder the liquid structure.
    # For batch quench: set sample_interval_ps to save decorrelated snapshots.
    "eq_high": {
        "ensemble":           "NVT",
        "temperature_K":      2500,  # K — must match melt T_end
        "steps":              100_000,  # 100 ps at 1 fs timestep
        "timestep_fs":        1.0,
        "taut_fs":            100,
        "pressure_bar":       1.0,   # NPT only
        "taup_fs":            1000,  # NPT only
        "compressibility":    4.5e-5,
        "log_interval":       10,
        "density_log_interval": 1000,  # print density every N steps
        # ── Cell control ─────────────────────────────────────────────────────
        # "free"           : cell evolves freely with ensemble (default)
        # "fix_volume"     : freeze cell — pure NVT, no barostat
        # "keep_cubic"     : fix angles at 90°, volume free (NVT only)
        # "target_density" : rescale cell to target_density_g_cm3 before MD
        "cell_mode":          "free",
        "target_density_g_cm3": None,  # e.g. 3.0 for amorphous In2O3
        # ── Snapshot sampling for batch quench ───────────────────────────────
        "sample_interval_ps": None,  # e.g. 10.0 ps  —  None = disabled
        "snapshot_dir":       "snapshots",
        "traj_file":          "stage4_eq_high",
        "log_file":           "stage4_eq_high_log.txt",
        "output_cif":         "stage4_eq_high.cif",
        "output_xyz":         "stage4_eq_high.xyz",
    },

    # ── Stage 5 : Quench (cooling ramp) ──────────────────────────────────────
    # NVT cooling ramp from T_melt back to 300 K.
    # NVT avoids barostat artefacts during rapid cooling.
    #
    # Cooling rate = |T_step| / (steps_per_T * timestep_fs * 1e-3) K/ps
    # Default: 100 K/ps  (same rate as heating for In2O3 protocol)
    "quench": {
        "ensemble":        "NVT",
        "T_start":         2500,     # K — must match eq_high temperature_K
        "T_end":           300,      # K
        "T_step":          -100,     # negative → cooling
        "steps_per_T":     1000,     # 1 ps per step
        "timestep_fs":     1.0,
        # rate_K_per_ps: None,       # uncomment to use rate-based API
        "taut_fs":         100,
        "pressure_bar":    1.0,      # NPT only
        "taup_fs":         1000,     # NPT only
        "compressibility": 4.5e-5,   # NPT only
        "log_interval":    10,
        "traj_file":       "stage5_quench",
        "log_file":        "stage5_quench_log.txt",
        "output_cif":      "stage5_quenched.cif",
        "output_xyz":      "stage5_quenched.xyz",
    },

    # ── Stage 6 : Low-T equilibration at 300 K ───────────────────────────────
    # NVT equilibration at room temperature after quenching.
    "eq_low": {
        "ensemble":        "NVT",
        "temperature_K":   300,
        "steps":           50_000,   # 50 ps at 1 fs timestep
        "timestep_fs":     1.0,
        "taut_fs":         100,
        "pressure_bar":    1.0,      # NPT only
        "taup_fs":         1000,     # NPT only
        "compressibility": 4.5e-5,   # NPT only
        "log_interval":    10,
        "traj_file":       "stage6_eq_low",
        "log_file":        "stage6_eq_low_log.txt",
        "output_cif":      "stage6_eq_low.cif",
        "output_xyz":      "stage6_eq_low.xyz",
    },

    # ── Stage 7 : Final optimisation (amorphous output) ──────────────────────
    "final_opt": {
        "fmax":         0.01,
        "max_steps":    1000,
        "fix_symmetry": False,       # no symmetry constraint for amorphous
        # FIRE is often better than LBFGS for amorphous structures
        "optimizer":    "LBFGS",
        # Use "cubic" to enforce orthogonal cell for amorphous structure
        # Use "ExpCellFilter" for better convergence of soft amorphous cells
        "cell_filter":  "UnitCellFilter",
        "logfile":      "opt_stage7.log",
        "traj_file":    "opt_stage7.traj",
        "output_cif":   "stage7_amorphous_final.cif",
        "output_xyz":   "stage7_amorphous_final.xyz",
    },
}
