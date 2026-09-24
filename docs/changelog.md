---
orphan: true
---

# Changelog

## v1.0.0rc2 (2026-05-22)

### Changed (breaking)

- **`--random-gen` output layout.** Initial structures are now written to
  `<work_dir>/random_initial/` and relaxed structures to
  `<work_dir>/random_opt/` (was: both flat in `<work_dir>/`). This makes
  `amorphgen --analyse --input-dir <work_dir>/random_opt/` work without
  any `*_opt.vasp` filtering. The per-structure `random_NNNN_opt.log`
  file moved into `random_opt/` alongside its structure.
- Default analysis cutoff changed from `"auto"` (minsep-based) to
  `"auto-rdf"` (first-RDF-minimum). This is the standard convention in
  neutron-diffraction analysis of glasses, and avoids systematically
  under-counting coordination for materials with broad first-shell
  distributions.
- **`--default-dtype` default** changed from `"float64"` to `"auto"`,
  which resolves per-backend: `float32` for CHGNet (its only supported
  dtype) and classical potentials; `float64` for MACE and SevenNet.
  Previously, users running CHGNet had to remember to pass
  `--default-dtype float32` explicitly, otherwise the run crashed with
  ``NotImplementedError`` from `_load_chgnet`. Explicit `float32` or
  `float64` flags continue to work as before.
- **`--dtype` is the new short form** of `--default-dtype`.
  ``--default-dtype`` is still accepted as a legacy alias, so existing
  scripts and YAML configs (which use the underlying ``default_dtype``
  key) are unaffected.
- Single-snapshot `--batch-quench` / `--hybrid-ensemble` runs
  no longer nest an extra ``run_0000/`` subdirectory. When the work-dir
  contains exactly one snapshot (typical of SLURM array workflows where
  each task processes a single input), outputs land directly under
  ``work_dir/`` instead of ``work_dir/run_0000/``. Multi-snapshot runs
  still write to ``work_dir/run_NNNN/`` per run.
  - Old layout: ``hybrid_runs/task_0000/run_0000/final_amorphous.xyz``
  - New layout: ``hybrid_runs/task_0000/final_amorphous.xyz``

### Added

- **`weighting` parameter on `structure_factor()`**: ``"unweighted"``
  (default, current behaviour, a single FFT of the all-atom g(r)).
- **`amorphgen.analysis.compare_ensembles()`** and `EnsembleSpec`:
  multi-ensemble overlay plots (RDF, coordination, bond angles, density)
  with one call. Used by the new Validation docs page.
- Per-structure density violin: in `--analyse --save-plot`: new
  `analysis_density.{png,pdf,csv}` output alongside the existing RDF /
  CN / angles plots.
- Validation docs page (`/validation/`) with four sub-tabs
  (a-Ga₂O₃, a-SiO₂, a-HfO₂, a-Si). Each tab includes results-vs-reference
  table, structure render, validation figure, and reproduce-it
  commands.

### Fixed

- Spurious M-M-M triplets in bond-angle analysis of binary oxides.
  In a multi-element compound with at least one ionic pair (e.g. HfO₂),
  same-element metallic pairs (Hf-Hf) are second-shell contacts
  mediated by the anion, not real first-shell bonds; they are now
  excluded from `compute_all_angles`. Pure-metal alloy systems
  (NiTi, CuZr) keep their X-X first-shell bonds as before.
- Per-structure E/atom column: in `--analyse --per-structure` was
  always `N/A` for structure files that don't carry energy in their
  header (VASP, CIF). The analyser now falls back to parsing the
  sibling `random_gen.log`.

## v1.0.0rc3 (2026-09-22)

### Added

- MLIP-optional install: torch is no longer a core dependency, the base
  `pip install` is lightweight (random-gen + analysis + classical potentials),
  and MACE/CHGNet/SevenNet arrive only via extras (`[mace]`, `[chgnet]`,
  `[sevennet]`). With no torch present, `--device auto` resolves to CPU and
  calculator-requiring commands fail fast via `require_backend()` with the
  exact install line. `--list-models` shows every model with installed/missing
  markers.
- Numerical-divergence guard. MD and relaxation now raise a clear
  `DivergenceError` (via `assert_finite`) the moment an energy or force turns
  non-finite (before a NaN/Inf frame reaches disk) with an actionable
  message (the MLIP is out-of-distribution at high T, or the timestep is too
  large). Guards the most common high-T melt-quench failure mode of universal
  MLIPs.
- **`--retry-mode {expand, reduce-minsep, none}`**, placement-stall policy for
  `--random-gen`: `expand` (default; grow the cell 5% per retry, right when the
  density is an estimate), `reduce-minsep` (hold the cell/density *exactly*
  fixed and soften only non-bonded minseps, for fixed-density film / isochoric
  studies), or `none` (no adjustment; a stall raises). Cation–anion bond
  minseps are never reduced.
- Min-CN floor (default). Every atom gets a hard coordination floor (auto
  anions = 2, cations = 3, each capped at the element's target CN) and a
  post-placement `_repair_min_cn()` pass relocates below-floor atoms, cutting
  dangling bonds (IrO₂ dangling-O ~22% → ~3% at placement, < 1% after
  relaxation).
- Frame-level MD resume: `--resume` now continues an interrupted MD stage
  (2–6) from the last frame of its trajectory (momenta carried), on top of the
  existing stage-level skip.
- Oxyhalide material class (e.g. BiOCl, NaTaOCl₄): packing factor
  interpolated by halogen fraction between metal-oxide and halide, with a 10%
  dopant gate so trace halogens (F-doped TiO₂ / FTO) keep their oxide routing.
- Homonuclear dimer detection (`--check-dimers`): flags peroxide-type
  O–O and other same-element close pairs, skipping metal self-pairs when anions
  are present. Plus `--sq` / `--sq-weighting` to expose the direct S(q) method
  on the CLI.
- **`--mq-ensemble`** mode: full melt-quench ensemble in one CLI command. Stages 1-4 from a crystalline input, then N independent quenches via auto-extracted snapshots from the stage-4 trajectory, collected to `final/`.
- **`--hybrid-ensemble`** mode: take a directory of disordered structures and run stages 4-5-6-7 on each.
- **`--rank-from-log`** mode: parse a random-gen log and rank structures by total energy (no calculator re-evaluation needed; works for VASP/CIF outputs that don't carry per-atom energy).
- **`--extract-snapshots`** mode: utility CLI to extract N uniformly-spaced frames from any trajectory file.
- **`--reference YAML`** flag for `--analyse`: validate computed structural metrics against literature ranges defined in a reference YAML; produces a match/concern/fail verdict per metric.
- Polymorphic `--snapshot-dir`: for `--batch-quench`: accepts either a directory of static structures or a single trajectory file (auto-extracts internally).
- SevenNet backend: integrated via the `sevenn` package. Supports the multi-fidelity foundation models (`7net-mf-ompa`, `7net-l3i5`, `7net-omat`, `7net-0`, ...) with automatic `modal` selection for multi-fidelity variants.
- Publication-quality plotting: `--save-pdf` (vector PDF), `--dpi N`, `--show-title`, Okabe-Ito colour-blind-safe palette, clean spines, proper unit symbols (Å, °).
- **`--resume` support for `--random-gen`**: skips completed structures on disk and continues from the first missing index. Validates files are non-empty and ASE-readable. Writes `run_metadata.json` and warns if composition changes between runs.
- Calculator pre-warm for `--random-gen --relax`: model load + first-inference happen once before the loop, so per-structure timing reflects only relax cost, not setup.
- Per-structure wall-time logging in `--random-gen --relax`: each structure's log shows `Wall time: X.XX s (N steps, Y s/step)` for diagnosing slowdowns.

### Fixed

- **Buckingham+Coulomb: Ewald summation replaces the Wolf sum.** The Wolf
  (damped-shifted) Coulomb sum used by `BuckinghamCalculator` was checked
  against a reference Ewald sum on a 576-atom GeO2 melt and found to be ~10 %
  off in forces and ~100 meV/atom off in energy differences at its default
  alpha = 0.2, rc = 10 A, with no setting that fixes it inside a 20 A box. The
  Coulomb term is now a full Ewald sum (real-space over the pair cutoff with
  alpha = 3.5/rc, NumPy reciprocal-space sum, self term); it reproduces the
  NaCl Madelung energy to 4 decimals and a reference Ewald to 0.02 meV/atom.
  Atom-self-image pairs (cutoff > L/2) are now counted. `coulomb_method: wolf`
  keeps the old scheme for comparison.

- Auto-RDF cutoff on unrelaxed structures. `auto_cutoff_rdf` took the first
  bump of g(r) above a fixed threshold as the first peak, which on unrelaxed
  random placements latched onto noise on the rising edge and returned a cutoff
  *below* the bond length (Si–O CN ≈ 0.2 in `--analyse` of raw random-gen
  output). The first peak is now the first local maximum at ≥ 50 % of the
  strongest feature, and the first minimum must be a genuine depletion
  (g ≤ half the peak). Cutoffs on relaxed structures are unchanged; when no
  minimum exists the radii-table cutoff is used with a warning.

- Critical: `batch_quench.py` stage-numbering bug: the dispatch loop used the old 6-stage numbering (`if s==4: quench; s==5: eq_low; s==6: final_opt`) instead of the canonical 7-stage numbering (`s==4: eq_high; s==5: quench; s==6: eq_low; s==7: final_opt`). With the CLI default of `--batch-stages 5 6 7`, this caused the controlled cooling step to be silently skipped, runs did NVT-eq-at-300K + final-opt instead of quench + eq_low + final_opt. **Re-run any batch-quench output produced before this fix if methodology accuracy matters (e.g. publication).** Unknown stage numbers now raise `ValueError` instead of silent skip.
- Resume bug in equilibrate stages (2, 4, 6): the trajectory file and the stage's final-output checkpoint shared the same default name `stage{N}_eq.xyz`. This had two effects: (1) successful runs overwrote the trajectory data with a single-frame final state, losing trajectory history; (2) interrupted runs left a partial trajectory file at the checkpoint location, causing `--resume` to wrongly skip the stage. Fixed by splitting the defaults: trajectory → `stage{N}_eq_traj.xyz`, final output → `stage{N}_eq.xyz`. **Pre-fix `stage{N}_eq.xyz` files are ambiguous and should be deleted before resuming.**
- **`--extract-snapshots` now honours `--format`.** Previously the mode was hardcoded to write extxyz `.xyz` files regardless of the `--format` flag, which silently ignored `--format vasp` and `--format cif`. Now writes the correct format with the correct extension (POSCAR-style with `sort=True` for `vasp`).
- **`--extract-snapshots` count flag unified.** Both `-n` / `--n-structures` (the standard count flag used everywhere else) and the legacy `--n-runs` now control the snapshot count. `--n-runs` is preserved for backwards compatibility with existing scripts.
- Critical: FT structure factor `S(q)` had two errors: (1) `compute_structure_factor` integrated `r·(g−1)·sinc(qr)` instead of the 3D isotropic `r²·(g−1)·sinc(qr)` (one factor of `r` short. (2) The *partial* S(q) used the partner-species density `n_b/V` in the transform prefactor instead of the total number density ρ₀ that the Faber-Ziman definition requires, scaling every partial by `c_b`; since the x-ray/neutron weighted totals are built from these partials, they were off by a composition-dependent factor (exactly 0.5 for a 50/50 binary) with equal scattering lengths the weighted total must equal the unweighted one, and it didn't). Both fixed; the equal-scattering-length identity is now a regression test. `compute_structure_factor_direct` was unaffected by either and remains the recommended method for the FSDP. **Re-generate any S(q) produced via the FT method.**
- Same-element partial `g(r)` was a factor of 2 too low. `_compute_partial_rdf_frame` (used by `plot_rdf_time_windows`) counted undirected same-element pairs (`i<j`) but normalised with the directed pair-density, so A–A partials asymptoted to ~0.5 instead of 1. Now counts both directions, matching `analysis.rdf.compute_rdf`.
- Trajectory-based time axes (and fitted diffusion coefficient) off by the trajectory stride: `compute_msd`/`plot_msd` (and the other trajectory-fed diagnostics (`plot_energy_convergence`, `plot_temperature`, `plot_block_averages`, `plot_rdf_time_windows`, `compute_cn_vs_time`/`plot_cn_vs_time`, `convergence_report`)) assumed one MD step per frame, but AmorphGen writes one frame per `TRAJ_LOG_INTERVAL` (100) steps. The time axis was 100× too short (and `D` 100× too large, which could misclassify a frozen system as liquid); running-average windows were likewise 100× too wide. All now take a `frame_stride` parameter (default `TRAJ_LOG_INTERVAL`). **Behaviour change:** on a *trajectory* input these functions now interpret the frame spacing as `timestep_fs × frame_stride`; a script analysing a non-AmorphGen trajectory that stores every step must pass `frame_stride=1` to keep the old time axis. Log-file inputs (which carry a real time column) are unchanged.
- Melt/quench temperature ramps hardened. The melt ramp used `range()` (crashed on a float `T_step`, and overshot the endpoint on a non-divisible span); the quench ramp used a `while` loop with no guard against a zero or mis-signed `T_step` (infinite loop) and dropped the endpoint on a non-divisible span. Both now use `resolve_ramp`, which infers direction from the endpoints, supports float steps, always lands exactly on `T_end`, and never overshoots.
- Classical calculators + variable-cell now fail clearly: Lennard-Jones and Buckingham implement only energy+forces (no stress), so an NPT stage (or a cell-filter relaxation, including the default `cell_filter='cubic'` of `--random-gen --relax`) used to crash with an opaque ASE `PropertyNotImplementedError`. A capability guard now raises an actionable error up front (use a stress-capable MLIP, or a fixed cell + NVT).
- Melt/quench ramp is validated before any file is touched. The ramp schedule (including the zero-step check) is now resolved *before* `attach_outputs` opens the log/trajectory, so a bad `T_step` raises without first truncating an existing trajectory.
- Diagnostic time axes default to the pipeline timestep. The trajectory diagnostics in `utils.equilibration` defaulted `timestep_fs=1.0` while `DEFAULT_CONFIG` runs at 0.5 fs, giving a 2× time axis (and D/2) on default-config trajectories analysed with library defaults. The default is now sourced from `DEFAULT_CONFIG` and the docstring corrected. Always pass your run's actual timestep if it differs.
- Three neutron scattering lengths corrected. `_NEUTRON_B` (the `weighting="neutron"` table) was checked entry-by-entry against the printed Sears (1992) Table 1. Fifty-two of fifty-five matched; three did not and are now the Sears values: **Cd 5.1 → 4.87**, **W 4.755 → 4.86**, **Au 7.90 → 7.63** fm (2–5% errors). Neutron-weighted S(q) for systems containing these elements changes accordingly; all other elements, and all x-ray/unweighted results, are unaffected. The x-ray form-factor table was likewise spot-checked digit-for-digit against the printed Waasmaier–Kirfel table (N, O, F, Ni, Cu, Zn, Ga, Ge, As); two last-digit transcription differences (Ni b₂, Cu b₁, effect on f(q) ≈ 5×10⁻⁶) were aligned to the print. Both tables are now pinned to their printed sources by a regression test.
- **`--analyse --save-report` no longer aborts when the report's folder doesn't exist yet.** `save_report` now creates the parent directory, matching `--save-plot`. Previously the run crashed with `FileNotFoundError` *after* the structural summary but *before* S(q), the reference validation and the plots were written.
- **`--random-gen --relax` honours `opt: cell_filter` from YAML.** Cell-filter precedence is now CLI > `random_gen:` > explicit `opt:` > `cubic`. The shipped `example_classical.yaml` sets `cell_filter: none` under `opt:` (classical potentials have no stress); previously that was ignored for random-gen, so the new stress guard told users to set a value their YAML already contained. The guard's message now also names where the setting goes (`-C none`, or `cell_filter: none` under `opt:`/`random_gen:`).

### Added

- **`--sq-method {direct,ft}`** for `--analyse --sq` (default `direct`; also `sq_method:` in the YAML `analyse:` block). `direct` is the Debye sum at reciprocal-lattice q-vectors (no real-space truncation; resolves the FSDP; CSV includes `n_per_bin`). `ft` is the Fourier transform of g(r) truncated at L/2, smoother, but it damps and shifts the FSDP (a-Ga₂O₃: first peak 2.09 → 1.59, 2nd/1st ratio 0.84 → 1.17 vs 0.86 measured), so it is offered for comparison with FT-based codes rather than as the default; the CLI prints a truncation note when it is used. Both methods are Faber-Ziman weighted.
- **`--sq-smooth SIGMA_Q`** (and `sigma_q=` on `compute_structure_factor_direct` / `StructureAnalyser.structure_factor_direct`; `sq_smooth:` in YAML). Gaussian re-binning of the direct-method S(q) in q, **weighted by the number of q-vectors per shell**, statistically a wider, softer bin, so it reduces the low-q speckle noise (each reciprocal-lattice vector is one noisy sample; shells hold few of them at low q) without moving peaks. The CLI and plots apply **σ_q = 0.05 by default** (`DEFAULT_SQ_SMOOTH`; pass `--sq-smooth 0` for the raw shell averages), while the library functions default to 0 so API callers get the exact values; the raw values are always kept in the CSV as `s_q_raw`. The S(q) plot's y-axis now starts at 0 (the Faber–Ziman low-q dip below zero is clipped from the plot only). Keep it well below the FSDP width (~0.3 Å⁻¹): on the a-Ga₂O₃ ensemble σ_q = 0.05 halves the noise for a 5% first-peak cost (2nd/1st ratio 0.84 → 0.87 vs 0.86 measured), whereas ≥ 0.12 starts to damp the FSDP the way the FT method does.
- Partial structure factors from the direct method: `compute_structure_factor_direct(..., partials=True)` (and `StructureAnalyser.structure_factor_direct(partials=True)`) returns the Faber–Ziman partials S_ab(q) for every element pair from the same reciprocal-lattice sum as the total, so they resolve the FSDP (the FT partials are L/2-truncated). Per q-vector, S_ab = 1 + (Re⟨F_aF_b*⟩/√(N_aN_b) − δ_ab)/√(c_ac_b) with F_a = Σ_{i∈a} e^{iq·rᵢ}; the weighted sum Σ(2−δ_ab)c_ac_bf_af_bS_ab/⟨f⟩² rebuilds the total to machine precision (tested at 10⁻¹⁴). Enables partial-by-partial comparison with isotope-substitution neutron data (e.g. GeO₂, Salmon et al., Nature 435, 75 (2005)).
- Reference YAMLs: for a-SiO₂, a-GeO₂ and a-HfO₂ added to `examples/` (the a-HfO₂ ranges are deliberately broad, tighten against the model set you compare to).
- Custom-calculator injection into the full pipeline. `MeltQuenchPipeline(..., calc=<ase calculator>)` now runs every stage with a user-supplied ASE calculator, bypassing the `get_calculator()` backend factory (subject to the stress guard above for NPT / cell-filter stages).

### Changed

- Default RDF smearing is now σ = 0.05 Å (was 0, the raw histogram) for `compute_rdf`, `StructureAnalyser.rdf()`, the analysis plots/CSVs and the CLI `--smearing` flag, a single constant, `amorphgen.analysis.rdf.DEFAULT_SMEARING`. This is comparable to thermal/experimental broadening, so simulated g(r) compares naturally with diffraction data: peak *positions* are unchanged, heights drop and widths grow (a-Ga₂O₃ Ga–O peak 7.5 → 5.1, FWHM 0.125 → 0.203 Å). Pass `--smearing 0` / `sigma=0.0` for the raw histogram. **S(q) is unaffected**: both structure-factor methods always transform the raw g(r), since smearing would damp S(q) by exp(−q²σ²/2) (~16% at q = 12 Å⁻¹). Coordination cutoffs (`auto-rdf`) are also unaffected (they use their own histogram).
- X-ray S(q) now uses q-dependent atomic form factors: both `compute_structure_factor` (FT) and `compute_structure_factor_direct` (Debye) weighted x-ray totals with the constant atomic number Z. Real form factors f₀(q) fall off with q at element-specific rates (at q = 2.45 Å⁻¹, f/Z = 0.81 for Ga but 0.71 for O), so constant-Z systematically mis-weights the partials at finite q, for a-Ga₂O₃ it under-weights the Ga–Ga-dominated first peak by ~10%. `weighting="xray"` now uses the Waasmaier–Kirfel (1995) 5-Gaussian f₀(q) (98 elements, H–Cf; new public helper `xray_form_factor(symbol, q)`), with the Faber–Ziman weights and the Debye self-scattering offset evaluated per q. Validated: f₀(0) = Z for every element, and agreement with the International Tables for Crystallography Vol. C Cromer–Mann set to < 0.5% for oxide/semiconductor elements; the FT result was also cross-checked against an independent structure-factor code on the same structures (RMS 0.010, was 0.048 with constant Z). **X-ray S(q) values change** (neutron and unweighted are unaffected): for the PBE0 a-Ga₂O₃ ensemble the direct-method 2nd/1st peak-height ratio moves from 0.94 to 0.83, against 0.86 measured (Liu et al., Adv. Mater. 2023).
- Material classification expanded to 20 class-aware packing regimes, each
  drawing radii from the right source (Shannon ionic / Cordero covalent /
  Goldschmidt metallic) so a bare composition auto-derives a physical density
  and minseps. Adds a joint charge-balance oxidation-state solver for mixed
  cation/anion compounds and a high-valent d⁰ → CN=6 override; boride and
  oxyhalide packing factors calibrated against crystal densities.
- **`amorphgen.analysis`** package now exports `rank_from_log`, `format_log_ranking`, `validate_against_reference`, and `format_validation_report` (previously only `StructureAnalyser`).
- CLI help reorganised into argument groups (modes / calculator / optimisation / pipeline / random-gen / batch-quench / batch-opt / analyse) for readability. All flags continue to work; only the help-text layout changed.
- Default MD timestep in `DEFAULT_CONFIG` is now **0.5 fs** (was 1.0 fs). Safer for heavy elements and unusual chemistries. Shipped example YAMLs explicitly set 1.0 fs which is fine for typical oxides with chgnet/MACE foundation models.
- **`make_cubic` reshape moved from start of stage 3 (melt) to start of stage 4 (eq_high).** Reshaping a fully molten liquid is benign (atoms diffuse and lose memory of the deformation in <1 ps); reshaping a still-crystalline structure at the start of stage 3 caused a small unphysical jolt at low T. The flag is now read from the `eq_high:` block, falling back to `melt.make_cubic` for one release as a backwards-compat bridge. Default behaviour is unchanged (cubic reshape on by default).

## v1.0.0rc4 (2026-09-24)

### Added

- **Random-gen placement 4–10x faster.** Profiling a 600-atom SiO2 placement showed 96 %
  of the 39 s in the min-CN repair pass, a Python loop making 1.4 million small NumPy
  calls. Its candidate search is now batched (128 trial positions per distance
  evaluation) and an atom whose environment yields no candidate for 10 consecutive
  batches is treated as saturated for that pass instead of exhausting the 2500-draw
  budget. 600-atom SiO2: 33 s -> 9 s; 350-atom IGZO: unchanged 3.3 s; repair outcome
  within the seed-to-seed scatter of the old pass. **Behaviour change:** the random
  draws of the repair pass differ, so a given `seed` gives different (still
  reproducible) structures than rc3-era builds for compositions that need repair.

- **Batched MD through torch-sim** (`--hybrid-ensemble --engine torchsim`): stages 4–7 run
  for all structures at once (NVT-Langevin with per-step temperature schedules for the
  quench, then batched relaxation). Same per-run files and `final/` collection as the ASE
  path; NVT only; seeded. `amorphgen/utils/torchsim_md.py`, `batch_quench.run_torchsim`.
  Resume is run-level and frame-level: a chunk killed inside an MD stage continues
  from the last frame common to all its runs (momenta included), so a walltime kill
  costs at most 100 steps.

- **`--batch-size auto`** (the new default for the torch-sim engine): the chunk size is
  taken from a GPU memory probe of the first structure (a short relaxation or MD
  block), using about 40% (relaxation) or 50% (MD) of the card. An integer still fixes
  it; on CPU `auto` is 16. `torchsim_engine.estimate_batch_size()`.

- **`--indices SPEC`** (`80-90`, `0,5,7-9`) for `--random-gen` and `--batch-opt`, and
  `--pattern GLOB` for `--batch-opt`: generate or relax only selected structure
  indices. Seeds are index-derived, so a split ensemble is identical to a full run.

- **Optional torch-sim engine** (`pip install "amorphgen[torchsim]"`, `--engine torchsim`
  or `engine: torchsim`): `--batch-opt` and `--random-gen --relax` relax all structures in
  one batched torch-sim FIRE call (MACE, SevenNet, LJ; CUDA or CPU) instead of one after
  another through ASE; outputs are identical in name and format. CHGNet/Buckingham stay on
  the ASE engine. Python 3.12+ only; no Apple MPS. The optimiser (`-O`) carries over
  (LBFGS default) and, with a cell filter, convergence requires |P| < `pressure_tol_gpa`
  (default 0.02 GPa) as well as the force tolerance. Relaxation runs in chunks
  (`--batch-size`, default `auto`, see below) with outputs written per chunk, and `--resume` skips
  already-relaxed structures. The GPU cache is cleared between chunks and a CUDA
  out-of-memory error splits the chunk in half and retries instead of aborting the job.
- **`--analyse` counts each structure once** when a directory holds the same stem in
  several formats (the optimiser writes `.xyz` and `.cif`); priority xyz > extxyz > vasp > cif.

- **Global `seed` / `--seed`** for end-to-end reproducibility. Previously only the
  random placement was seeded; the velocity initialisation and the Langevin thermostat
  of stages 2–6 used ASE's unseeded generator, so two runs from the same seed gave
  different trajectories. A top-level `seed:` now derives a separate NumPy stream per
  stage and per run directory (SeedSequence) for both. Limits (GPU MLIP forces,
  frame-level resume) are documented in the YAML guide.

- **Polyhedral connectivity analysis** (`--connectivity`, `StructureAnalyser.polyhedral_connectivity()`,
  YAML `connectivity: true`): corner/edge/face sharing between cation-centred polyhedra
  and the fraction of cations in edge- or face-sharing pairs, the descriptor that
  separates a corner-sharing network glass from a random packing with the same
  short-range order. Validated on cristobalite (100 % corner) and rutile (2 edge + 8
  corner links per Ti).

- **`--rings [PAIR]` and `--voronoi [ELEMENT]`** for `--analyse` (YAML: `rings:`, `voronoi:`
  in the `analysis` block). Ring statistics and Voronoi indices were previously
  reachable only from YAML and only printed; they are now printed, appended to
  `--save-report`, and written as `analysis_rings.{csv,png}` / `analysis_voronoi.csv`.

- **`--version`** prints the installed AmorphGen version.

### Fixed

- **Review round 2 (~40 findings), the ones that changed results silently:**
  temperature ramps ran one extra segment at `T_start`, so the realised rate was
  n/(n+1) of the configured one (0.68× for T_step 1000); a negative `rate` collapsed a
  quench to one step per segment; the block-average equilibration test compared block
  means with the whole-run SEM (√n_blocks too strict); Gaussian-smoothed g(r) fell to
  ~0.5 at rmax because `np.convolve(mode='same')` zero-pads; bond angles in mixed-cation
  oxides included cation–cation contacts (Mg–Zn–O); bond counts were doubled; the
  direct S(q) could miss q-vectors in skewed cells; the Voronoi/temperature running
  window ignored the frame stride. All fixed.
- **Crashes / lost output:** the optimiser `.traj` was never written (manual `step()`
  loop bypassed ASE's observers); `traj_format: xyz` broke `--resume` and `lammps-dump`
  could not be written (formats are now `extxyz` and `traj`, `xyz` an alias); a torn
  last trajectory frame discarded the whole checkpoint (now truncated to the complete
  frames); `convert` overwrote `s.xyz`/`s.cif` → `s.vasp`; mixed-case model names
  (`MACE-MPA-0`) failed; Ewald raised an unhelpful error without a cell and had no
  neutralising background for non-neutral cells; LJ dropped self-image pairs beyond L/2;
  resumed MD logs restarted Step/Time at 0; a pair missing from a user cutoff dict fell
  back to the largest cutoff instead of "not bonded".
- **Config / CLI:** values typed with their default value were ignored in several modes
  (`-C FrechetCellFilter`, `-n 1` in `--extract-snapshots`, `-f 0.01` with `--relax`,
  `--format`, `--cutoff`, `--sq-weighting`); YAML `opt:` was dropped by
  `--hybrid-ensemble`/`--batch-opt` and `--random-gen --relax`; `1e-3` in YAML was read
  as a string; `--mq-ensemble`, `--hybrid-ensemble`, `--batch-quench` and
  `--extract-snapshots` all defaulted to `melt_quench_run/`; `_load_classical` mutated
  the caller's config.
- **Random-gen retry ladder:** the cell-expansion rungs are skipped (with a log line)
  when the density or cell length is fixed by the user, instead of silently doing
  nothing; a density cut by expansion is now logged as a WARNING with the percentage; a
  reduced minsep no longer carries over after a skipped structure; covalent
  nonmetal–nonmetal bonds (P–O, S–O, C–N) are no longer softened by `reduce-minsep`;
  atoms stay inside the box with `pbc=False`.

- **Ring statistics used the wrong nodes.** Auto-detection of the ring network
  sorted element symbols, so for SiO₂ it took O as the ring node and reported
  3-rings for cristobalite; single-element networks (a-Si) were routed through a
  bridging atom of the same species and also gave 3-rings. Nodes are now the
  least electronegative element (cation / network former) with the most
  electronegative as bridge, and single-element networks use the direct bonds:
  cristobalite and diamond both report 6-rings. **Re-run any ring statistics
  produced without an explicit `bond_pair`.**
- **Heating/cooling ramps crashed with `npt_method: mtk`** (`IsotropicMTKNPT`
  has no `set_temperature`) and with `parrinello-rahman` (the trajectory hook
  wrapped the live atoms between segments, which ASE's NPT integrator rejects).
  A `set_md_temperature` helper updates every integrator's target temperature,
  and the hook now writes a wrapped copy (with energy/forces) instead.
- **`--random-gen` dropped the automatic CN tolerance** (`target_cn, _ = ...`),
  so CLI and batch runs placed with tolerance 0 while the Python API used the
  class default (e.g. 2 for alloys) and failed placement more often.
- **Batch "reduce M-M minsep" rung used a CN-unaware table**, replacing the
  CN-aware minseps generate_random builds (Si–O 1.33 → 1.44 Å for SiO₂), so the
  step meant to ease placement made it harder. The batch path now builds the
  same CN-aware table.
- **Pipeline defaults differed with and without `--config`.** CLI parser
  defaults (eq_high NVT 10 000 steps, eq_premelt 50 000, eq_low 10 000)
  disagreed with `DEFAULT_CONFIG` (NPT/MTK 20 000, 100 000, 20 000), so a YAML
  containing only `model:` silently changed the protocol. Parser defaults now
  come from `DEFAULT_CONFIG` and only typed flags are passed as overrides in
  both modes. **Behaviour change:** bare-CLI runs now use the documented
  `DEFAULT_CONFIG` protocol (stage 4 NPT/MTK, longer equilibrations); pass the
  flags explicitly to keep the previous values.

- **`--random-gen` resume is now seed-stable.** Per-structure seeds are derived from the structure *index* (via `SeedSequence`) rather than a running counter that advanced on every attempt and was not advanced for skipped indices, so a `--resume` run now reproduces exactly the structures a fresh run would generate, while retries of a failed placement still draw fresh randomness. **Behaviour change:** the seed→structure mapping changed, so a given `seed` (YAML `random_gen: seed:` / API `seed=`) now produces *different* (but reproducible and resume-stable) structures than it did before this release. Regenerate rather than expecting old seeds to reproduce old structures.
