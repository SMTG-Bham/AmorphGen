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

## v1.0.0rc3 (2026-08-03)

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
- **`--resume` support for `--random-gen`**: skips completed structures on disk and continues from the first missing index. Validates files are non-empty and ASE-readable. Writes `run_metadata.json` and warns if composition changes between runs.
- Calculator pre-warm for `--random-gen --relax`: model load + first-inference happen once before the loop, so per-structure timing reflects only relax cost, not setup.
- Per-structure wall-time logging in `--random-gen --relax`: each structure's log shows `Wall time: X.XX s (N steps, Y s/step)` for diagnosing slowdowns.

### Fixed

- Auto-RDF cutoff on unrelaxed structures. `auto_cutoff_rdf` took the first
  bump of g(r) above a fixed threshold as the first peak, which on unrelaxed
  random placements latched onto noise on the rising edge and returned a cutoff
  *below* the bond length (Si–O CN ≈ 0.2 in `--analyse` of raw random-gen
  output). The first peak is now the first local maximum at ≥ 50 % of the
  strongest feature, and the first minimum must be a genuine depletion
  (g ≤ half the peak). Cutoffs on relaxed structures are unchanged; when no
  minimum exists the radii-table cutoff is used with a warning.

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

