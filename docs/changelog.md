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
  - **Migration**: to restore the old flat layout, post-process with
    `mv <work_dir>/random_*/* <work_dir>/`.
- **Default analysis cutoff** changed from `"auto"` (minsep-based) to
  `"auto-rdf"` (first-RDF-minimum). This is the standard convention in
  neutron-diffraction analysis of glasses, and avoids systematically
  under-counting coordination for materials with broad first-shell
  distributions (a-Si, a-HfO₂, chalcogenides). Legacy `"auto"` is still
  accepted.
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
- **Single-snapshot `--batch-quench` / `--hybrid-ensemble` runs**
  no longer nest an extra ``run_0000/`` subdirectory. When the work-dir
  contains exactly one snapshot (typical of SLURM array workflows where
  each task processes a single input), outputs land directly under
  ``work_dir/`` instead of ``work_dir/run_0000/``. Multi-snapshot runs
  still write to ``work_dir/run_NNNN/`` per run.
  - **Old layout**: ``hybrid_runs/task_0000/run_0000/final_amorphous.xyz``
  - **New layout**: ``hybrid_runs/task_0000/final_amorphous.xyz``

### Added

- **`weighting` parameter on `structure_factor()`**: ``"unweighted"``
  (default, current behaviour — a single FFT of the all-atom g(r)),
  ``"xray"`` (Faber-Ziman partial sum with Z² weights), or
  ``"neutron"`` (same but with tabulated coherent scattering lengths
  for ~50 common elements). The X-ray weighting recovers the
  first-sharp-diffraction peak (FSDP) in amorphous oxides that
  cancels in the unweighted sum.
- **`structure_factor_direct()`** — new method that computes S(q)
  directly from atomic positions via the Debye formula evaluated at
  reciprocal-lattice q-vectors:
  $$ S(q) = \\frac{1}{N\\langle f\\rangle^2}\\left|\\sum_i f_i e^{i\\vec{q}\\cdot\\vec{r}_i}\\right|^2 $$
  Bypasses the rmax truncation that under-estimates peak intensities
  in the FT-of-g(r) method. For a-Ga₂O₃ the direct method gives FSDP
  intensity S(q=2.4)=2.0 vs FT-of-g(r) S(q=2.4)=0.84, matching the
  experimental value (Kaewmeechai et al., Phys. Rev. B 111, 035203,
  2025, Fig. S2b) and the GAP_500 simulation in the same reference.
  Slower than the FT method (~4-5× per ensemble); use for paper-
  quality S(Q) comparison with experiment.
- **`amorphgen.analysis.compare_ensembles()`** and `EnsembleSpec`:
  multi-ensemble overlay plots (RDF, coordination, bond angles, density)
  with one call. Used by the new Validation docs page.
- **Per-structure density violin** in `--analyse --save-plot`: new
  `analysis_density.{png,pdf,csv}` output alongside the existing RDF /
  CN / angles plots, matching the comparison-plot aesthetic.
- **Validation docs page** (`/validation/`) with four sub-tabs
  (a-Ga₂O₃, a-SiO₂, a-HfO₂, a-Si). Each tab includes results-vs-reference
  table, structure render, validation figure, and reproduce-it
  commands.

### Fixed

- **Spurious M-M-M triplets** in bond-angle analysis of binary oxides.
  In a multi-element compound with at least one ionic pair (e.g. HfO₂),
  same-element metallic pairs (Hf-Hf) are second-shell contacts
  mediated by the anion, not real first-shell bonds; they are now
  excluded from `compute_all_angles`. Pure-metal alloy systems
  (NiTi, CuZr) keep their X-X first-shell bonds as before.
- **Per-structure E/atom column** in `--analyse --per-structure` was
  always `N/A` for structure files that don't carry energy in their
  header (VASP, CIF). The analyser now falls back to parsing the
  sibling `random_gen.log` (using the existing `rank_from_log` parser)
  to fill in the E/atom column when the file-level lookup fails.

## v1.0.0rc3 (2026-08-03)

### Added

- **MLIP-optional install.** Torch is no longer a core dependency — the base
  `pip install` is lightweight (random-gen + analysis + classical potentials),
  and MACE/CHGNet/SevenNet arrive only via extras (`[mace]`, `[chgnet]`,
  `[sevennet]`). With no torch present, `--device auto` resolves to CPU and
  calculator-requiring commands fail fast via `require_backend()` with the
  exact install line. `--list-models` shows every model with installed/missing
  markers.
- **Numerical-divergence guard.** MD and relaxation now raise a clear
  `DivergenceError` (via `assert_finite`) the moment an energy or force turns
  non-finite — before a NaN/Inf frame reaches disk — with an actionable
  message (the MLIP is out-of-distribution at high T, or the timestep is too
  large). Guards the most common high-T melt-quench failure mode of universal
  MLIPs.
- **`--retry-mode {expand, reduce-minsep, none}`** — placement-stall policy for
  `--random-gen`: `expand` (default; grow the cell 5% per retry, right when the
  density is an estimate), `reduce-minsep` (hold the cell/density *exactly*
  fixed and soften only non-bonded minseps, for fixed-density film / isochoric
  studies), or `none` (no adjustment; a stall raises). Cation–anion bond
  minseps are never reduced.
- **Min-CN floor (default).** Every atom gets a hard coordination floor (auto
  anions = 2, cations = 3, each capped at the element's target CN) and a
  post-placement `_repair_min_cn()` pass relocates below-floor atoms — cutting
  dangling bonds (IrO₂ dangling-O ~22% → ~3% at placement, < 1% after
  relaxation).
- **Frame-level MD resume.** `--resume` now continues an interrupted MD stage
  (2–6) from the last frame of its trajectory (momenta carried), on top of the
  existing stage-level skip.
- **Oxyhalide material class** (e.g. BiOCl, NaTaOCl₄): packing factor
  interpolated by halogen fraction between metal-oxide and halide, with a 10%
  dopant gate so trace halogens (F-doped TiO₂ / FTO) keep their oxide routing.
- **Homonuclear dimer detection** (`--check-dimers`) — flags peroxide-type
  O–O and other same-element close pairs, skipping metal self-pairs when anions
  are present. Plus `--sq` / `--sq-weighting` to expose the direct S(q) method
  on the CLI.
- **`--mq-ensemble`** mode: full melt-quench ensemble in one CLI command. Stages 1-4 from a crystalline input, then N independent quenches via auto-extracted snapshots from the stage-4 trajectory, collected to `final/`.
- **`--hybrid-ensemble`** mode: take a directory of disordered structures and run stages 4-5-6-7 on each.
- **`--rank-from-log`** mode: parse a random-gen log and rank structures by total energy (no calculator re-evaluation needed; works for VASP/CIF outputs that don't carry per-atom energy).
- **`--extract-snapshots`** mode: utility CLI to extract N uniformly-spaced frames from any trajectory file.
- **`--reference YAML`** flag for `--analyse`: validate computed structural metrics against literature ranges defined in a reference YAML; produces a match/concern/fail verdict per metric.
- **Polymorphic `--snapshot-dir`** for `--batch-quench`: accepts either a directory of static structures or a single trajectory file (auto-extracts internally).
- **SevenNet backend**: integrated via the `sevenn` package. Supports the multi-fidelity foundation models (`7net-mf-ompa`, `7net-l3i5`, `7net-omat`, `7net-0`, ...) with automatic `modal` selection for multi-fidelity variants.
- **Publication-quality plotting**: `--save-pdf` (vector PDF), `--dpi N`, `--show-title`, Okabe-Ito colour-blind-safe palette, clean spines, proper unit symbols (Å, °).
- **`--resume` support for `--random-gen`**: skips completed structures on disk and continues from the first missing index. Validates files are non-empty and ASE-readable. Writes `run_metadata.json` and warns if composition changes between runs.
- **Calculator pre-warm** for `--random-gen --relax`: model load + first-inference happen once before the loop, so per-structure timing reflects only relax cost, not setup.
- **Per-structure wall-time logging** in `--random-gen --relax`: each structure's log shows `Wall time: X.XX s (N steps, Y s/step)` for diagnosing slowdowns.

### Fixed

- **Critical: `batch_quench.py` stage-numbering bug.** The dispatch loop used the old 6-stage numbering (`if s==4: quench; s==5: eq_low; s==6: final_opt`) instead of the canonical 7-stage numbering (`s==4: eq_high; s==5: quench; s==6: eq_low; s==7: final_opt`). With the CLI default of `--batch-stages 5 6 7`, this caused the controlled cooling step to be silently skipped — runs did NVT-eq-at-300K + final-opt instead of quench + eq_low + final_opt. **Re-run any batch-quench output produced before this fix if methodology accuracy matters (e.g. publication).** Unknown stage numbers now raise `ValueError` instead of silent skip.
- **Resume bug in equilibrate stages (2, 4, 6).** The trajectory file and the stage's final-output checkpoint shared the same default name `stage{N}_eq.xyz`. This had two effects: (1) successful runs overwrote the trajectory data with a single-frame final state, losing trajectory history; (2) interrupted runs left a partial trajectory file at the checkpoint location, causing `--resume` to wrongly skip the stage. Fixed by splitting the defaults: trajectory → `stage{N}_eq_traj.xyz`, final output → `stage{N}_eq.xyz`. **Pre-fix `stage{N}_eq.xyz` files are ambiguous and should be deleted before resuming.**
- **`--extract-snapshots` now honours `--format`.** Previously the mode was hardcoded to write extxyz `.xyz` files regardless of the `--format` flag, which silently ignored `--format vasp` and `--format cif`. Now writes the correct format with the correct extension (POSCAR-style with `sort=True` for `vasp`).
- **`--extract-snapshots` count flag unified.** Both `-n` / `--n-structures` (the standard count flag used everywhere else) and the legacy `--n-runs` now control the snapshot count. `--n-runs` is preserved for backwards compatibility with existing scripts.

### Changed

- **Material classification** expanded to 20 class-aware packing regimes, each
  drawing radii from the right source (Shannon ionic / Cordero covalent /
  Goldschmidt metallic) so a bare composition auto-derives a physical density
  and minseps. Adds a joint charge-balance oxidation-state solver for mixed
  cation/anion compounds and a high-valent d⁰ → CN=6 override; boride and
  oxyhalide packing factors calibrated against crystal densities.
- **`amorphgen.analysis`** package now exports `rank_from_log`, `format_log_ranking`, `validate_against_reference`, and `format_validation_report` (previously only `StructureAnalyser`).
- **CLI help** reorganised into argument groups (modes / calculator / optimisation / pipeline / random-gen / batch-quench / batch-opt / analyse) for readability. All flags continue to work; only the help-text layout changed.
- **Default MD timestep** in `DEFAULT_CONFIG` is now **0.5 fs** (was 1.0 fs). Safer for heavy elements and unusual chemistries. Shipped example YAMLs explicitly set 1.0 fs which is fine for typical oxides with chgnet/MACE foundation models.
- **`make_cubic` reshape moved from start of stage 3 (melt) to start of stage 4 (eq_high).** Reshaping a fully molten liquid is benign (atoms diffuse and lose memory of the deformation in <1 ps); reshaping a still-crystalline structure at the start of stage 3 caused a small unphysical jolt at low T. The flag is now read from the `eq_high:` block, falling back to `melt.make_cubic` for one release as a backwards-compat bridge. Default behaviour is unchanged (cubic reshape on by default).

### Removed

- **M3GNet backend (via matgl).** Loader code and registry kept in place; install pipeline removed from `[all]` extra. Currently broken upstream (DGL drops Mac wheels; matgl 2.x model paths return 401 from HuggingFace). Will be reinstated when matgl/DGL packaging stabilises. Use **SevenNet** as the recommended drop-in replacement (similar architecture, no DGL dependency).
- **Experimental placement-algorithm subpackage** (`amorphgen/pipeline/placement/`) and the `--placement-algorithm` CLI flag. We tested four alternatives (Voronoi-CRN, WWW + Keating, Metropolis repair, ARTn-lite) on Si64 + CHGNet and found the existing default coordination-aware placement matched or outperformed all of them. The 70% CN=4 ceiling is set by CHGNet's energy landscape, not by placement quality, so a more sophisticated placement does not help unless paired with a Si-specific potential (which is outside AmorphGen's general-purpose scope). Modules and a full carry-forward report are archived locally (and gitignored) under `experiments/cn_fix_2026-05/` for any future revival.
