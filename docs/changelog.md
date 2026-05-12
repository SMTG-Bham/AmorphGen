---
orphan: true
---

# Changelog

## v1.0.0 (Unreleased)

### Added

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

- **`amorphgen.analysis`** package now exports `rank_from_log`, `format_log_ranking`, `validate_against_reference`, and `format_validation_report` (previously only `StructureAnalyser`).
- **CLI help** reorganised into argument groups (modes / calculator / optimisation / pipeline / random-gen / batch-quench / batch-opt / analyse) for readability. All flags continue to work; only the help-text layout changed.
- **Default MD timestep** in `DEFAULT_CONFIG` is now **0.5 fs** (was 1.0 fs). Safer for heavy elements and unusual chemistries. Shipped example YAMLs explicitly set 1.0 fs which is fine for typical oxides with chgnet/MACE foundation models.
- **`make_cubic` reshape moved from start of stage 3 (melt) to start of stage 4 (eq_high).** Reshaping a fully molten liquid is benign (atoms diffuse and lose memory of the deformation in <1 ps); reshaping a still-crystalline structure at the start of stage 3 caused a small unphysical jolt at low T. The flag is now read from the `eq_high:` block, falling back to `melt.make_cubic` for one release as a backwards-compat bridge. Default behaviour is unchanged (cubic reshape on by default).

### Removed

- **M3GNet backend (via matgl).** Loader code and registry kept in place; install pipeline removed from `[all]` extra. Currently broken upstream (DGL drops Mac wheels; matgl 2.x model paths return 401 from HuggingFace). Will be reinstated when matgl/DGL packaging stabilises. Use **SevenNet** as the recommended drop-in replacement (similar architecture, no DGL dependency).
- **Experimental placement-algorithm subpackage** (`amorphgen/pipeline/placement/`) and the `--placement-algorithm` CLI flag. We tested four alternatives (Voronoi-CRN, WWW + Keating, Metropolis repair, ARTn-lite) on Si64 + CHGNet and found the existing default coordination-aware placement matched or outperformed all of them. The 70% CN=4 ceiling is set by CHGNet's energy landscape, not by placement quality, so a more sophisticated placement does not help unless paired with a Si-specific potential (which is outside AmorphGen's general-purpose scope). Modules and a full carry-forward report are archived locally (and gitignored) under `experiments/cn_fix_2026-05/` for any future revival.
