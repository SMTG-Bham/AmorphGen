# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AmorphGen is a Python package (v1.0.0) for generating amorphous atomic structures via melt-and-quench molecular dynamics simulations and random sequential placement. It supports multiple calculator backends: MACE, CHGNet, SevenNet (MLIPs) and Lennard-Jones, Buckingham+Coulomb (classical pair potentials).

## Build & Install

```bash
pip install -e .              # Lightweight, torch-free: random-gen + analysis + classical potentials
pip install -e ".[all]"       # Install with all working backends (mace, chgnet, sevennet)
pip install -e ".[dev]"       # Install with test dependencies
pip install -e ".[docs]"      # Install with docs dependencies
```

Requires Python >=3.10. Core deps: ase>=3.22, numpy>=1.24, scipy>=1.10, pyyaml>=6.0.
**Torch is deliberately NOT a core dep** — it arrives via the MLIP extras; with
no torch present, `device="auto"` resolves to cpu (`utils.common.resolve_device`)
and the CI `light-install` job guards this contract.

## Testing

```bash
pytest test/ -v --tb=short              # Run Tier 1 & 2 tests (no GPU needed)
pytest test/ -v --tb=short --run-mace   # Include Tier 3 MACE integration tests (GPU)
pytest test/test_utils.py -v            # Run a single test file
pytest test/test_utils.py::test_merge_config -v  # Run a single test
```

**729 tests** across 23 files. **Test tiers** (defined in `test/conftest.py`):
- **Tier 1**: Pure unit tests, no calculator
- **Tier 2**: Uses ASE's EMT calculator, no GPU required (CI runs these)
- **Tier 3**: Full MACE/CHGNet/SevenNet tests, requires `--run-mace` flag and GPU

## Architecture

### Operational modes

1. **Full 7-stage melt-quench pipeline** (default): optimise -> equilibrate -> heat -> equilibrate -> cool -> equilibrate -> final optimise
2. **MQ-ensemble** (`--mq-ensemble`): integrated workflow — stages 1-4 once + N independent quenches via auto-extracted snapshots from stage 4 trajectory. Outputs collected to `final/`. One CLI call.
3. **Hybrid-ensemble** (`--hybrid-ensemble`): take a directory of disordered structures, run stages 4-5-6-7 on each. Cheaper than crystal-MQ; useful for random-gen + quench workflows.
4. **Random structure generation** (`--random-gen`): random placement with automated bonding-type-aware minimum separations from Shannon ionic radii
5. **Batch quench** (`--batch-quench`): Extract snapshots from a trajectory and quench each independently. Polymorphic `--snapshot-dir` accepts either a directory of static structures or a single trajectory file (auto-extracts).
6. **Structure analysis** (`--analyse`): RDF, coordination numbers, bond angles, ring statistics, Voronoi, energy ranking
7. **Utility modes**: `--extract-snapshots TRAJ`, `--rank-from-log LOG`, `--list-models`

### Key modules

- **`amorphgen/cli.py`** -- CLI entry point (`amorphgen` command), dispatches to all modes. 78 flags with short aliases (-m, -d, -f, -O, -C, -n, -o). Supports formula composition (`In2O3*16`) and atom count format (`In=32,O=48`). `--examples` prints a concise common-usage block (same text as the `-h` epilog; single source `_EXAMPLES` in cli.py — module docstring keeps only advanced examples).
- **`amorphgen/pipeline/run_pipeline.py`** -- `MeltQuenchPipeline` orchestrator; 7-stage workflow with `--resume` support
- **`amorphgen/pipeline/random_gen.py`** -- `generate_random()` and `batch_random()` for random structure generation with SC placement. Vectorized placement loop with precomputed minsep/dmax/CN lookup tables.
- **`amorphgen/utils/radii.py`** -- Shannon ionic radii (49 elements), metallic radii (38 elements), bonding classification, minsep calculation with caps, density estimation with 20-class material classification + joint oxidation-state solver
- **`amorphgen/utils/calculators.py`** -- Multi-backend calculator factory (`get_calculator()`). Supports MACE, CHGNet, SevenNet, Lennard-Jones, Buckingham+Coulomb. MACE import has try/except with install instructions. SevenNet multi-fidelity (`mf`) models default `modal='mpa'`.
- **`amorphgen/utils/classical.py`** -- LJ and Buckingham+Coulomb calculators. Vectorized NumPy with optional PyTorch GPU. Wolf summation for Coulomb.
- **`amorphgen/utils/common.py`** -- MD dynamics builder, trajectory I/O, config merging, `compute_density_gcm3()` helper
- **`amorphgen/analysis/analyser.py`** -- `StructureAnalyser` class with 15 documented public methods. Delegates to submodules: rdf.py, structure.py, rings.py, voronoi.py, energy.py, cutoff.py, plotting.py.
- **`amorphgen/configs/default_config.py`** -- Central `DEFAULT_CONFIG` dict. Default timestep 0.5 fs.
- **`amorphgen/configs/yaml_config.py`** -- YAML config loader with schema validation (errors for wrong types/devices/ensembles, warnings for unknown keys)

### Random structure generation

Minsep and density are computed automatically from Shannon ionic radii:

- **Bond classification**: ionic (M-O), metallic (M-M), covalent (Si-Si), anion packing (O-O)
- **Minsep**: `(r1 + r2) * scale_factor`, with CN-aware radii from `--target-cn`
- **Minsep caps**: M-M and same-element capped at 2.80 A, ionic at 3.00 A, anion packing at 3.00 A. Prevents impossible placement.
- **Density estimation**: Unified class-aware sphere packing on per-element radii — Shannon ionic (CN=6) for ionic classes, Cordero covalent for covalent/group-IV/pnictide/chalcogenide classes, Goldschmidt metallic for alloys (and for the cation in transition-metal carbides and metal borides). The legacy elemental-density-mixing branch was retired in favour of this unified path. `estimate_density()` is kept as a public helper for mass-averaged crystal density. Class-aware packing factors (live values from `radii.py:PACKING_FACTORS`):

| Class | Packing factor | Radius source | Examples |
|-------|---------------|--------------|----------|
| rutile_dioxide | 0.66 | Shannon ionic | TiO2, SnO2, RuO2, IrO2 (small-cation MO2) |
| fluorite_dioxide | 0.63 | Shannon ionic | ZrO2, HfO2, CeO2, ThO2, UO2 (large-cation MO2) |
| small_cation_nitride | 0.62 | Shannon ionic | AlN, GaN, Si3N4, TiN |
| high_valent_oxide | 0.60 | Shannon ionic | V2O5, Nb2O5, MoO3, WO3 (OS ≥ 5) |
| transition_metal_carbide | 0.60 | Goldschmidt cation + Cordero C | TiC, WC, ZrC (rocksalt-like dense) |
| alloy | 0.60 | Goldschmidt metallic | NiTi, CuZr, brass, pure metals |
| halide | 0.58 | Shannon ionic | Li2ZrCl6, LiF, NaCl |
| oxyhalide | 0.52–0.58 (interpolated by halogen fraction) | Shannon ionic | BiOCl, LaOCl, NaTaOCl4, ZrOCl2 |
| hydride | 0.55 | Shannon ionic | LiH, MgH2, NaAlH4 |
| metal_oxide | 0.52 | Shannon ionic | In2O3, Al2O3, Ga2O3, MgO, ZnO |
| nitride | 0.52 | Shannon ionic | ZrN, HfN, ScN (large cation) |
| default | 0.52 | Shannon ionic (fallback) | — |
| covalent_oxide | 0.50 | Shannon ionic | SiO2, GeO2, B2O3 |
| boride | 0.60 | Goldschmidt cation + Cordero B | TiB2, MgB2, ZrB2 (LaB6-type cage borides run ~100%) |
| covalent_network_oxide | 0.35 | Cordero covalent | BeO (small polarizing cation) |
| pnictide | 0.32 | Cordero covalent | GaAs, InP, InAs (III-V compounds) |
| covalent_carbide | 0.32 | Cordero covalent | SiC, B4C (open network) |
| group_iv | 0.30 | Cordero covalent | Si, Ge, C (tetrahedral covalent network) |
| chalcogenide | 0.30 | Cordero covalent | ZnS, CdTe, GeTe (II-VI / IV-VI) |
| chalcogenide_glass | 0.23 | Cordero covalent | GeS2, GeSe2, As2S3, As2Se3 (sulfide/selenide network glasses of Ge, Si, As, Sb, B, P) |
| elemental_semiconductor | 0.28 | Cordero covalent | a-Se, a-Te, a-As, a-Sb, a-P |

Metal-rich metal + (P, B, Si, Ge, As, Sb) compositions with under 35 % metalloid
(Ni80P20, Fe80B20, Pd80Si20) are alloys (metallic glasses), not pnictides or
borides; all-nonmetal oxides (P2O5, SO3) are covalent oxides with the nonmetal
in its highest positive Shannon state. The dioxide (rutile/fluorite) and nitride
(small/large-cation) splits are decided by a cation-radius rule; high_valent_oxide is gated on OS ≥ 5; BeO is
routed to covalent_network_oxide by an explicit small-cation exception. Cation
oxidation states (for the Shannon radius) come from a joint charge-balance
solver that resolves multivalent cations in mixed-cation/anion compounds.
Oxyhalides (O + halogen anions, e.g. BiOCl, NaTaOCl4) get a packing factor
interpolated between metal_oxide (0.52) and halide (0.58) by the halogen
fraction of the halogen+O anion pool (`_oxyhalide_packing_factor`); a
dopant-level halogen (<10% of that pool, e.g. F-doped TiO2 / FTO) does NOT
trigger the oxyhalide class — the compound keeps its oxide routing.

- **SC placement**: coordination-aware atom placement near under-coordinated sites. `dmax = minsep * dmax_factor` (default 1.5) defines the bonding shell. Covers ionic, covalent, metallic (alloys only), and pure element bonds.
- **Auto-retry** (two levels, policy via `retry_mode` / `--retry-mode`):
  - *Soft-pack* (first response to a stall in `"expand"` mode, `_push_apart()`): the structure is re-placed at the SAME cell with floors x0.72, then every pair closer than its floor is pushed apart iteratively until all pairs reach 0.985 of their floors. Random sequential addition jams near a 0.38 hard-sphere fraction; overlap removal reaches 0.6, so dense oxides (MgO, BeO), alloys, borides and nitrides now keep their estimated density instead of expanding (MgO 1.99 -> 2.67 g/cm3). `atoms.info["soft_pack"] = True` marks it. Falls through to expansion only if the floors cannot be reached.
  - *Per-structure* (`generate_random`, budget 4 retries): `"expand"` (default) grows the cell 5%/retry keeping minseps physical — right when density is an estimate; `"reduce-minsep"` holds the cell (density) EXACTLY fixed and softens only non-bonded minseps (same-element + anion-anion) 5%/retry — for fixed-density film / isochoric studies; `"none"` disables auto-retry entirely — a stall raises immediately (both density AND minseps stay exact). Cation-anion bond minseps are never reduced in any mode.
  - *Batch escalation ladder* (`batch_random`, after 10 consecutive failures per rung): expand mode → density_scale ×0.92/0.85/0.78, then M-M minsep −5/10/15%, then skip the structure; reduce-minsep mode skips the density rungs (cell must not move) and goes straight to the minsep rungs; none mode has no rungs at all (seed resampling only, then skip).
- **CN targets**: auto-detected from composition. Alloys get CN=8 with tolerance=2. Oxides get CN=4-5 depending on cation type.
- **Min-CN floor (default, no flag)**: a built-in rule that eliminates dangling bonds. Every atom is given a hard coordination floor — auto **anions=2, cations=3** (each capped at the element's target CN, override via `min_cn=int|dict`). After placement, `_repair_min_cn()` runs by default (`repair_floor=True`) and relocates any below-floor atom into a position bonded to ≥floor acceptors. Guards ensure it never over-coordinates past target+tol, never creates a new violation, and never lowers the count of atoms meeting target CN. Cuts IrO2 dangling-O from ~22% → ~3% at placement (→<1% after relaxation). Placement itself is unchanged; this is a post-pass. Not exposed as a CLI flag — it just always runs.

### Configuration precedence

CLI arguments > YAML config (`--config`) > `DEFAULT_CONFIG`. Deep-merged via `merge_config()`.

### Pipeline stages

| Stage | Module | Function |
|-------|--------|----------|
| 1, 7 | `opt_cell.py` / `final_opt.py` | Structure optimisation (LBFGS/FIRE + cell filter) |
| 2, 4, 6 | `equilibrate.py` | Constant-T equilibration (NVT/NPT) |
| 3 | `melt_cell.py` | Heating ramp (rate in K/ps) |
| 5 | `quench.py` | Cooling ramp (rate in K/ps) |

Default timestep: 0.5 fs. Default melt temperature: 3000 K.

### MD dynamics

NVT uses Langevin thermostat; NPT uses NPTBerendsen. Built via `build_md_dynamics()` in `common.py`.

### Output formats

- VASP output is auto-sorted by species (`sort=True`) for clean POSCAR format
- Supported: xyz (extxyz), vasp, cif
- All outputs are ASE-readable without conversion

### CLI short aliases

| Short | Long | What |
|-------|------|------|
| `-m` | `--model` | Calculator model |
| `-d` | `--device` | Device (auto/cpu/cuda/mps) |
| `-f` | `--fmax` | Force convergence |
| `-O` | `--optimizer` | LBFGS/FIRE/etc |
| `-C` | `--cell-filter` | Cell constraint |
| `-n` | `--n-structures` | Number to generate |
| `-o` | `--work-dir` | Output directory |

### Composition input

CLI accepts two formats:
- Formula: `--composition "In2O3*16"` (16 formula units = 80 atoms)
- Atom counts: `--composition In=32,O=48`

Shell quoting required for `*` (glob character). Python API always takes dict: `{"In": 32, "O": 48}`.

Element symbols validated against periodic table (catches typos like "Ox").

## Documentation

Sphinx with MyST (Markdown) and sphinx_rtd_theme. Build with:
```bash
cd docs && make html
```

Live preview: `sphinx-autobuild . _build/html --open-browser`

Key pages: index.md (landing), getting-started/quickstart.md, guides/random-generation.md, guides/backends.md, api/calculators.md

Logo files in `docs/_static/`: logo.svg (sidebar mark), logo_hero.png (landing page), favicon.png, main_Fig.png (workflow figure).

## CI

GitHub Actions runs pytest on Python 3.10/3.11/3.12 (Tier 1 & 2 only). See `.github/workflows/test.yml`.

## Important notes

- **SMTG repo** (`AmorphGen-smtg`): Never commit, push, or mention Claude Code there. User syncs manually. Replace `cywkmc21` with `SMTG-Bham` in URLs.
- **VASP output**: Always sort atoms by species with `sort=True` for clean POSCAR.
- **Shell quoting**: All docs examples with `*` in composition must use quotes (`"In2O3*16"`).
- **Device default**: `auto` (detects GPU). Not `cuda` -- would crash on laptops.
- **Timestep default**: 0.5 fs (safe for all systems). Was 1.0 fs before this session.
- **Resume granularity**: stage-level AND frame-level. `--resume` first skips completed stages (checkpoint scanning), then the first incomplete MD stage (2-6) continues from the last frame of its `stage{N}_*_traj.xyz` trajectory with the remaining steps — momenta are carried from the frame; only Langevin RNG / Berendsen barostat scaling state is lost (negligible in equilibrium MD). Ramp stages (3, 5) recover their position in the temperature schedule from the elapsed-step count. Optimisation stages (1, 7) restart whole. Implemented via `read_md_checkpoint()` + `attach_outputs(append=...)` in `utils/common.py`; fresh (non-resume) runs truncate stale trajectories so frame counting stays exact.

## Future work / known limitations

- **Large-cation nitride density is placement-limited, not parameter-limited** (verified 2026-07-31): with the auto-expand retry active, raising the nitride packing factor makes the FINAL density *worse* (ZrN 82%→73% of crystal at pf 0.58 — failed placements trigger expansion overshoot), and disabling SC placement + the min-CN floor changes nothing (ZrN saturates at ~73% either way). The bottleneck is random-sequential-placement saturation under the minsep constraints, intrinsic to the method. Remedies: `--target-density` if the density matters at generation time, or (better) NPT/cell-relaxation after generation — the MLIP densifies the seed regardless. pf 0.52 is kept as the highest value that avoids expansion overshoot.
