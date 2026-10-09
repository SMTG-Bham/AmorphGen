# Consolidation audit: issue #68

This records the implementation and remaining decisions from the repository-wide
[duplication audit](https://github.com/SMTG-Bham/AmorphGen/issues/68). Changes are
split into local reviewable commits. Shared code owns a specific operation;
callers retain their validation, scientific estimators, output schemas and
checkpoint policies unless a change is explicitly described below.

## Implemented responsibilities

| Responsibility | Canonical implementation and migrated callers | Preserved contract |
| --- | --- | --- |
| Safety and repulsive-core defaults | `configs/_defaults.py`; pipeline defaults and runtime validators | Copies isolate mutable caller settings; units and values unchanged. |
| Repeated stage schemas | `configs/yaml_config.py`: common MD, plateau and ramp fields | Independent stage dictionaries; quench still does not accept `make_cubic`; physical defaults remain stage-specific. |
| CLI distance parsing | `cli._parse_pair_distances`, called by `_parse_minsep` and `_parse_dmax` | Option-specific error text, skipped empty entries, duplicate last-wins behavior; downstream finite-value validation remains separate. |
| Calculator configuration | `utils.calculators.calculator_kwargs`; CLI and ASE stage entry points | Explicit `None`, per-entry-point defaults, parameter blocks, lazy backend loading, calculator injection/caching and pre-chdir construction. |
| Structure format mapping and discovery | `utils.structure_io`; conversion, generation, optimization, snapshot extraction and ensemble CLI | Conversion gathers all formats; ensembles select the first nonempty format; unsupported-format policies remain at callers. |
| Sorted Cartesian POSCAR output | `utils.structure_io.write_sorted_vasp`; conversion, ensemble collection, random generation and ASE optimization | Preserves numeric presort, ASE symbol sort, constraints and caller input. Torch direct-coordinate and unsorted snapshot output remain separate. |
| File SHA256 | `utils.persistence.sha256_file`; provenance, relaxation sidecars, sequential controller and pipeline resume | Binary streaming, 1 MiB chunks, existing digest strings and caller error policy. Structured/state hashes remain distinct protocols. |
| Atomic metadata publication | `utils.persistence.atomic_write_text`; manifests, random/sequential records and relaxation sidecars | Same-directory replace, cleanup and UTF-8; JSON schema/formatting remains caller-owned; sidecars retain their non-fsync policy. |
| Resume settings differences | `utils.persistence.changed_settings`; manifest and random resume | Sorted dotted paths; random resume retains its first-difference interface. |
| ASE optimizer and cell-filter setup | `utils.relaxation.get_optimizer_class` and `build_cell_filter`; random and stage relaxation | Supported optimizers, case sensitivity, Frechet fallback, stress checks and cubic virial force scale. |
| Analysis JSON conversion | `analysis/_serialization.py`; convergence, scattering, descriptors, melt-memory and uncertainty | Strict exporters still reject NaN/Inf; undefined statistical values are sanitized separately; Path support and exporter error wording remain local. |
| Density and continuous summaries | `utils.common.compute_density_gcm3`, `analysis.uncertainty.summarize_observations`; structural analysis | Original mass/volume conversion, population site spread, equal-structure uncertainty and absent-observation handling; unused comparison density implementation removed. |
| Energy metadata lookup | `analysis.energy.stored_info_energy`; ranking, summary and screening | Field precedence only is shared. Screening never evaluates calculators; ranking and reports keep their own coercion and fallback behavior. |
| Plotting/report primitives | `analysis.plotting` shared axis style, palette and density artists; `analyser._format_cn_entry` | Workflow-specific CI annotations, spine widths, output messages and CN filtering remain. `plot_tr` now uses the existing figure factory without changing the caller's Matplotlib backend. |
| Sequential observable names | `configs.descriptor_names`; CLI and generation controller | CLI checks chemical elements; controller requires elements in the actual composition. Numeric/API/YAML validation remains distinct. |
| MD ramps and schedules | `pipeline.md_ramp.run_ramp`, `utils.common.resolve_ramp_schedule`; heating, cooling and torch schedule construction | K/fs/K-per-ps units, rounding, endpoints, stage seeds, legacy filenames, resumed segments and completed-state safety checks. Torch stages 4/5/6 share dispatch while retaining their checkpoint policies. |
| MD table and frame selection | `utils.common.format_md_log_row` and `select_frame_indices`; ASE/torch writers, extraction, sampling diagnostics and batch quench | Identical rows, cadence and offsets; cheap uniform/last selection retains each caller's invalid-strategy policy. Unused torch state converter removed. |
| Equilibration diagnostics | Existing `extract_energies` and `plot_temperature` used by block/report analysis | Same numerical series and zero-target semantics; the log-temperature report adopts the existing temperature plot styling. |
| Raw production RDF | `analysis._rdf_kernel.raw_frame_rdfs`; ensemble RDF and automatic cutoff fitting | One neighbor search per frame, directed counts, N−1 same-species density and open distance boundaries; cutoff missing frames still contribute zero, ensemble missing pairs stay unavailable. |
| Tutorial diagnostics | `Tutorials/analysis_helpers.py`; T2–T6 notebooks | T3–T5 retain distinct minimum-image RDF grids; raw CN site order, geometric angle NaNs and standard elemental masses are preserved. T1/T2/T6/T7 incompatible RDF variants are annotated and retained. |
| CLI test harness and exact test setup | `test/amorphgen_test_helpers.py`; CLI, Cu engine and minsep-family tests | Real CLI exceptions/results, independent geometric/density oracle formulas, explicit 4.5/6 Å neighbor search cutoffs and fresh structures. |
| Duplicate Slurm resubmission | Original `examples/run_iro2_cubic_array_bluebear.slurm` with documented `sbatch` overrides | Same task IDs, seeds, output directories, resources and resume command; redundant resubmit script removed. |

## Deliberately separate or deferred

These are dispositions of audited candidates, not uninspected code. Changes to
their contracts belong in separate scientific or compatibility work.

| Candidate | Decision and evidence | Work required before further consolidation |
| --- | --- | --- |
| Full ASE relaxation loops | Keep loop orchestration separate. Stage optimization owns optimizer context, trajectories, `nsteps` and observers; random generation has different observer/step bookkeeping and shared reference-calculator lifetime. Setup is already shared. | Characterize optimizer observers, restart state, line-search internals and zero-step behavior for both entry points; then consider one in-memory stepping primitive. |
| Temporal RDF versus ensemble RDF | Keep temporal histogram adapter separate: it floors/clips bins, includes zero distance and represents missing pairs as zero. Ensemble analysis excludes zero and outer-boundary distances and represents unavailable pairs explicitly. | Decide overlap, exact-boundary and singleton/self-image policy; independent frame-level fixtures before any behavior change. |
| Other tutorial RDF variants | T1/T2 use cubic minimum-image distances, exact shell volumes and N² normalization; T6 uses periodic neighbors and N²; T7's same-species unordered pair counts have a factor-of-two normalization discrepancy. These cannot be aliases to the production RDF without changing results. | Specify the teaching estimator and normalization, then migrate in a separate scientific correction with independent pair-count tests and regenerated examples. |
| Total-correlation example | `examples/total_correlation_function.py` transforms averaged S(Q) at mean density. Production transforms each realization before averaging; radial-grid start, shell availability and Lorch cutoff also differ. | Choose ensemble weighting, grid and window contracts explicitly; validate mixed-cell ensembles and empty integration windows. Retain current example behavior until that decision. |
| Other density constants | `examples/collect_class_benchmark.py`, the IGZO Slurm example and composition-based cell estimates use distinct constants or different inputs. | Agree precision and isotope-mass policy before changing historical outputs. Composition estimates are not the same operation as measured Atoms density. |
| Site coordination and neighbor traversals | Structural summaries, screening, oxygen chemistry and temporal diagnostics differ in bonding selection, cutoffs, zero-distance treatment and sample unit. | Share only a demonstrated identical kernel; preserve chemistry and independent-structure versus time-series statistics. |
| General numerical validators | CLI/YAML accept built-in numeric types and aggregate/contextualize errors; scientific APIs accept Real/Integral/NumPy containers. Geometry checks also differ for full versus partial periodicity. | Define an intentional public type/error contract before merging validation. Generic statistical target validation already lives in `analysis.sequential`; do not restrict it to generation-only names. |
| MSD implementations | Equilibration uses reference-frame fractional displacements; snapshot sampling removes affine cell motion with mixed-PBC minimum images and has a different time-origin estimator. | Agree physical observable, changing-cell convention and time-origin averaging before reuse. |
| Serialization and identity protocols | Manifests, strict scientific exports, state dictionaries and frame fingerprints intentionally serialize different fields/types. | Version any schema/protocol change. Only byte-file hashing and atomic text publication are shared. |
| ASE and torch checkpoint/writer orchestration | Frame-zero, output cadence, synchronized batched truncation, stress availability and calculator-property capture differ. | Retain engine-specific state transitions; share only schedules and row formatting. |
| Seeds, retries, CN repair and device selection | Different stream namespaces, target-versus-floor acceptance, ordinary retry/skip policies versus complete sequential evidence, and MPS support are intentional. | Do not unify based on similar syntax. Sequential generation must retain every planned observation. |
| Remaining I/O wrappers | Snapshot output is unsorted; torch POSCAR output uses direct coordinates; format conversion has collision preflight; some output wrappers deliberately differ in directory creation and messages. | Avoid an option-heavy universal writer. Keep policy at callers and reuse only identical format primitives. |
| Small script/test similarities | Benchmark relaxation commands, T5's educational snapshot/batch loop, curve fixture construction, parsing-only CLI helpers and Slurm array parsing have different defaults, side effects or too little common responsibility. | Keep straightforward local code unless an actual maintenance need appears. |
| Default work-directory composition parsing | The CLI's best-effort directory hint intentionally falls back on malformed/formula input, whereas `_parse_composition` validates it. | A change to generated default paths needs a separate compatibility decision. |
| Legacy finite checks | `common.assert_finite` differs from `SafetyMonitor` in exception and constrained-force handling; external callers may still use it. | Establish deprecation policy before removing or redirecting it. |
| Existing delegates and independent scientific methods | `final_opt`, plateau equilibration, analyser facade methods, direct/FT S(Q), uncertainty estimators, smoothing in different units, elasticity/vibration finite differences and defect/contact graphs retain distinct roles. | Preserve them. Similar code is not evidence of equivalent science. |
| Independent test references | Analytic stress/force models, harmonic and failing calculators, and independent minimum-distance checks remain independent of production code. | Never replace an oracle with the implementation it tests. |

## Validation and scope

The unmodified baseline passed **2,376 tests with 23 skips** on Python 3.12.
The consolidated core suite passed **2,438 tests with the same 23 skips**.
Each implementation commit also has focused regression coverage. Additional
checks passed for **85 tests on Python 3.10 with minimum dependencies**, **61
real CPU torch-sim tests**, and **122 external-backend/core tests** in the
LAMMPS/ACE environment (three unrelated optional skips). Fatal/undefined-name
lint and test unused-code lint passed. A further **88 tests passed** for
calculator loading (including real CHGNet), torch repulsion, provenance and
preemption.

The source distribution includes the shared test helper, tutorial helper and
all seven notebooks used by the characterization tests. A wheel built from the
unpacked source distribution passed **474 tests against an isolated installed
wheel**. The documentation, including this record, built with warnings treated
as errors and **zero warnings**. Documentation resolved public intersphinx
inventories; simulations used installed local runtimes. No changes were pushed.

In addition to ordinary regressions, seeded fresh and resumed heating/cooling
runs under NVT, Berendsen and MTK produced **42 byte-identical output files**
against the earlier implementation. RDF and automatic-cutoff changes have
independent count/normalization fixtures and before/after differential checks.

Two invalid-input protections are explicit: a zero torch ramp rate now raises
the same descriptive `ValueError` as ASE instead of division-by-zero, and a
zero-duration quench fails before opening/truncating output files. Valid
simulation schedules are unchanged. Sequential source fingerprints now cover
the extracted helpers, so subsequent scientific-helper edits cannot escape
resume compatibility checks; as before, source changes invalidate old
sequential checkpoints rather than mixing different generating procedures.

The original audit inspected first-party package code, scripts, test helpers and
all seven tutorial notebooks. Generated build/documentation output, cached
artifacts, vendored code and external dependencies were excluded. Expensive
tutorial MD workflows and GPU/model-download runs are not substituted by the
small deterministic regression fixtures; unavailable checks are reported with
the final commit set.
