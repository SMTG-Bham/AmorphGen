# EXAFS descriptors and FEFF spectra

AmorphGen provides three related calculations:

* `--exafs` measures geometric first-shell coordination, distance cumulants and
  an approximate effective-distance correction. These are not experimentally
  fitted coordination numbers.
* `--harmonic-fit` estimates Gaussian first-shell fitting bias using the
  ensemble's distances and a bundled FEFF6L backscatter-amplitude table.
  Its generic amplitude table and scalar mean free path make it an approximation
  to an experimental fit, not a replacement for a full FEFF/Larch analysis.
* `--feff-write`, `--feff-run` and `--feff-collect` calculate actual FEFF spectra
  for sampled absorber sites and average chi(k) before transforming to R space.
  With FEFF installed, one command goes from structures to averaged spectra,
  figures and CSV tables (see below).

FEFF is an external executable and is not bundled. The input writer targets
FEFF8L/FEFF10 cards. The optional numerical integration test targets FEFF8.5L.

A complete example using two shipped Ga₂O₃ structures is available in
`examples/exafs/run_example.py`. From the repository root, run
`python examples/exafs/run_example.py all --executable /path/to/feff`.
It retains the structures, FEFF inputs and outputs, spectra and a plot.
Two sites are an execution demonstration, not a converged ensemble.
[Tutorial 9](https://github.com/SMTG-Bham/AmorphGen/blob/main/Tutorials/T9_exafs/tutorial_9_exafs.ipynb)
goes through all three calculations on the Tutorial 8 cells, including a
FEFF convergence check.

## Validated settings

:::{note}
These settings passed a 2 % convergence test with FEFF8.5L on MACE-MPA-0
ensembles of a-Ga₂O₃ (Ga K edge) and a-InGaZnO₄ (Zn, Ga and In K edges). The
test compares the complex Fourier transform up to R = 4 Å (k² or k³, Hanning
window 3–12 Å⁻¹, dk = 1 Å⁻¹).

| Setting | Value | Option |
|---|---|---|
| Cluster radius | 10 Å | `--feff-radius 10` |
| Path length | 8 Å | `--feff-card RPATH=8` |
| Path filters | 1.0 % and 0.5 % | `--feff-card "CRITERIA=1.0 0.5"` |
| Scattering order | 4 legs | default (`NLEG 4`) |
| Potential populations | parent cell | `--feff-stoichiometry structure` |
| Potentials | non-self-consistent, Hedin–Lundqvist, S₀² = 1 | defaults |

Each refinement changed the transform by at most 0.54 % (a-Ga₂O₃, 3140 sites in
20 cells) and 0.80 % (a-InGaZnO₄, 10 sites per element in 20 cells, worst edge):
path filters 0.25 % and 0.125 %, `RPATH` 10 Å, `NLEG` 6 and a 12 Å cluster.
Every absorber site was used: all 160 Ga sites in each of 100 a-Ga₂O₃ cells,
and all 50 sites per element in each of 504 a-InGaZnO₄ cells. Half the median
difference between two halves of the cells was 0.99 % (a-Ga₂O₃) and at most
1.81 % (a-InGaZnO₄).

The test covers the first shells and the region up to 4 Å, for static
structures. It does not cover the full k-space spectrum, where tighter filters
and longer paths still change χ(k) by up to about 12 % through paths beyond
4 Å. Nor does it cover thermal disorder, self-consistent potentials, FEFF10 or
other materials. The defaults (6 Å cluster, `RPATH` equal to the radius, FEFF's own
`CRITERIA`, cluster-inferred populations) are kept for compatibility. They are
not path-converged: going from `RPATH` 6 to 8 Å changes the a-Ga₂O₃ χ(k) by
6.3 %.

```bash
amorphgen --analyse --input-dir structures \
  --feff-write feff --feff-absorber Ga --feff-radius 10 \
  --feff-stoichiometry structure --feff-card RPATH=8 \
  --feff-card "CRITERIA=1.0 0.5" --feff-run --feff-workers 8 \
  --feff-kmin 3 --feff-kmax 12 --save-plot analysis
```

Choose the number of sites and cells for your own system by a convergence
test. `examples/exafs/ga2o3_validation.py` uses these settings by default.
:::

## Descriptors

```bash
amorphgen --analyse --input-dir structures \
  --exafs Ga-O --harmonic-fit Ga-O --save-plot analysis
```

Pairs are ordered: `Ga-O` means Ga absorber, O scatterer. `--exafs-bootstrap`
and `--harmonic-bootstrap` resample whole structures for sampling errors.
They assume independent configurations; correlated MD frames need block
selection/resampling. To analyse multiple frames through the Python API, load
them explicitly with `ase.io.read("trajectory.traj", index=":")` and pass the
resulting list of Atoms to `StructureAnalyser`.

The harmonic residual uses scatterer-dependent |f(k)|/k weighting. The bundled
table can be regenerated using
`docs/notes/exafs_crosscheck/generate_backscatter_table.py` in a separate
environment containing xraylarch. Its FEFF6L attribution is retained in the
generator and table metadata.

## Write and run FEFF jobs

```bash
amorphgen --analyse --input-dir structures \
  --feff-write feff_ga --feff-absorber Ga --feff-sites 5 --feff-radius 6
bash feff_ga/run_all.sh
```

Choose the site count by a convergence study. Five is only a small starting
sample, not a convergence recommendation.

## From structures to spectra in one command

On a machine with FEFF, `--feff-run` runs the trees straight after writing them,
then collects, reports and plots them as `--feff-collect` does:

```bash
amorphgen --analyse --input-dir structures \
  --feff-write feff --feff-absorber In,Ga,Zn --feff-sites 20 \
  --feff-run --feff-executable /path/to/feff8l --feff-workers 8 \
  --feff-card RPATH=8 --feff-card "CRITERIA=1.0 0.5" \
  --save-plot analysis
```

* Several absorbers each get a tree, `feff/In`, `feff/Ga` and `feff/Zn`; one
  absorber uses `feff/` itself.
* `--feff-workers` runs that many FEFF jobs at once (FEFF is single-threaded).
* `--feff-executable` defaults to `$FEFF_EXECUTABLE`, then `feff` on PATH.
* `--feff-card KEY=VALUE` sets any header card, for example the path length,
  scattering order and path filters that path-convergence tests vary. It is
  repeatable and overrides `analysis.feff_cards`.
* Add `--resume` to restart an interrupted run: jobs that finished and still
  match their recorded input and output fingerprints are kept, and everything
  else is run.
* A tree with a failed job is not averaged, because a partial average leans
  towards whichever sites ran. The command stops with FEFF's own diagnostic.

The YAML equivalent goes in the `analysis:` block:

```yaml
analysis:
  feff_write: feff
  feff_absorber: In,Ga,Zn
  feff_sites: 20
  feff_run: true
  feff_executable: /path/to/feff8l
  feff_workers: 8
  feff_cards: {RPATH: 8, CRITERIA: "1.0 0.5"}
  save_plot: analysis
```

From Python, `feff_pipeline` does the same and returns each tree's run
summary, averaged result and report:

```python
import glob
from ase.io import read
from amorphgen.analysis.feff import feff_pipeline

structures = [read(f) for f in sorted(glob.glob("opt/*.xyz"))]
out = feff_pipeline(structures, ["In", "Ga", "Zn"], "feff",
                    executable="/path/to/feff8l", workers=8, n_sites=20,
                    cards={"RPATH": "8", "CRITERIA": "1.0 0.5"},
                    plot_dir="analysis")
print(out["Ga"]["report"])
```

For a cluster, write the trees without `--feff-run` and run each
`joblist.txt` as an array, as below, then collect.

On a cluster with AmorphGen and FEFF installed, run each selected job through
the managed runner, from the job-tree directory:

```bash
DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" joblist.txt)
python -m amorphgen.analysis.feff --jobs-dir . --job "$DIR"
```

The runner also accepts `--executable /path/to/feff`, `--timeout SECONDS`,
`--workers N` and `--skip-complete`. It returns a failing exit status if any
requested job fails.

Inputs are fingerprinted in `manifest.json`. Writing identical input is
idempotent; changing the structure or calculation settings requires a new
job-tree directory. This prevents old spectra from being silently attached to
new structures.

Each run uses a scratch directory, carries existing intermediate files for
explicit partial-stage FEFF workflows, and requires fresh chi.dat output.
`run_status.json` records completion and input/output fingerprints.
Failed or interrupted runs cannot be collected as successful results, even if
older output files remain. A per-job lock prevents concurrent managed runs.
After a machine crash, verify that no calculation remains before removing a
leftover `.amorphgen_feff.lock` and rerunning.

## Collect and transform

```bash
amorphgen --analyse --input-dir structures \
  --feff-collect feff_ga --feff-kmin 3 --feff-kmax 12 --feff-dk 1 \
  --feff-kweight 2 --save-plot analysis
```

`--feff-collect` also accepts a directory of per-absorber trees, such as `feff/`
above, and collects each tree in turn.

The collector validates recorded inputs, output hashes and the numerical data.
Malformed rows, non-finite values and unsorted/duplicate k values are rejected.
Missing jobs are reported; a partial ensemble may be biased.

All spectra are interpolated only over their shared k range. Hanning tapers are kept inside the data,
and the report records the actual window. With data ending at k=12 and dk=1,
the upper taper centre cannot exceed 11.5 Å⁻¹. To use an upper window endpoint
of 12 with dk=1, the shared data must extend to at least 12.5 Å⁻¹; the collector
retains that extra half-taper. Match the actual window,
k weight and R range when comparing with an experiment.

Sites are averaged within each structure, then structures are equally weighted.

### Figures and tables

With `--save-plot DIR`, each absorber gives two standalone figures in the style
of the other analysis plots, each with a CSV of the plotted numbers beside it:

| File | Content |
|---|---|
| `analysis_<El>_feff_chik.{png,csv}` | k-weighted chi(k) |
| `analysis_<El>_feff_chir.{png,csv}` | Magnitude of chi(R), phase-uncorrected |
| `analysis_<El>_feff_chik_uncertainty.{csv,json}`, `_per_structure.csv` | Structure-level summaries, as for the other curves |

`--save-pdf` adds PDFs. Each CSV starts with a `#` line recording the absorber,
edge, site and structure counts, k weight and transform window. Empty cells are
undefined values. The columns are:

| Column | Meaning |
|---|---|
| `k_invA`, `R_A` | Grid (Å⁻¹, Å) |
| `chi`, `k2chi_invA2` | Equal-weight mean of structure spectra, unweighted and k-weighted |
| `chi_site_std` | Pooled site-to-site population spread; not an error bar |
| `chi_structure_sem` | Sample SD of structure means / sqrt(number of structures) |
| `chir_mag_invA3`, `chir_re_invA3`, `chir_im_invA3` | Fourier transform (for k weight 2) |
| `chir_mag_bootstrap_low`, `_high` | 95% pointwise bootstrap interval of the magnitude of chi(R) |

The plotted bands are 95% pointwise bootstrap intervals that resample whole
structures, as in the other ensemble analyses. For `|chi(R)|` each resampled
mean is transformed before taking its magnitude. A magnitude is not linear in
the mean, so with few structures the band can sit above the curve near a peak.

With fewer than two contributing structures there is no band, the SEM is
empty, and the report says it is unavailable. The bands cover structure
sampling only, not path truncation, potential accuracy, calibration or
finite-temperature effects. A narrow band alone does not establish
convergence in absorber count.

### Importing previous or externally run results

Outputs made directly with `feff`, rather than the managed runner, have no
completion record. If you have independently confirmed that the outputs match
the current feff.inp files, explicitly import them once:

```bash
amorphgen --analyse --input-dir structures \
  --feff-collect feff_ga --feff-trust-external --save-plot analysis
```

This validates and records the current files, marks them as external imports,
and states in the report that historical correspondence was trusted by the
caller. It cannot reconstruct missing provenance. It does not override changed
inputs, altered recorded outputs, or a failed managed run. Subsequent collection
does not need the flag. Keep original result directories backed up before import.

## Scientific interpretation

Relaxed structures provide static configurational disorder; thermal motion is
not automatically added. S02=1 is the default amplitude normalisation, not an
experimental calibration. Validate cluster size, path settings and absorber/
configuration sampling before drawing structural conclusions. The harmonic
quality flags are diagnostics rather than guarantees of experimental agreement.

Reference YAML `exafs` rows compare point estimates and report this limitation.
Existing confidence-aware validation for other structural descriptors is
unchanged. Geometric and harmonic estimates must be distinguished from
parameters fitted directly to measured spectra.

## Tests

```bash
python -m pytest test/test_exafs.py test/test_harmonic.py \
  test/test_feff.py test/test_feff_integrity.py
python -m pytest test/test_feff_integration.py --run-feff
```

The second command requires an installed FEFF8.5L executable; optionally set
`FEFF_EXECUTABLE`. It compares an independently generated Cu cluster against
the upstream reference. No executable or model is downloaded.

## Potential composition for periodic solids

The default `--feff-stoichiometry cluster` retains the original input format:
FEFF infers potential populations from the finite cluster. A spherical cut of
an oxide need not have the same elemental proportions as the parent cell.
This can affect its potentials and spectrum even when the cluster geometry
is correct.

Use `--feff-stoichiometry structure` (or `stoichiometry="structure"` in
`write_feff_jobs`) to write reduced parent-cell element counts in the
POTENTIALS `xnatph` columns. The excited absorber receives population 0.001;
angular momentum limits are left to FEFF's element-based defaults. This
mode requires full three-dimensional periodicity and a cluster containing a
scattering potential for each parent element. It is inappropriate for an
isolated molecule or an implicitly homogenised multiphase cell without further
assessment. Both modes use one scattering potential per element.

The setting is recorded in the manifest and input fingerprints. Changing it
requires a new job directory. Earlier calculations with omitted `xnatph`
remain reproducible; they are not relabelled as structure-stoichiometry results.
Compare the two modes, along with path length and scattering order, when
validating a material. The FEFF [POTENTIALS documentation](https://feff.phys.washington.edu/feff/Docs/feff8/feff85/feff85/POTENTIALS.html)
describes these populations and their role in the Fermi-level calculation.

The optional real-FEFF suite now also includes NiO K-edge, UO2 L3-edge and
self-consistent Cu references, using independently generated ASE crystals and
explicit stoichiometry. Direct Fourier-quadrature and periodic-representation
checks run without FEFF in `test/test_feff_numerics.py`.

## Path length, path filters and FEFF's path limit

`RPATH` follows the cluster radius unless set through `cards`. FEFF also
discards weak paths with its `CRITERIA` filters: by default, paths below 2.5%
of the strongest plane-wave amplitude during the path search and below 4%
curved-wave amplitude in the final sum. In a disordered oxide these filters,
not `RPATH`, can set the effective path length. For the a-Ga₂O₃ validation
ensemble at default `CRITERIA`, no path beyond about 8 Å survives, and raising
`RPATH` from 8 to 12 Å leaves χ(k) unchanged. An unchanged spectrum under
increasing `RPATH` then shows only that the filters are active, not that the
long-path contribution is negligible. Test `RPATH` together with looser
filters, e.g. `cards={"CRITERIA": "1.0 0.5"}` in the Python API.

FEFF8.5L's final summation holds at most 15,000 paths per absorber and ignores
the rest without any message. Loose `CRITERIA`, long `RPATH` or high `NLEG` can
exceed this. The managed runner compares FEFF's `list.dat` with the number of
paths the sum read, marks a truncated run as failed, and records the listed
count as `paths_listed` in `run_status.json`. Tighten `CRITERIA` or reduce
`RPATH`/`NLEG` if this occurs.
