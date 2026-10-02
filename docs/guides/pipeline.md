# Melt-and-quench pipeline

The melt-and-quench pipeline is a 7-stage molecular dynamics workflow that generates an amorphous candidate from a crystalline input. Validate the resulting structure and the chosen calculator for your material.

## Overview

```text
┌───────────┐    ┌───────────┐    ┌───────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌───────────┐
│ 1. Opt    │───▶│ 2. Pre-eq │───▶│ 3.Melt│───▶│ 4. Hi-eq │───▶│ 5.Quench│───▶│ 6. Lo-eq │───▶│ 7. Opt    │
│  relax    │    │  300 K    │    │ →3000K│    │  3000 K  │    │ →300 K  │    │  300 K   │    │  final    │
└───────────┘    └───────────┘    └───────┘    └──────────┘    └─────────┘    └──────────┘    └───────────┘
```

## Stage descriptions

### Stage 1: Structure optimisation
Relaxes the input structure (cell + atomic positions) to remove any initial stress. Uses LBFGS with `FrechetCellFilter` by default. Optimisation minimises the potential energy; it is not an MD stage at a prescribed temperature.

### Stage 2: Pre-melt equilibration
NVT equilibration at 300 K (100000 steps at 0.5 fs, or 50 ps, by default). Thermalises the system before heating.

### Stage 3: Melt (heat ramp)
Segmented temperature ramp from 300 K to the target melt temperature (default 3000 K). Uses **NPT with the Berendsen weak-coupling barostat** by default, the cell volume can change as the system heats.

### Stage 4: High-temperature equilibration
Holds the system at the melt temperature (20000 steps at 0.5 fs, or 10 ps, by default). Check that the system has melted and lost crystalline memory before selecting snapshots. Uses **NPT with the Martyna-Tobias-Klein (MTK) Nose-Hoover-chain integrator** by default, which samples the NPT ensemble. Users who want the legacy constant-volume behaviour can set `eq_high.ensemble: NVT`.

### Stage 5: Quench (cooling ramp)
Segmented cooling from the melt temperature back to 300 K. The quench rate controls the degree of structural disorder. Uses NVT by default.

### Stage 6: Low-temperature equilibration
Equilibrates the quenched structure at 300 K using NVT (20000 steps at 0.5 fs, or 10 ps, by default). The cell is fixed during this stage.

### Stage 7: Final optimisation
Final cell + position relaxation of the amorphous structure. Writes `stage7_opt.xyz` and `stage7_opt.cif`; `--format vasp` also writes `stage7_opt.vasp`. Stage 7 inherits `opt:` settings, with any `final_opt:` keys applied on top.

## NPT integrators

AmorphGen exposes three NPT integrators via the per-stage `npt_method` YAML key:

| `npt_method` | ASE class | Use case |
|---|---|---|
| `berendsen` *(default)* | `NPTBerendsen` | Robust during the 300 K → 3000 K melt ramp; **does not** sample the NPT ensemble correctly; avoid using its fluctuations to estimate thermodynamic response functions. |
| `mtk` | `IsotropicMTKNPT` | Martyna-Tobias-Klein Nose-Hoover-chain NPT. Samples NPT fluctuations. Suitable for equilibration plateaux; may become unstable during rapid temperature ramps. |
| `parrinello-rahman` | `MelchionnaNPT` | Nose-Hoover + Parrinello-Rahman flexible-cell NPT. Allows the cell shape (not just volume) to change; useful for anisotropic glasses. Requires upper-triangular cell. |

### Stability knobs

Two parameters tune the Berendsen barostat for stiffer/slower volume control during the heating ramp:

| Key | Default | What it does |
|---|---|---|
| `taup_factor` | 10.0 | Ratio of barostat coupling time to thermostat coupling time (`taup = taup_factor * ttime`). Larger → slower, more stable barostat. Applied to Berendsen `taup` and MTK `pdamp`. |
| `compressibility_GPa` | 100.0 | Reference bulk modulus in GPa. The Berendsen compressibility is its inverse; larger values reduce the volume response. |

**Example: slower volume response during the melt ramp**

```yaml
melt:
  ensemble: NPT
  npt_method: berendsen
  taup_factor: 30.0              # slower barostat (3× default)
  compressibility_GPa: 200.0     # larger reference bulk modulus
```

Choose these coupling parameters for the material and inspect the volume trajectory; they do not establish that the calculator predicts the correct density.

**Example: NPT sampling at the equilibration plateau**

```yaml
eq_high:
  ensemble: NPT
  npt_method: mtk
```

This is the default for `eq_high`; the legacy NVT behaviour is restored by setting `ensemble: NVT`.

## Running specific stages

You can run a subset of stages using the `--stages` flag:

```bash
# Only quench and post-process (e.g. after restarting from a snapshot)
amorphgen snapshot.xyz --model mace-mpa-0 --stages 5 6 7 -o quench_run/
```

The input must be appropriate for the first requested stage; selecting stages 5–7 does not create a melt. Temperatures are independent settings: when changing the melt endpoint, also set `eq_high.T` and `quench.T_start` to match.

## Customising parameters

See the {doc}`/api/config` page for all available configuration keys.

## Run manifest

`MeltQuenchPipeline.run()` writes `run_manifest.json` in its output directory,
including when invoked through the CLI's melt-and-quench pipeline. This is the
machine-readable record alongside the existing `pipeline_summary.log`.
It is created before reading the input or loading the calculator, and updated
atomically before and after each stage.

The JSON has `schema_version: 1` and an `attempts` list. Each invocation appends
an attempt, including a resume that skips every stage. Earlier attempts keep
their configuration, timings, and errors. A resume from an older run with no
manifest starts a new history; it cannot reconstruct that run's provenance.

Each attempt records:

- `package_version`, Python version, platform, UTC start/end times, and elapsed seconds.
- The merged `config`, base `seed`, and effective `seed_index`. MD streams are
  derived from `(seed, stage, seed_index)`; a null seed means unseeded noise.
- The actual `engine` (`ase` for this pipeline), plus requested and resolved
  calculator `precision` and `device` where available.
- `model` name, calculator class, local checkpoint path where applicable,
  SHA-256 digest, and hash source.
- Requested stages, original and selected input paths, the resume flag, and
  each stage's status and timing. Checkpoint skips are recorded separately
  from stages executed in the current attempt.
- Attempt status (`running`, `completed`, `failed`, or `interrupted`) and the
  exception type/message on failure. An unstarted stage remains `pending`.

For local model files, `model.hash_source` is `file` and `model.sha256` hashes
the checkpoint bytes. For foundation or injected models exposing loaded
weights, `state_dict-v1` hashes sorted tensor names, shapes, dtypes, and bytes;
this digest depends on the loaded precision and is not a checkpoint-file hash.
When a hash cannot be obtained, it is null with a `hash_unavailable_reason`.
Injected calculators are identified independently of the configured default
model. Python objects in configuration, such as reference calculators, are
represented by their qualified class name, not serialised object state.

Caught exceptions and keyboard interruptions are saved before being re-raised.
A forced process kill or power loss leaves the last saved attempt/stage
`running`; the next invocation retains that record. The manifest records
provenance and progress, not the full thermostat or random-generator state
needed for a bit-identical restart. Use one pipeline writer per output directory.
