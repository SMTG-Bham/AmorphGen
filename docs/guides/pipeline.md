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
