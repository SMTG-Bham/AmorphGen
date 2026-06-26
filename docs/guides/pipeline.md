# Melt-and-quench pipeline

The melt-and-quench pipeline is a 7-stage molecular dynamics workflow that transforms a crystalline input into a realistic amorphous structure.

## Overview

```text
┌───────────┐    ┌───────────┐    ┌───────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌───────────┐
│ 1. Opt    │───▶│ 2. Pre-eq │───▶│ 3.Melt│───▶│ 4. Hi-eq │───▶│ 5.Quench│───▶│ 6. Lo-eq │───▶│ 7. Opt    │
│  300 K    │    │  300 K    │    │ →3000K│    │  3000 K  │    │ →300 K  │    │  300 K   │    │  final    │
└───────────┘    └───────────┘    └───────┘    └──────────┘    └─────────┘    └──────────┘    └───────────┘
```

## Stage descriptions

### Stage 1: Structure optimisation
Relaxes the input structure (cell + atomic positions) to remove any initial stress. Uses LBFGS with `UnitCellFilter` by default.

### Stage 2: Pre-melt equilibration
Short NVT equilibration at 300 K. Thermalises the system before the rapid heating stage, improving trajectory stability.

### Stage 3: Melt (heat ramp)
Linear temperature ramp from 300 K to the target melt temperature (default 3000 K). Uses **NPT with the Berendsen weak-coupling barostat** by default — the cell expands physically as the system heats.

### Stage 4: High-temperature equilibration
Holds the system at the melt temperature to ensure thorough melting and loss of crystalline memory. Uses **NPT with the Martyna-Tobias-Klein (MTK) Nose-Hoover-chain integrator** by default, giving true canonical fluctuations around the equilibrium melt volume. Users who want the legacy constant-volume behaviour can set `eq_high.ensemble: NVT`.

### Stage 5: Quench (cooling ramp)
Linear cooling from the melt temperature back to 300 K. The quench rate controls the degree of structural disorder. Uses NVT by default.

### Stage 6: Low-temperature equilibration
Equilibrates the quenched structure at 300 K to relax any residual thermal stress.

### Stage 7: Final optimisation
Final cell + position relaxation of the amorphous structure. Produces the output file.

## NPT integrators

AmorphGen exposes three NPT integrators via the per-stage `npt_method` YAML key:

| `npt_method` | ASE class | Use case |
|---|---|---|
| `berendsen` *(default)* | `NPTBerendsen` | Robust during the 300 K → 3000 K melt ramp; **does not** produce true canonical fluctuations (averages are correct, fluctuation-derived quantities like heat capacity are not). |
| `mtk` | `IsotropicMTKNPT` | Martyna-Tobias-Klein Nose-Hoover-chain NPT. True canonical fluctuations. Recommended for equilibration plateaux; may become unstable during rapid temperature ramps. |
| `parrinello-rahman` | `MelchionnaNPT` | Nose-Hoover + Parrinello-Rahman flexible-cell NPT. Allows the cell shape (not just volume) to change; useful for anisotropic glasses. Requires upper-triangular cell. |

### Stability knobs

Two parameters tune the Berendsen barostat for stiffer/slower volume control during the heating ramp:

| Key | Default | What it does |
|---|---|---|
| `taup_factor` | 10.0 | Ratio of barostat coupling time to thermostat coupling time (`taup = taup_factor * ttime`). Larger → slower, more stable barostat. Applied to Berendsen `taup` and MTK `pdamp`. |
| `compressibility_GPa` | 100.0 | Reference compressibility for the Berendsen barostat. The default (100 GPa) is liquid-like; oxides with bulk modulus 150–300 GPa benefit from 200 GPa for less aggressive volume control. |

**Example: tighten the melt ramp for a-In₂O₃ / a-Ga₂O₃ / a-HfO₂**

```yaml
melt:
  ensemble: NPT
  npt_method: berendsen
  taup_factor: 30.0              # slower barostat (3× default)
  compressibility_GPa: 200.0     # stiffer, oxide-realistic
```

On a Cu/EMT benchmark at 1500 K over 300 fs, these settings reduce the maximum volume excursion from 6.7 % (defaults) to 1.8 % — a 74 % reduction.

**Example: true canonical fluctuations at the equilibration plateau**

```yaml
eq_high:
  ensemble: NPT
  npt_method: mtk
```

This is the new default for `eq_high`; the legacy NVT behaviour is restored by setting `ensemble: NVT`.

## Running specific stages

You can run a subset of stages using the `--stages` flag:

```bash
# Only quench and post-process (e.g. after restarting from a snapshot)
amorphgen POSCAR --model mace-mpa-0 --stages 5 6 7
```

## Customising parameters

See the {doc}`/api/config` page for all available configuration keys.
