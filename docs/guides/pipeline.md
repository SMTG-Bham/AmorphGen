# Melt-and-quench (MQ) pipeline

The melt-and-quench pipeline is a 7-stage molecular dynamics workflow that transforms a crystalline input into an amorphous structure.

## Overview

```text
┌───────────┐    ┌───────────┐    ┌─────────┐    ┌────────────┐    ┌─────────┐    ┌───────────┐    ┌───────────┐
│ 1. Opt    │───▶│ 2. Pre-eq │───▶│ 3. Melt │───▶│ 4. High-T  │───▶│ 5.Quench│───▶│ 6. Low-T  │───▶│ 7. Opt    │
│ crystal   │    │  low T    │    │low →high│    │   eq       │    │high →low│    │   eq      │    │  final    │
└───────────┘    └───────────┘    └─────────┘    └────────────┘    └─────────┘    └───────────┘    └───────────┘
```

Temperatures are configurable per system. The `low T` value (default 300 K) sets the equilibration temperature in stages 2 and 6 plus the quench end-point in stage 5; the `high T` value (default 3000 K) sets the melt end-point in stage 3 and the equilibration temperature in stage 4 plus the quench start-point in stage 5. Both can be overridden via CLI flags or YAML config.

## Stage descriptions

### Stage 1: Structure optimisation
Relaxes the input structure (cell + atomic positions) to remove any initial stress. Uses LBFGS with `UnitCellFilter` by default.

### Stage 2: Pre-melt equilibration
Short NVT equilibration at 300 K. Thermalises the system before the rapid heating stage, improving trajectory stability.

### Stage 3: Melt (heat ramp)
Linear temperature ramp from 300 K to the target melt temperature (default 3000 K). Uses NVT with velocity rescaling.

### Stage 4: High-temperature equilibration
Holds the system at the melt temperature to ensure thorough melting and loss of crystalline memory.

### Stage 5: Quench (cooling ramp)
Linear cooling from the melt temperature back to 300 K. The quench rate controls the degree of structural disorder.

### Stage 6: Low-temperature equilibration
Equilibrates the quenched structure at 300 K to relax any residual thermal stress.

### Stage 7: Final optimisation
Final cell + position relaxation of the amorphous structure. Produces the output file.

## Running specific stages

You can run a subset of stages using the `--stages` flag:

```bash
# Only quench and post-process (e.g. after restarting from a snapshot)
amorphgen POSCAR --model mace-mpa-0 --stages 5 6 7
```

## Customising parameters

See the {doc}`/api/config` page for all available configuration keys.
