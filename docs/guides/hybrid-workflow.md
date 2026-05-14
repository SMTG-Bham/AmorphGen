# Hybrid workflow

Combine random structure generation with melt-quench to efficiently sample the amorphous energy landscape — without paying the cost of melting from a crystal.

## Concept

```text
Random-gen structures → anneal at high T → quench → eq → opt
        (already disordered, skip stages 1-3)
```

Because random-placement structures are already disordered, you can skip the crystal-melt pre-stages and start directly at high T, then quench. This is significantly cheaper than the full 7-stage pipeline and produces a diverse ensemble of amorphous structures.

## Single-command CLI: `--hybrid-ensemble`

```bash
# 1. Generate 20 random structures (any composition)
amorphgen --random-gen --composition "TiO2*8" -n 20 \
    --relax --device cuda --model chgnet --format vasp \
    -o random_TiO2/

# 2. Run hybrid (stages 4-5-6-7) on each, in one CLI call
amorphgen --hybrid-ensemble --input-dir random_TiO2/ \
    --config hybrid.yaml --device cuda --model chgnet \
    -o tio2_hybrid/
```

Output layout:

```text
tio2_hybrid/
├── quench_runs/
│   ├── run_0000/  # stages 4-7 outputs for input 0
│   ├── run_0001/  # ...
│   └── run_0019/
└── final/
    ├── hybrid_0000.vasp
    ├── ...
    └── hybrid_0019.vasp
```

The `run_NNNN/` index matches the source snapshot index parsed from the input
filename (`snapshot_NNNN_*.xyz`). When splitting the per-input runs across
SLURM array tasks, point all tasks at the same `quench_runs/` directory —
AmorphGen handles the per-snapshot naming. See {doc}`mq-ensemble`'s "HPC
job-array tip" for the full SLURM template.

`--resume` is honoured at every step — re-running the command picks up incomplete runs.

## Recommended `hybrid.yaml` for an oxide

```yaml
model: chgnet
device: cuda

# Stage 4: anneal at high T (within MLIP training window)
eq_high:
  ensemble: NVT
  T: 3000              # above Tm but inside chgnet/MACE training data
  steps: 20000         # 20 ps anneal — random inputs need less than crystal-melt
  timestep: 0.5
  friction: 0.01

# Stage 5: cool 3000 → 300 K at 100 K/ps
quench:
  ensemble: NVT
  T_start: 3000
  T_end: 300
  T_step: -100
  rate: 100            # K/ps
  timestep: 0.5
  friction: 0.01

# Stage 6: equilibrate at 300 K
eq_low:
  ensemble: NVT
  T: 300
  steps: 5000          # 5 ps
  timestep: 0.5
  friction: 0.01

# Stage 7: final relax
opt:
  fmax: 0.05
  optimizer: LBFGS
  cell_filter: cubic   # preserves cubic shape from random-gen
```

## Why use the hybrid workflow?

- **Faster** than running full 7-stage pipelines on N structures from crystals (skips crystal opt + premelt + heating ramp per structure).
- **Better sampling** — random initial configurations provide diverse starting points.
- **Defensible cell volume** — random-gen sets a sensible amorphous density up front; NVT preserves it.
- **Annealing at chgnet/MACE-trained T** (e.g. 3000 K) avoids extrapolation while still being above the melting point of most oxides.

## Comparison to `--mq-ensemble` (crystal melt-quench)

| Feature | `--mq-ensemble` | `--hybrid-ensemble` |
|---------|-----------------|---------------------|
| Starting structure | Crystalline supercell | Disordered (random-gen output) |
| Stages run | 1-2-3-4 + N×(5-6-7) | N×(4-5-6-7) |
| Crystal melt time | Yes (long stage 3) | No |
| Cost per structure | High | Medium |

For tightly comparing to published DFT melt-quench results, use `--mq-ensemble`. For rapidly generating large ensembles for screening, use `--hybrid-ensemble`.

## Example: a-TiO₂

See Tutorial 5 ({doc}`/tutorials/index`) for a complete worked example with a-TiO₂ (Ti₈O₁₆, 24 atoms): random gen → high-T equilibration → 5× batch quench → ensemble structural analysis. The CLI commands above generalise to any oxide; substitute your composition for `TiO2*8`.
