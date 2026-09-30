# Hybrid workflow

Combine random structure generation with melt-quench to efficiently sample the amorphous energy landscape, without paying the cost of melting from a crystal.

## Concept

```text
Random-gen structures → anneal at high T → quench → eq → opt
        (already disordered, skip stages 1-3)
```

Because random-placement structures are already disordered, you can skip the crystal-melt pre-stages and start directly at high T, then quench. This avoids the crystal preparation stages. Validate the anneal duration and temperature before treating the outputs as an amorphous ensemble.

## Single-command CLI: `--hybrid-ensemble`

```bash
# 1. Generate 20 random structures (any composition)
amorphgen --random-gen --composition "TiO2*8" -n 20 \
    --relax --device cuda --model chgnet --format vasp \
    -o random_TiO2/

# 2. Run hybrid (stages 4-5-6-7) on each, in one CLI call
amorphgen --hybrid-ensemble --input-dir random_TiO2/random_opt/ \
    --config hybrid.yaml --device cuda --model chgnet --format vasp \
    --resume -o tio2_hybrid/
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

For `snapshot_NNNN_*.xyz` inputs, the `run_NNNN/` index comes from the
filename. Other filenames, including random-gen outputs, use their position
in the sorted input list. A single input with the ASE engine writes directly
inside `quench_runs/`; torch-sim keeps the `run_NNNN/` directory.

Give each SLURM array task its own output directory to avoid collisions.
See the "HPC job-array tip" in {doc}`mq-ensemble` for the per-task pattern;
use `--batch-stages 4 5 6 7` for hybrid inputs.

`--hybrid-ensemble` processes all inputs in the first matching format (`*.xyz`, `*.extxyz`, `*.vasp`, `*.cif`, then `POSCAR*`); `-n` and `--n-runs` do not limit this mode. Use a directory containing only the intended structures.

With `--resume`, completed runs are skipped and interrupted MD continues from saved frames. Keep the input set, ordering and protocol unchanged when resuming; final files are collected again.

### Batched on a GPU with torch-sim

For MACE, SevenNet or Lennard-Jones on a CUDA GPU, add `--engine torchsim`
(needs Python 3.12 or newer and `pip install "amorphgen[mace,torchsim]"` for MACE) to run
stages 4 to 7 for all inputs together in batched calls:

```bash
amorphgen --hybrid-ensemble --input-dir random_TiO2/random_opt/ \
    --config hybrid.yaml --device cuda --model mace-mpa-0 \
    --engine torchsim -o tio2_hybrid/ --resume
```

Per-run files and the `final/` collection are the same as above. The MD
stages must be NVT (torch-sim's NPT is not mapped), the chunk size follows a
GPU memory probe (`--batch-size auto`) and `--resume` continues a killed job
from the last frame every run of a chunk has reached. Throughput depends on
the model, system sizes and device; benchmark a representative chunk.
Details in {doc}`backends`.

## Example `hybrid.yaml` for an oxide

```yaml
model: chgnet
device: cuda

# Stage 4: anneal at high T (validate for the material and model)
eq_high:
  ensemble: NVT
  T: 3000              # illustrative temperature; check model reliability
  steps: 20000         # 10 ps anneal at 0.5 fs
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
  steps: 5000          # 2.5 ps at 0.5 fs
  timestep: 0.5
  friction: 0.01

# Stage 7: final relax
opt:
  fmax: 0.05
  optimizer: LBFGS
  cell_filter: cubic   # preserves cubic shape from random-gen
```

## When to use the hybrid workflow

- Faster than running full 7-stage pipelines on N structures from crystals (skips crystal opt + premelt + heating ramp per structure).
- Better sampling: random initial configurations provide diverse starting points.
- Density control: random-gen provides a starting density estimate or accepts `--target-density`. NVT preserves that volume during MD; `cell_filter: cubic` still changes it during relaxation. Use `--cell-filter none` in both generation and hybrid commands to keep the cell fixed throughout.
- Choose the anneal temperature and duration for the material and validate the resulting structure; a temperature alone does not establish that an MLIP is operating within its training distribution. See {doc}`best-practices`.

## Comparison to `--mq-ensemble` (crystal melt-quench)

| Feature | `--mq-ensemble` | `--hybrid-ensemble` |
|---------|-----------------|---------------------|
| Starting structure | Crystalline supercell | Disordered (random-gen output) |
| Stages run | 1-2-3-4 + N×(5-6-7) | N×(4-5-6-7) |
| Crystal melt time | Yes (long stage 3) | No |
| Shared preparation | Stages 1–4 run once | Each input is annealed separately |
| Method reporting | Record the shared melt and snapshot spacing | Record random placement, density and anneal settings |

Choose the route that matches the preparation protocol you intend to study. Comparing with published results also requires matching composition, density, temperature schedule and analysis settings.

## Example: a-TiO₂

See Tutorial 5 ({doc}`/tutorials/index`) for a complete worked example with a-TiO₂ (Ti₈O₁₆, 24 atoms): random gen → high-T equilibration → 5× batch quench → ensemble structural analysis. The CLI commands above generalise to any oxide; substitute your composition for `TiO2*8`.
