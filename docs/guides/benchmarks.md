# Benchmarks: Random Generation Methods

This page records exploratory comparisons of AmorphGen workflows on SiO₂,
Si, and Li₂ZrCl₆, followed by example commands for running a new comparison.

The historical result tables do not include complete per-run inputs, random
seeds, model versions, raw outputs or citations for the experimental rows.
Treat those values as illustrative observations, not verified reference data
or evidence that one backend is generally more accurate. The examples below
demonstrate the current interface; they do not reproduce every table row.

## Methods compared

| Method | Description | Recorded time (48–72 atoms, Mac M-series) |
|--------|-------------|----------------------------------------|
| **SC+opt** | SC random placement → static optimisation | ~2 min |
| **CHGNet MQ** | SC random → hybrid melt-quench (CHGNet, MPS) | ~8-20 min |
| **MACE MQ** | SC random → hybrid melt-quench (MACE, CPU) | ~1-2 hours |

The hybrid melt-quench (MQ) workflow skips pre-melt equilibration and heating (stages 2–3)
since the random structure is already disordered:

```
Random (SC) → Optimise (1) → High-T equilibrate (4) → Quench (5) → Low-T equilibrate (6) → Final optimise (7)
```

Times depend on the atom count, model, device, precision and number of MD
steps. Measure a representative run for your own allocation.

(results)=
## Historical results

### SiO₂ (48 atoms: Si₁₆O₃₂)

| Method | Density (g/cm³) | Si-O CN | CN=4 (%) | O-Si-O angle (°) | Si-O dist (Å) |
|--------|:-:|:-:|:-:|:-:|:-:|
| SC+opt (CHGNet) | 2.23 | 4.0 | 85% | 108.9 ± 14.4 | 1.665 |
| CHGNet MQ (short) | 2.25 | 4.0 | 100% | 109.4 ± 6.4 | 1.645 |
| CHGNet MQ (long) | 1.94 | 4.0 | 100% | 109.5 ± 5.1 | 1.634 |
| MACE MQ | 2.25 | 4.0 | 100% | — | — |
| Experiment | 2.20 | 4.0 | ~100% | 109.5 ± 10 | 1.620 |

The listed MQ runs reached 100% tetrahedral Si, while the CHGNet long run
also had a lower density. Coordination alone is therefore insufficient to
assess agreement with a reference structure.

### Si (40 atoms)

| Method | Density (g/cm³) | Si-Si CN | CN=4 (%) |
|--------|:-:|:-:|:-:|
| SC+opt (CHGNet) | 2.24 | 4.0 | 80% |
| CHGNet MQ (short) | 2.28 | 4.0 | 75% |
| CHGNet MQ (long) | 2.31 | 4.0 | 90% |
| MACE MQ | 2.36 | 4.3 | 75% |
| Experiment | 2.29 | 4.0 | ~100% |

The listed long CHGNet run had a larger CN=4 fraction than the short run.
A controlled ensemble comparison is needed to separate protocol effects
from seed-to-seed variation.

### Li₂ZrCl₆ (72 atoms: Li₁₆Zr₈Cl₄₈)

| Method | Density (g/cm³) | Zr-Cl CN | CN=6 (%) | Li-Cl CN | CN=6 (%) |
|--------|:-:|:-:|:-:|:-:|:-:|
| SC+opt (CHGNet) | 1.76 | 5.5 | 52% | 4.1 | 1% |
| CHGNet MQ (NVT) | 1.76 | 5.8 | 75% | 4.4 | 0% |
| CHGNet MQ (dense) | 1.82 | 5.5 | 50% | 4.6 | 6% |
| MACE MQ | 2.40 | 6.0 | 100% | 5.4 | 56% |
| Experiment | 2.39 | 6.0 | 100% | 6.0 | 100% |

These runs used different model/protocol combinations. In particular, the
MACE example below fixes the cell at an input density of 2.4 g/cm³; agreement
with that density is imposed by the protocol and cannot demonstrate a model
density prediction. These observations do not establish a ranking for chloride
systems generally.

(recommendations)=
## Designing a comparison

Keep composition, initial structures, density constraints, precision, MD
schedule and analysis cutoffs consistent when comparing calculators. To
compare static relaxation with hybrid MD, retain the initial seeds and report
the extra MD sampling cost alongside the structural metrics. Use several
seeds and report the spread, rather than selecting one favourable structure.

(reproduction)=
## Example protocols

Save each YAML block under the filename shown before running its commands.
The temperatures and durations are example inputs to validate for your system.

### SiO₂ with CHGNet MQ

```yaml
# SiO2_chgnet_mq.yaml
model: chgnet
device: auto        # uses an available supported device; override if needed
default_dtype: float32
seed: 42

opt:
  fmax: 0.05
  max_steps: 200
  cell_filter: cubic

random_gen:
  composition:
    Si: 16
    O: 32
  n_structures: 1
  output_format: xyz
  target_density: 2.2
  target_cn:
    Si: 4
    O: 2

eq_high:
  ensemble: NVT
  T: 3000
  steps: 5000
  timestep: 0.5

quench:
  ensemble: NVT
  T_start: 3000
  T_end: 300
  T_step: -100
  steps_per_T: 500
  timestep: 0.5

eq_low:
  ensemble: NVT
  T: 300
  steps: 2000
```

```bash
# Generate
amorphgen --random-gen --config SiO2_chgnet_mq.yaml --work-dir SiO2_random

# Run hybrid MQ (stages 1,4,5,6,7)
amorphgen SiO2_random/random_initial/random_0000.xyz \
    --config SiO2_chgnet_mq.yaml \
    --stages 1 4 5 6 7 \
    --work-dir SiO2_mq

# Analyse
amorphgen --analyse SiO2_mq/stage7_opt.xyz --save-plot SiO2_mq/plots
```

### Li₂ZrCl₆ with MACE MQ

```yaml
# Li2ZrCl6_mace_mq.yaml
model: mace-mpa-0
device: cpu          # or cuda on HPC
default_dtype: float64
seed: 42

opt:
  fmax: 0.05
  max_steps: 300
  cell_filter: none  # fixed cell at the chosen input density

random_gen:
  composition:
    Li: 16
    Zr: 8
    Cl: 48
  n_structures: 1
  output_format: xyz
  target_density: 2.4
  target_cn:
    Zr: 6
    Li: 6
  dmax:
    Zr-Cl: 3.2
    Li-Cl: 3.2

eq_high:
  ensemble: NVT
  T: 1200
  steps: 10000
  timestep: 0.5

quench:
  ensemble: NVT
  T_start: 1200
  T_end: 300
  T_step: -50
  steps_per_T: 500
  timestep: 0.5

eq_low:
  ensemble: NVT
  T: 300
  steps: 5000
```

```bash
amorphgen --random-gen --config Li2ZrCl6_mace_mq.yaml --work-dir LZC_random
amorphgen LZC_random/random_initial/random_0000.xyz \
    --config Li2ZrCl6_mace_mq.yaml \
    --stages 1 4 5 6 7 \
    --work-dir LZC_mq
amorphgen --analyse LZC_mq/stage7_opt.xyz --save-plot LZC_mq/plots
```

### Python API

```python
from amorphgen.pipeline.random_gen import generate_random
from amorphgen import MeltQuenchPipeline
from amorphgen.analysis import StructureAnalyser
from ase.io import write

# Step 1: Generate
atoms = generate_random(
    composition={"Si": 16, "O": 32},
    target_density=2.2,
    target_cn={"Si": 4, "O": 2},
    seed=42,
)
write("random_SiO2.xyz", atoms, format="extxyz")

# Step 2: Hybrid MQ
pipe = MeltQuenchPipeline(
    input_file="random_SiO2.xyz",
    work_dir="SiO2_mq",
    cfg_override={
        "model": "chgnet", "device": "auto", "seed": 42,
        "opt": {"fmax": 0.05, "cell_filter": "cubic"},
        "eq_high": {"ensemble": "NVT", "T": 3000, "steps": 5000},
        "quench": {"ensemble": "NVT", "T_start": 3000, "T_end": 300,
                   "T_step": -100, "steps_per_T": 500, "timestep": 0.5},
        "eq_low": {"T": 300, "steps": 2000},
    },
)
pipe.run(stages=[1, 4, 5, 6, 7])

# Step 3: Analyse
sa = StructureAnalyser("SiO2_mq/stage7_opt.xyz", cutoff="auto")
sa.summary()
sa.plot(output_dir="SiO2_mq/plots")
```

## Equilibration convergence

Use the convergence report to inspect equilibration. Its diagnostics do not
on their own establish decorrelation or convergence of every property:

```python
from amorphgen.utils.equilibration import convergence_report

# From log file (fast, energy + temperature only)
report = convergence_report(
    "SiO2_mq/stage4_eq.log",
    n_atoms=48,
    T_target=3000,
    output_dir="SiO2_mq/convergence",
)

# From trajectory (full analysis: energy, MSD, RDF, CN)
report = convergence_report(
    "SiO2_mq/stage4_eq_traj.xyz",
    timestep_fs=0.5,
    frame_stride=100,  # default AmorphGen trajectory interval in MD steps
    T_target=3000,
    pairs_cn=[("Si", "O", 4.0), ("O", "Si", 2.0)],
    output_dir="SiO2_mq/convergence",
)
```

Inspect energy drift, the block-average diagnostic, MSD and RDF agreement
between time windows together. Choose acceptance criteria for the intended
property, and check that extending the trajectory does not materially change
it. `frame_stride` is the number of integration steps between saved frames;
set it to match the trajectory being analysed.

## Random-placement comparison configs

The downloadable configs compare plain random rejection sampling with explicit
SC (Seed-Coordinate) targets. These configs generate structures only; add
`--relax` for a static optimisation and specify relaxation limits on the CLI
(e.g. `--opt-steps 500`). Use `--device cpu` or `--device cuda` if MPS is not
available.

| System / starting density | Plain placement | SC placement |
|---|---|---|
| SiO₂ / 2.2 g/cm³ | {download}`SiO2_std.yaml <benchmark_configs/SiO2_std.yaml>` | {download}`SiO2_sc.yaml <benchmark_configs/SiO2_sc.yaml>` |
| SiO₂ / estimated | {download}`SiO2_auto.yaml <benchmark_configs/SiO2_auto.yaml>` | {download}`SiO2_auto_sc.yaml <benchmark_configs/SiO2_auto_sc.yaml>` |
| Li₂ZrCl₆ / 2.4 g/cm³ | {download}`Li2ZrCl6_std.yaml <benchmark_configs/Li2ZrCl6_std.yaml>` | {download}`Li2ZrCl6_sc.yaml <benchmark_configs/Li2ZrCl6_sc.yaml>` |

The plain-placement configs set `random_gen.target_cn: {}` explicitly. Omitting
that key enables automatically inferred SC targets in the current release.
