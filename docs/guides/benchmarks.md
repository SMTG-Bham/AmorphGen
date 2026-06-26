# Benchmarks: Random Generation Methods

This page compares different amorphous structure generation approaches
available in AmorphGen, tested on SiO₂, Si, and Li₂ZrCl₆.

## Methods compared

| Method | Description | Typical time (48-72 atoms, Mac M-series) |
|--------|-------------|----------------------------------------|
| **SC+opt** | SC random placement → static optimisation | ~2 min |
| **CHGNet MQ** | SC random → hybrid melt-quench (CHGNet, MPS) | ~8-20 min |
| **MACE MQ** | SC random → hybrid melt-quench (MACE, CPU) | ~1-2 hours |

The hybrid melt-quench (MQ) workflow skips the melt stage (stages 2-3)
since the random structure is already disordered:

```
Random (SC) → Optimise → High-T equilibrate → Quench → Low-T equilibrate → Final optimise
  (stage 1)    (stage 4)      (stage 5)        (stage 6)       (stage 7)
```

## Results

### SiO₂ (48 atoms: Si₁₆O₃₂)

| Method | Density (g/cm³) | Si-O CN | CN=4 (%) | O-Si-O angle (°) | Si-O dist (Å) |
|--------|:-:|:-:|:-:|:-:|:-:|
| SC+opt (CHGNet) | 2.23 | 4.0 | 85% | 108.9 ± 14.4 | 1.665 |
| CHGNet MQ (short) | 2.25 | 4.0 | 100% | 109.4 ± 6.4 | 1.645 |
| CHGNet MQ (long) | 1.94 | 4.0 | 100% | 109.5 ± 5.1 | 1.634 |
| **MACE MQ** | **2.25** | **4.0** | **100%** | — | — |
| Experiment | 2.20 | 4.0 | ~100% | 109.5 ± 10 | 1.620 |

**Key finding:** Both CHGNet and MACE MQ achieve 100% tetrahedral Si
coordination. CHGNet MQ gives the tightest bond angle distribution
(5.1° std vs 14.4° for static optimisation).

### Si (40 atoms)

| Method | Density (g/cm³) | Si-Si CN | CN=4 (%) |
|--------|:-:|:-:|:-:|
| SC+opt (CHGNet) | 2.24 | 4.0 | 80% |
| CHGNet MQ (short) | 2.28 | 4.0 | 75% |
| CHGNet MQ (long) | 2.31 | 4.0 | 90% |
| MACE MQ | 2.36 | 4.3 | 75% |
| Experiment | 2.29 | 4.0 | ~100% |

**Key finding:** Longer equilibration improves CN=4 fraction (90% with
long CHGNet MQ). Pure Si needs slow cooling rates for tetrahedral
network formation.

### Li₂ZrCl₆ (72 atoms: Li₁₆Zr₈Cl₄₈)

| Method | Density (g/cm³) | Zr-Cl CN | CN=6 (%) | Li-Cl CN | CN=6 (%) |
|--------|:-:|:-:|:-:|:-:|:-:|
| SC+opt (CHGNet) | 1.76 | 5.5 | 52% | 4.1 | 1% |
| CHGNet MQ (NVT) | 1.76 | 5.8 | 75% | 4.4 | 0% |
| CHGNet MQ (dense) | 1.82 | 5.5 | 50% | 4.6 | 6% |
| **MACE MQ** | **2.40** | **6.0** | **100%** | **5.4** | **56%** |
| Experiment | 2.39 | 6.0 | 100% | 6.0 | 100% |

**Key finding:** MACE is dramatically better for chloride systems —
correct density (2.40 vs 1.76), perfect Zr octahedra (100% CN=6),
and much improved Li coordination (56% vs 0-6% CN=6). CHGNet
significantly underestimates the density of chloride systems.

## Recommendations

| System type | Recommended method | Notes |
|-------------|-------------------|-------|
| **Oxides** (SiO₂, In₂O₃, Ga₂O₃) | CHGNet MQ on MPS/GPU | Fast, accurate CN and angles |
| **Pure elements** (Si, Ge) | CHGNet MQ (long) on MPS/GPU | Need slow cooling for CN=4 |
| **Chlorides** (Li₂MCl₆) | **MACE MQ on CUDA** | CHGNet fails on density; MACE essential |
| **Quick screening** | SC+opt (any backend) | 2 min, gives ~85% correct CN |

## Reproduction

### SiO₂ with CHGNet MQ

```yaml
# SiO2_chgnet_mq.yaml
model: chgnet
device: mps
default_dtype: float32

opt:
  fmax: 0.05
  max_steps: 200
  cell_filter: cubic

random_gen:
  composition:
    Si: 16
    O: 32
  target_density: 2.2
  target_cn:
    Si: 4
    O: 2

eq_high:
  ensemble: NVT
  T: 3000
  steps: 5000

quench:
  ensemble: NVT
  T_start: 3000
  T_end: 300
  T_step: -100
  steps_per_T: 500

eq_low:
  ensemble: NVT
  T: 300
  steps: 2000
```

```bash
# Generate
amorphgen --random-gen --config SiO2_chgnet_mq.yaml --work-dir SiO2_random

# Run hybrid MQ (stages 1,4,5,6,7)
amorphgen SiO2_random/random_0000.xyz \
    --config SiO2_chgnet_mq.yaml \
    --stages 1 4 5 6 7 \
    --work-dir SiO2_mq

# Analyse
amorphgen --analyse --input-dir SiO2_mq/ --save-plot SiO2_mq/plots
```

### Li₂ZrCl₆ with MACE MQ

```yaml
# Li2ZrCl6_mace_mq.yaml
model: mace-mpa-0
device: cpu          # or cuda on HPC
default_dtype: float64

opt:
  fmax: 0.05
  max_steps: 300
  cell_filter: none  # fixed cell at experimental density

random_gen:
  composition:
    Li: 16
    Zr: 8
    Cl: 48
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

quench:
  ensemble: NVT
  T_start: 1200
  T_end: 300
  T_step: -50
  steps_per_T: 500

eq_low:
  ensemble: NVT
  T: 300
  steps: 5000
```

```bash
amorphgen --random-gen --config Li2ZrCl6_mace_mq.yaml --work-dir LZC_random
amorphgen LZC_random/random_0000.xyz \
    --config Li2ZrCl6_mace_mq.yaml \
    --stages 1 4 5 6 7 \
    --work-dir LZC_mq
amorphgen --analyse --input-dir LZC_mq/ --save-plot LZC_mq/plots
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
        "model": "chgnet", "device": "mps",
        "opt": {"fmax": 0.05, "cell_filter": "cubic"},
        "eq_high": {"T": 3000, "steps": 5000},
        "quench": {"T_start": 3000, "T_end": 300, "T_step": -100},
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

Use the convergence report to verify that MD equilibration has converged:

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
    "SiO2_mq/stage4_eq.xyz",
    timestep_fs=0.5,
    T_target=3000,
    pairs_cn=[("Si", "O", 4.0), ("O", "Si", 2.0)],
    output_dir="SiO2_mq/convergence",
)
```

Key convergence criteria:
- **Energy drift** < 0.001 eV/atom/ps
- **Block average test** PASSED (block means within 2× SEM)
- **MSD** linear (liquid) at high T, plateau (glass) at low T
- **RDF** overlapping across time windows
