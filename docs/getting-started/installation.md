# Installation

## Requirements

- Python 3.10, 3.11, or 3.12
- ASE (Atomic Simulation Environment)
- At least one MLIP backend

## Install from PyPI

Install AmorphGen with your preferred backend:

```bash
# MACE only
pip install "amorphgen[mace]"

# CHGNet only
pip install "amorphgen[chgnet]"

# SevenNet only
pip install "amorphgen[sevennet]"

# MACE + CHGNet (recommended default)
pip install "amorphgen[mace,chgnet]"

# SevenNet + CHGNet (alternative env for SevenNet users)
pip install "amorphgen[sevennet,chgnet]"

# Conflict-free convenience bundle (MACE + CHGNet + analysis)
pip install "amorphgen[all]"

# Development install
pip install "amorphgen[all,dev]"
```

:::{warning}
**Don't install MACE and SevenNet in the same environment.** SevenNet
depends on `e3nn>=0.5`, while pre-trained MACE foundation models
(`mace-mpa-0`, …) were pickled with `e3nn==0.4.x` and fail to load
against the newer `e3nn`. The `[all]` extra therefore includes only
MACE + CHGNet (not SevenNet).

`[mace]`, `[chgnet]`, and `[sevennet]` are each fine on their own —
just don't combine `[mace]` and `[sevennet]`. The recommended pattern
is two conda environments:

```bash
# env A — MACE + CHGNet (default)
conda create -n amorphgen python=3.11
conda activate amorphgen
pip install "amorphgen[mace,chgnet]"

# env B — SevenNet + CHGNet (when SevenNet is required)
conda create -n amorphgen-sevennet python=3.11
conda activate amorphgen-sevennet
pip install "amorphgen[sevennet,chgnet]"
```

CHGNet has no `e3nn` dependency and is safe alongside either backend.
See {doc}`../guides/backends` for the full explanation of the conflict.
:::

## Install from source

```bash
git clone https://github.com/SMTG-Bham/AmorphGen.git
cd AmorphGen
pip install -e ".[mace,chgnet,dev]"
```

## Backend compatibility

| Backend | PyPI package | GPU support |  
|---------|-------------|-------------| 
| MACE    | `mace-torch` | CUDA ✅ |  
| CHGNet  | `chgnet`    | CUDA ✅ |  
| SevenNet | `sevenn`   | CUDA ✅ |  
| Classical (LJ, Buckingham) | built-in | N/A |  

## HPC setup (SLURM)

```bash
# Load your cluster's CUDA module (name varies by site)
module load CUDA/11.8.0

# MACE + CHGNet env (recommended default)
conda create -n amorphgen python=3.11
conda activate amorphgen
pip install "amorphgen[mace,chgnet]"
```

If you also want SevenNet, create a second environment as described
in the warning above and switch between them in your SLURM scripts via
`source activate amorphgen` or `source activate amorphgen-sevennet`.

## Verify installation

```python
import amorphgen
print(amorphgen.__version__)  # 1.0.0

from amorphgen.utils.calculators import list_models
list_models()  # prints all available models grouped by backend
```
