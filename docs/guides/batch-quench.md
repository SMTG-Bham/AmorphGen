# Batch quench

Quench multiple snapshot structures through the final stages of the pipeline in a single run.

## Use case

After a high-temperature equilibration, you may want to extract multiple uncorrelated snapshots and quench each independently. This produces an ensemble of amorphous structures from a single melt trajectory.

## CLI usage

```bash
amorphgen --batch-quench \
    --snapshot-dir snapshots/ \
    --model mace-mpa-0 \
    --device cuda \
    --batch-stages 5 6 7 \
    --resume
```

The `--resume` flag skips structures that have already been completed, which is essential for HPC jobs with walltime limits.

## Python API

```python
from amorphgen.pipeline import batch_quench
from amorphgen.utils import get_calculator

calc = get_calculator(model="mace-mpa-0", device="auto")

results = batch_quench.run(
    snapshot_files=["snap_0.xyz", "snap_1.xyz", "snap_2.xyz"],
    n_runs=20,
    select="uniform",
    cfg_override={"model": "mace-mpa-0", "device": "auto"},
    work_dir="batch_run",
    stages=[5, 6, 7],
    calc=calc,
    resume=True,
)
```
