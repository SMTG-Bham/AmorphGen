# Batch quench

Quench multiple snapshot structures through the final stages of the pipeline in a single run.

## Use case

After a high-temperature equilibration, you may want to extract multiple snapshots and quench each separately. For trajectory input, `--select decorrelated` chooses burn-in and spacing from scalar autocorrelation and per-species diffusion. Uniform sampling remains the batch default and alone does not establish independence.

## CLI usage

```bash
amorphgen --batch-quench \
    --snapshot-dir snapshots/ \
    --model mace-mpa-0 \
    --device cuda \
    --batch-stages 5 6 7 --n-runs 20 \
    --resume -o batch_run/
```

The `--resume` flag skips runs containing `final_amorphous.xyz` (or the legacy `.extxyz` equivalent) and resumes interrupted MD from saved trajectory frames. Reuse the same inputs, configuration and selection when resuming.

`--n-runs` defaults to 20 and selects at most that many inputs. Use `--select uniform` (default) or `--select last`. A directory is searched in order for `*.xyz`, `*.extxyz`, `*.vasp`, `*.cif`, then `POSCAR*`; only the first matching format is used. Keep inputs in one format to avoid accidentally excluding files.

`--snapshot-dir` also accepts a trajectory file. Frames are extracted to
`batch_run/snapshots_extracted/`. Use `--select decorrelated` for adaptive
burn-in and spacing, matching the `--mq-ensemble` default:

```bash
amorphgen --batch-quench --snapshot-dir shared/stage4_eq_traj.xyz \
    --config mq.yaml --n-runs 20 --select decorrelated \
    --batch-stages 5 6 7 --resume -o batch_run/
```

Decorrelated selection can return fewer than the requested number of frames;
if decorrelation cannot be resolved, it returns only the final frame and
reports the limitation. `snapshot_sampling.json` and `.txt` record burn-in,
spacing, selected frame indices and the estimated effective independent
snapshot count. The estimate is a diagnostic, not proof of equilibration or
independence of the resulting glasses. See {doc}`mq-ensemble` for the method.

Set `--burn-in-frames N` to override adaptive burn-in, or to discard leading
frames with uniform/last selection. An explicit zero is honoured.
`--decorrelation-distance` overrides the displacement proxy's distance in Å.
Adaptive selection requires an ordered trajectory; use uniform/last selection
for a directory of preselected structures. To anneal already-disordered inputs
before quenching, use `--batch-stages 4 5 6 7`.

## Python API

```python
from amorphgen.pipeline import batch_quench
from amorphgen.utils import get_calculator

calc = get_calculator(model="mace-mpa-0", device="auto")

results = batch_quench.run(
    snapshot_files=["snap_0.xyz", "snap_1.xyz", "snap_2.xyz"],
    n_runs=3,
    select="uniform",
    cfg_override={"model": "mace-mpa-0", "device": "auto"},
    work_dir="batch_run",
    stages=[5, 6, 7],
    calc=calc,
    resume=True,
)
```

`n_runs` selects at most that many available snapshots; it does not create repeated
runs from one snapshot. With the ASE engine, one selected input writes directly
to `work_dir`; multiple inputs get separate `run_NNNN/` directories. Use a separate
work directory for each single-input job in a SLURM array.
