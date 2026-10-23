# Reproducible Ga2O3 FEFF example

This example uses two relaxed 400-atom Ga2O3 structures already distributed
with Tutorial 8. It requires an installed AmorphGen checkout containing this
feature and a separate FEFF8.5L executable. No model or structure is downloaded.
The two calculations are an execution example, not a converged prediction.

From the repository root:

```bash
python examples/exafs/run_example.py all --output ga2o3_exafs_example \
  --executable /path/to/feff
```

Each 6 Å cluster contains one sampled Ga absorber; sampling uses seed 0.
The script retains the input structures, source hashes, FEFF input/output,
completion fingerprints, averaged chi(k), its Fourier transform and a plot.
The transform uses k² weighting, a Hanning window from 3 to 12 Å⁻¹ and
dk=1 Å⁻¹. S₀²=1 is an amplitude convention, not an experimental calibration.
The structure-level SEM from two configurations is illustrative only.

For cluster execution, split the same workflow into stages:

```bash
python examples/exafs/run_example.py prepare --output ga2o3_exafs_example
# Run this step on an allocated CPU node with FEFF available:
python examples/exafs/run_example.py run --output ga2o3_exafs_example --executable /path/to/feff
python examples/exafs/run_example.py collect --output ga2o3_exafs_example
```

The existing job tree can be rerun with unchanged inputs. Use a new output
directory after changing the structure, radius or FEFF cards.

`ga2o3_validation.py` provides the larger reproducibility/convergence study.
It requires an explicit ensemble directory and experimental curve; it does not
substitute the two tutorial cells for the original twenty-cell ensemble.
