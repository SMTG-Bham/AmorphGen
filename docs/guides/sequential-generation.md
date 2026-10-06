# Generate until converged

Use `--until-converged` to generate independent random structures, relax them
in torch-sim batches, and stop when every declared descriptor mean reaches
its requested precision. The stopping rule controls error across repeated
checks and all declared targets.

The existing `--analyse --convergence` report estimates uncertainty and future
sample requirements from an existing ensemble. Those Student-t planning
curves do not provide an optional-stopping guarantee. Use this generation
mode for adaptive stopping; see {doc}`analysis` for retrospective analysis.

## Run a complete example

Install the optional engine following the
[torch-sim installation instructions](../getting-started/installation.md#the-torch-sim-engine).
From a repository checkout, run:

```bash
amorphgen --random-gen --config examples/until_converged.yaml -o sequential_demo
```

The supplied configuration uses a small CPU Lennard-Jones system to exercise
the workflow. Its loose tolerances and potential are demonstration settings,
not a calibrated material model.

```{literalinclude} ../../examples/until_converged.yaml
:language: yaml
```

CLI declarations override YAML values per descriptor. For example,
`--tolerance density=0.5 --descriptor-bounds density=0,4` changes that target
while retaining other targets from the configuration. Bounds must remain
justified for the complete generating procedure.

## Declare the sampling contract

Set the model, composition, generation and relaxation settings, base seed,
confidence, descriptor definitions, support bounds and tolerances **before
sampling**. The required mode is `--random-gen --relax --engine torchsim`;
YAML can enable relaxation and `random_gen.until_converged` as above.

Every tolerance requires matching `analysis.descriptor_bounds` or a repeated
`--descriptor-bounds NAME=LOW,HIGH` argument. These bounds describe the true
population support of a structure's descriptor under the fixed procedure.
They are not bounds on the population mean, observed extrema, or values
estimated from a pilot sample. Observations outside them stop the run; passing
this runtime check alone does not establish that the support assumption is
true. Wider valid bounds generally require more structures.

The native generation controller supports these scalar descriptors:

| Name | One observation per relaxed structure | Units |
|---|---|---|
| `density` | Mass divided by volume | g/cm³ |
| `energy.per_atom` | Potential energy divided by atom count | eV/atom |
| `energy.total` | Total potential energy | eV |
| `coordination.X-Y` | Mean number of Y neighbors of an X atom | neighbors |
| `total_coordination.X` | Mean number of all neighbors of an X atom | neighbors |
| `bond_distance.X-Y` | Mean distance of the selected bonds | Å |
| `bond_angle.X-Y-Z` | Mean selected angle, with Y at the center | degrees |

Structural targets require a fixed positive **numeric** `--cutoff` or
`analysis.cutoff`, in Å. Automatic RDF cutoffs and pair-specific cutoff maps
are not accepted by this controller. All named elements must occur in the
composition. Choose targets defined for every possible generated structure:
an absent bond or angle descriptor fails the batch rather than becoming zero
or disappearing from the target family. RDF and angle-distribution curves
are currently available for analysis, not as native adaptive-generation targets.

Each structure has equal weight. Atoms, bonds and correlated trajectory frames
do not count as independent replicates. Independent index seeds are derived
from an explicit nonnegative base seed. Generation and relaxation settings
are frozen; the controller does not screen out inconvenient structures,
switch protocols after a failure, or replace failures with fresh seeds.

## Stopping rule

Let $c$ be the confidence, $K$ the number of declared scalar targets, and $n$
the cumulative number of independent structures. For each target and every
$n \geq 2$, the error allocation is

$$\delta_n = \frac{1-c}{K n(n-1)}.$$

For a target bounded by $[L,U]$, with sample variance $s_n^2$ using `ddof=1`,
the empirical Bernstein half-width is

$$h_n = \sqrt{\frac{2s_n^2\log(4/\delta_n)}{n}}
       + \frac{7(U-L)\log(4/\delta_n)}{3(n-1)}.$$

This uses the two-sided form of
[Maurer and Pontil (2009), Theorem 11](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf).
The implementation also enlarges the radius for reported-mean rounding.
Because $\sum_{n=2}^{\infty}1/[n(n-1)]=1$, a union bound over sample sizes and
targets limits total error to $1-c$. Under the stated independent, identically
distributed and bounded-observation assumptions, all declared population means
are covered simultaneously at all looks, including the selected stopping time.

After each complete batch, the controller stops only when the minimum sample
count is reached and **every** half-width is at most its tolerance. Zero observed
variance still has a positive finite-sample uncertainty term. This establishes
precision for the fixed generation-and-relaxation distribution; it does not
establish physical equilibration, model accuracy or negligible finite-cell error.

## Batches, limits and outcomes

| CLI setting | YAML setting | Default |
|---|---|---|
| `--convergence-batch-size` | `random_gen.convergence_batch_size` | 8 |
| `--convergence-min-structures` | `random_gen.convergence_min_structures` | 2 |
| `--convergence-max-structures` | `analysis.convergence_max_structures` | 1000 |
| `--convergence-confidence` | `analysis.convergence_confidence` | 0.95 |
| `--batch-size` | `opt.batch_size` | `auto` |

The convergence batch size determines when evidence is checked. `--batch-size`
controls hardware chunks within that batch; `auto` uses the convergence batch
size in this mode. Smaller hardware chunks do not create additional statistical
looks. A final batch may be shorter to respect the cap. Use
`--convergence-max-structures` for the total cap; `-n` and `--indices` are not
accepted. Output uses `.xyz` files carrying extended-XYZ metadata and energies.

| Result status | CLI exit code | Meaning |
|---|---|---|
| `converged` | 0 | All precision targets and the minimum count were met |
| `max_structures_reached` | 2 | The cap was reached without convergence |
| Failure | 1 | Invalid configuration, generation, relaxation or evidence |

Relaxation must meet the configured force tolerance and, when applicable,
pressure tolerance. A failed or incomplete batch contributes no observations;
the previously committed sample remains intact. The cap is a resource limit
and is never reported as statistical convergence.

## Checkpoints and resume

The output contains `random_initial/random_NNNN.xyz`,
`random_opt/random_NNNN_opt.xyz`, and `adaptive_convergence.json`. The atomic
checkpoint retains the full observation prefix, seeds, file hashes, batch
history, frozen contract and confidence-sequence report. Its
`convergence.inference_contract.sequentially_valid` is `true`, conditional on
the recorded assumptions.

Resume the same output directory explicitly:

```bash
amorphgen --random-gen --config examples/until_converged.yaml \
    -o sequential_demo --resume
```

Resume verifies the original settings, model, software identity and every
committed initial/final artifact, then recomputes observations and uncertainty.
It retains cumulative sample counts and error spending. An interrupted batch
replays the same structure indices and seeds; it is never skipped. An already
converged or capped run returns its terminal result without drawing more samples.

Changing targets, bounds, tolerances, seed, model, software, batch sizes or the
cap invalidates resume. A new protocol requires a fresh output directory; do
not combine incompatible runs or reset the error budget while retaining their
evidence. Concurrent controllers are blocked by the output lock.
