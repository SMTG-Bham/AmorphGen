# Pipeline

The pipeline module implements the 7-stage melt-and-quench molecular dynamics workflow.

## MeltQuenchPipeline

```{eval-rst}
.. autoclass:: amorphgen.pipeline.run_pipeline.MeltQuenchPipeline
   :members:
   :undoc-members:
   :show-inheritance:
```

## Stage modules

### Stage 1: Structure optimisation

```{eval-rst}
.. automodule:: amorphgen.pipeline.opt_cell
   :members:
```

### Stage 2, 4, 6: Equilibration

```{eval-rst}
.. automodule:: amorphgen.pipeline.equilibrate
   :members:
```

### Stage 3: Melt (heat ramp)

```{eval-rst}
.. automodule:: amorphgen.pipeline.melt_cell
   :members:
```

### Stage 5: Quench (cooling ramp)

```{eval-rst}
.. automodule:: amorphgen.pipeline.quench
   :members:
```

### Stage 7: Final optimisation

Delegates to `opt_cell.run` with `stage_key="final_opt"`. Settings inherit
from `opt`, with individual values overridden by `final_opt`.

```{eval-rst}
.. automodule:: amorphgen.pipeline.final_opt
   :members:
```

### Batch quench

```{eval-rst}
.. automodule:: amorphgen.pipeline.batch_quench
   :members:
```
