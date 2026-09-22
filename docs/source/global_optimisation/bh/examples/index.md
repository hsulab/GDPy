(bh-examples)=

# examples

These examples demonstrate complete basin-hopping searches, including structure
generation, local relaxation, hopping moves, and inspection of the results.
They use small populations and short move budgets to keep the workflows easy
to inspect. Increase the search and relaxation budgets for production work.

Exploration inputs live under
`examples/global_optimisation/explorations/basin_hopping/`. Shared calculation
inputs live under `examples/global_optimisation/runtimes/`. Each guide shows
both files. Select a runtime with
`gdp --runtime <runtime.yaml> explore <exploration.yaml>`; use a new output
directory when changing the runtime or search settings.

All examples use direct scheduling by default. See {ref}`scheduler-transport`
for queue and SSH execution, and {ref}`exploration-output-layout` for output
directories and restart metadata. EMT examples require no model download;
TACE and MatterSim require their optional model dependencies.

The examples below cover finite **Cluster** systems. Select a chemical system
to open its configuration and guide.

| Main system | Specific system | Potential | Notes |
| --- | --- | --- | --- |
| Cluster | {ref}`Cu₈ <bh-cu8-example>` | EMT | Basic search and trajectory export |
| Cluster | {ref}`Cu₆Ni₂ and Cu₄Ni₄ <bh-composition-broadcast-example>` | EMT | Broadcast two compositions in one allocation |
| Cluster | {ref}`Cu₆Niₓ <bh-variable-composition-example>` | EMT | Variable composition through Ni insertion/removal |
| Cluster | {ref}`Cu₄O₄ <bh-cuox-tace-example>` | TACE | Oxide cluster search |
| Cluster | {ref}`Cu₄O₄ <bh-cuox-thanos-example>` | MatterSim | Extinction and chain restarts |

```{toctree}
:maxdepth: 1
:hidden:

cluster
compositions
variable-composition
cuox_tace
cuox_thanos
```
