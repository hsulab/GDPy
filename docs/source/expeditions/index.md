(expeditions)=

# Exploration

Exploration proposes and selects candidate structures. Execution evaluates
those candidates through a complete runtime. This keeps Monte Carlo, genetic
algorithms, and other search policies independent of the potential and the
software used to run it.

```yaml
schema_version: 3
potential:
  provider: deepmd
  parameters:
    model: [graph-0.pb, graph-1.pb]
    type_list: [Al, Cu, O]
executor:
  provider: lammps
  method: min
  parameters:
    fmax: 0.05
    steps: 400
    constraint: lowest 120
```

This runtime executes directly on the current machine by default. Add a
{ref}`scheduler configuration <scheduler-transport>` for queue or SSH execution.

The exploration layer owns proposal state, convergence, and selection. The
execution layer owns materialization, job submission, restart, and result
collection.

Global-optimisation inputs use a common top-level architecture. The `recipe`
contains the random seed and settings specific to the selected method; runtime
and optional scheduler configuration remain separate execution concerns.

```yaml
method: genetic_algorithm
recipe:
  random_seed: 7
  population:
    builders:
      random: {}
  # remaining method-specific settings
runtime: {}
```

Omitting the top-level `scheduler` also runs the exploration itself directly
on the current machine.

## List of exploration methods

```{toctree}
:maxdepth: 2

mc.md
ga.md
```
