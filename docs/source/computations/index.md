(computations)=

# gdp compute

Run the same calculation on a collection of structures with `gdp compute`.
Choose a potential, select a task, then decide where to execute it.

```shell
gdp -d results -r runtime.yaml compute structures.xyz
```

| Start here | What it covers |
| --- | --- |
| {doc}`compute` | A copper-dimer demo, preparation, submission, status, and collected results. |
| {doc}`tasks/index` | Single points, relaxation, molecular dynamics, transition states, and vibrations. |
| {doc}`runtime` | Potential/executor compatibility and optional modifiers. |
| {doc}`schedulers` | Local CPUs/GPUs, batch queues, SSH, and resource allocation. |

## Runtime layout

A schema-v3 runtime has a `potential`, an `executor`, optional `modifiers`,
an optional `scheduler`, and optional worker `options`. In a runtime YAML file,
these are top-level keys; do not wrap them in another `runtime` key.

```yaml
potential:
  provider: emt
  parameters: {}
executor:
  provider: ase
  method: min
  parameters:
    fmax: 0.05
    steps: 100
```

The potential supplies energies and forces. The executor controls the
calculation. With no scheduler configured, gdpx runs directly on the current
machine. A queue scheduler and transport change where and how it runs without
changing the task settings.

Potential-specific model settings belong in the {doc}`potential guides
<../potentials/providers>`. For training models, see {doc}`../trainers/index`.
