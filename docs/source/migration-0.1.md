# Migrating to 0.1

GDPy 0.1 is a clean API break. Deprecated forwarding packages, schema-v1
translation, implicit runtime broadcasting, and schema-1 compute plans have
been removed.

## Imports

Use the package that owns the concept:

- potentials, materializers, executors, and trainers: `gdpx.providers`
- runtime resolution, workers, and schedulers: `gdpx.execution`
- adaptive search algorithms: `gdpx.exploration`
- graph variables and operations: `gdpx.workflow`
- builders and geometry: `gdpx.structures`
- selectors and validators: `gdpx.analysis`
- biases and collective variables: `gdpx.modifiers`
- loaders and arrays: `gdpx.data`

## Runtime configuration

Every runtime is complete and explicitly versioned:

```
schema_version: 2
potential:
  provider: deepmd
  method: default
  parameters:
    model: graph.pb
executor:
  provider: lammps
  method: md
  parameters:
    steps: 10000
modifiers: []
scheduler:
  provider: local
  method: default
  parameters: {}
options:
  batch_size: 1
  worker: batch
  share_workdir: false
  retain_info: false
```

Lists represent independent runtimes. Nested lists passed to
`create_worker_chains` represent ordered chains. There is no implicit
Cartesian product.

## Workflow and CLI

Workflow variables are `potential`, `executor`, `runtime`, and
`runtime_chain`. The save operation is `save_potential`. Use
`gdp --runtime runtime.yaml compute ...`; the old global potential flag and
schema-1 compute plans are not accepted.
