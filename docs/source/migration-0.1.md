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
schema_version: 3
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
  provider: direct
  method: default
  parameters: {}
  transport:
    provider: local
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

Schema 3 separates dispatch from transport. Replace the schema-2 scheduler
providers as follows:

- `provider: local` becomes scheduler `direct` with transport `local`.
- `provider: remote` becomes the actual scheduler (`direct`, `slurm`, `lsf`,
  or `pbs`) with a nested `ssh` transport.

Schema-2 runtime files and compute plans are not translated automatically.

## Workflow and CLI

Workflow variables are `potential`, `executor`, `runtime`, and
`runtime_chain`. The save operation is `save_potential`. Use
`gdp --runtime runtime.yaml compute ...`; the old global potential flag and
schema-1 compute plans are not accepted.

## Global-optimisation objectives

Genetic algorithms and concurrent hopping configure search scoring with the
recipe-level `objective` mapping. The former names are not accepted:

- `property` is now `objective`.
- `chempot` is now `chemical_potentials`.

Energy is the default objective and should be omitted. A composition-dependent
objective has the following form:

```yaml
objective:
  target: formation_energy
  chemical_potentials:
    Cu: -3.50
    O: -4.95
```

## Global-optimisation database

Population-based global-optimisation methods now store candidates and restart
metadata in `candidates.db` inside each expedition directory. Remove the former
GA `database` and concurrent-hopping `population.database_fname` settings;
custom database filenames are no longer accepted.
