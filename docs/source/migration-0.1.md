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
- reusable Monte Carlo proposals and acceptance rules: `gdpx.sampling`

## Basin hopping and shared moves

`method: basin_hopping` now selects the former **concurrent hopping** search.
Change `method: concurrent_hopping` to `method: basin_hopping` and keep its
recipe, including `population`, `operators`, `num_mcmoves`, and `mcworker`.
The Python entry point is `gdpx.exploration.basin_hopping.BasinHopping`.

The former `BasinHopping(MonteCarlo)` alias has been deleted. Configurations
using that alias must change to `method: monte_carlo`. Old concurrent-hopping
names and imports are not retained as compatibility aliases.

Moves now live in `gdpx.sampling.moves`; use `parse_operators` from
`gdpx.sampling`. Existing operator configuration keys are retained. Replace
calls to `run()`/`metropolis()` with `propose()` and the operator's separate
`acceptance.accept()` rule. Proposals borrow their input in place: close each
successful proposal with `commit()` or `rollback()`, or use its context manager.

New MC checkpoints store versioned operator configurations instead of pickled
operator instances. Legacy operator checkpoints cannot be resumed; start a
new working directory. New checkpoints and pending-move records support restart.

This extraction preserves the existing acceptance formulas. It does not certify
detailed balance of biased moves or change MC into an execution method.

## Runtime configuration

Every runtime is complete. An omitted `schema_version` selects the current
schema; explicit unsupported versions are rejected. Serialized runtimes and
saved compute plans retain explicit versions.

```
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
