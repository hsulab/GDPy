# migrating to 0.1

gdpx 0.1 is a clean API break. Deprecated forwarding packages, schema-v1
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
- reusable Monte Carlo proposals and acceptance rules: `gdpx.exploration.sampling`

## Basin hopping and shared moves

GA and BH now use `method: global_optimisation` and a `strategy.method` of
`genetic_algorithm` or `basin_hopping`. Remove the `recipe` wrapper: put
population construction under `system`, and put objective, convergence,
archive, operators, and method-specific settings under `strategy`. Keep the
seed at the top level. GA generation
policies move from `population.generation` to `strategy`; `population.name` is
removed because crossover compatibility is automatic. See
{doc}`global_optimisation/population` for the complete migration table.

The former concurrent-hopping search is the `basin_hopping` strategy. Configure
initialization and batched hop calculations with the top-level `runtime`;
`recipe.mcworker` is no longer accepted. MC also uses `system` and `strategy`;
simulated annealing retains its `recipe` wrapper.
For standard and hybrid MC, move the run budget and optional early-stop settings
from `strategy.steps` and `strategy.earlystop` to
`strategy.convergence.steps` and `strategy.convergence.earlystop`.
The Python entry point is `gdpx.exploration.basin_hopping.BasinHopping`.

The former `BasinHopping(MonteCarlo)` alias has been deleted. Configurations
using that alias must change to `method: monte_carlo`. Old concurrent-hopping
names and imports are not retained as compatibility aliases.

Moves now live in `gdpx.exploration.sampling.moves`; use `parse_operators` from
`gdpx.exploration.sampling`. Existing operator configuration keys are retained. Replace
calls to `run()`/`metropolis()` with `propose()` and the operator's separate
`acceptance.accept()` rule. Proposals borrow their input in place: close each
successful proposal with `commit()` or `rollback()`, or use its context manager.

New MC checkpoints store versioned operator configurations instead of pickled
operator instances. Legacy operator checkpoints cannot be resumed; start a
new working directory. New checkpoints and pending-move records support restart.

This extraction preserves the existing acceptance formulas. It does not certify
detailed balance of biased moves or change MC into an execution method.

## Runtime configuration

Every runtime is complete. User-authored runtime files omit `schema_version`
and use the current schema. Serialized runtimes and saved compute plans retain
explicit versions.

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
dispatch:
  batch_size: 1
  worker: batch
  share_workdir: false
  retain_info: false
```

Lists represent independent runtimes. Nested lists passed to
`create_worker_chains` represent ordered chains. There is no implicit
Cartesian product. Use `executor.broadcast` for an explicit Cartesian sweep of
executor parameters; ordinary parameter lists remain literal values.

Schema 3 separated the scheduler from its transport. Replace the schema-2
scheduler providers as follows:

- `provider: local` becomes scheduler `direct` with transport `local`.
- `provider: remote` becomes the actual scheduler (`direct`, `slurm`, `lsf`,
  or `pbs`) with a nested `ssh` transport.

Schema 4 also replaces the runtime `options` mapping with the typed `dispatch`
section shown above. Older runtime files and compute plans are not translated
automatically.

## Workflow and CLI

Workflow variables are `potential`, `executor`, `runtime`, and
`runtime_chain`. The save operation is `save_potential`. Use
`gdp --runtime runtime.yaml compute ...`; the old global potential flag and
schema-1 compute plans are not accepted.

## Global-optimisation objectives

Genetic algorithms and concurrent hopping configure search scoring with the
`strategy.objective` mapping for global optimisation. The former names are not accepted:

- `property` is now `objective`.
- `chempot` is now `chemical_potentials`.

Energy is the default objective and should be omitted. A composition-dependent
objective has the following form:

```yaml
strategy:
  objective:
    target: formation_energy
    chemical_potentials:
      Cu: -3.50
      O: -4.95
```

## Global-optimisation database

Population-based global-optimisation methods now store candidates and restart
metadata in `candidates.db` inside each exploration directory. Remove the former
GA `database` and concurrent-hopping `population.database_fname` settings;
custom database filenames are no longer accepted.

## Driver worker output layout

New driver-based runs keep two JSON catalogs under `_meta/`:

```text
_meta/
  inputs.json
  scheduler.json
  jobscripts/
    run-<uuid>.script
```

`inputs.json` contains lossless structures indexed by fingerprint, provenance,
retained information, frozen calculation sets, immutable job manifests, and the
optional compute plan.
`scheduler.json` contains scheduler providers, job lifecycle records, and cached
shared-workdir results including calculator properties. Workers in one compute
plan share these catalogs. A hidden `.metadata.lock` coordinates atomic updates
between processes.

Preparation renders each job script once; submission and resubmission reuse its
UUID and path. Generated `gdp compute run --job <uuid>` commands resolve their
input from the catalog. Remote results merge only into the corresponding job;
remote scheduler records cannot overwrite controller state.

Each calculation folder now represents one fixed set of structures and settings.
Preparation freezes **all** batches, even when only one batch is submitted.
Repeating the same request resumes it; remaining planned batches and failed jobs
can still be submitted or retried. Adding, removing, or reordering structures,
changing settings, seeds, or task mappings, or appending workers requires a new
working directory. A conflict is checked before writing metadata or scripts.
There is no append option or automatic migration.

Calculation directories (`cand*`), collected `results/`, and trajectory archives
remain in each calculation folder. MC uses `calculations/step.0000/` for its
initial calculation and `calculations/step.NNNN/` for subsequent steps. HMC uses
`calculations/step.NNNN/procedure.NNNN/` directly for MD and adds
`proposal.NNNN/` beneath that directory for MC. Pending calculations resume in
their original folders;
checkpoint rollback removes entire calculation folders beyond the saved step.

Input catalogs now use version 2 and embedded compute plans use schema version 6.
Older driver layouts, including version-1 input catalogs, separate SHA-256
snapshots/manifests, schema-4/5 plans, `_data/`, and root-level job databases,
require fresh folders. Manifest-file `--job` arguments and old append-based
MC/HMC layouts are rejected. Existing outputs are not modified or migrated.
Corrupt catalogs raise errors instead of falling back to legacy files.

## Driver input fingerprints

Driver workers and compute plans use the same versioned SHA-256 structure
fingerprint. It covers frame and atom order, atomic arrays (including custom
arrays), cell, PBC, and constraints. Standard arrays include tags, masses,
momenta, initial charges, and initial magnetic moments. Missing standard arrays
are equivalent to their ASE defaults. Numeric arrays use fixed types and byte
order; signed zero is normalized. Positions are not rounded, wrapped, sorted,
or aligned. Non-finite numbers and unsupported values are rejected.

Bookkeeping in `Atoms.info`, attached calculators, and calculated results do not
affect input identity. Lossless snapshots embedded in `inputs.json` preserve calculation
inputs, including constraints and full coordinate precision. Snapshots are
checked against their saved fingerprints before reuse.

Each job manifest in `inputs.json` records a separate job fingerprint covering the
structure fingerprint, runtime configuration, batch mapping, workdir names,
and exact random seeds or generator states. Resubmission and staged execution
read that saved batch. Restarting without an explicit seed reuses the saved
random choices. The entire calculation set is checked, including unsubmitted
batches and workdir names. Use a new run directory for changed inputs or settings.

Runtime configuration remains schema version 3. Older driver metadata lacks the
frozen calculation-set contract and is rejected even when it uses SHA-256.
Old MD5 records also cannot be migrated reliably because their XYZ snapshots may have
lost constraints or coordinate precision.
