(operations)=

# resources and steps

Resources hold configuration or lightweight values. Built-in resource types
include `potential`, `executor`, `runtime`, `runtime_chain`, `scheduler`,
`builder`, `selector`, `trainer`, and `validator`.

Steps consume resources or earlier step results. Common types include `build`,
`read_stru`, `write_stru`, `compute`, `extract`, `select`, `react`, `train`, and
`validate`. Each step stores its state beneath the workflow run directory so a
run can resume.

Dependencies are ordinary names under `inputs`; there is no interpolation
syntax:

```yaml
steps:
  relax:
    __type__: compute
    inputs:
      structures: structures
      runtime: local_relaxation
    options:
      extract_data: true
```

The input key is the constructor argument and the value is the resource or step
name. Put literals under `options`. This separation makes the dependency graph
available to `gdp workflow plan` and `gdp workflow graph` without constructing
or running nodes.

Steps that publish more than one result expose named outputs. Select one with a
structured reference:

```yaml
inputs:
  dataset: {node: transfer, output: dataset}
```

Every multi-output node declares one default head. For an exploration, use:

```yaml
inputs:
  structures: sample
```

This is equivalent to `structures: {node: sample}` or the explicit
`structures: {node: sample, output: minimised_candidates}`. Omitting the head
chooses the declared default, not the entire output mapping or its first key. Single-output nodes retain their ordinary value. The same rules apply to
step inputs, resource inputs, and repeated-workflow state updates. Resource
initial state uses the resource's default head. `transfer` defaults to `dataset`.

Named output references and default declarations are checked during workflow
validation. Missing or unexpected declared outputs fail during execution before
consumers run. Referencing several heads does not execute the producer more
than once.

Node implementations declare `output_names` and `default_output` and return a
`NamedOutputs` mapping containing every declared head. Configuration-dependent
heads can be declared with `output_names_for(spec)` and the corresponding
instance `output_names`, as used by `transfer`. A normal dictionary returned by
a single-output node remains one value; it is not implicitly split into heads.

## Uncertainty per compute step

A `compute` step can override the potential's `estimate_uncertainty` flag without
changing the shared runtime or loop-carried potential. Omit the option (or use
`null`) to inherit the potential setting. The resolved override is stored in the
worker runtime configuration, including for broadcast variants and resumed jobs.

For DeepMD active learning, use the first model for NVT and evaluate the ensemble
on saved frames afterwards:

```yaml
steps:
  run_nvt:
    __type__: compute
    inputs:
      structures: read_stru
      runtime: exploration_runtime
    options:
      estimate_uncertainty: false
  est_devi:
    __type__: compute
    inputs:
      structures: run_nvt
      runtime: uncertainty_runtime
    options:
      estimate_uncertainty: true
```

Both runtimes can reference the same `current_potential` ensemble. The overrides
apply to each iteration's potential, including newly trained models.


### Separate exploration trajectories

The `explore` operation exposes four heads:

- `minimised_candidates` (default): final minimised structures. For SA, this
  includes every successful minimum in `candidates.db`, including generation 0,
  duplicates, and candidates not retained by the population comparator.
- `dynamics`: MD workers, for extracting saved cooling trajectories.
- `minimisation`: minimisation workers, for extracting saved relaxation trajectories.
- `continuation`: cross-iteration exploration state; currently `None` for SA.

The default structures can feed `select` or `compute` directly, without an
`extract` step. `results`, `md_workers`, and `minimisation_workers` are replaced
by these heads; choose an explicit trajectory head when migrating extraction
steps. For population SA, `minimisation` includes generation 0 and
minimisations of selected MD snapshots. Extract the sources independently when they need separate selection
budgets:

```yaml
md_trajectories:
  __type__: extract
  inputs:
    compute: {node: sample, output: dynamics}
  options:
    merge_workers: true
    use_archive: false

min_trajectories:
  __type__: extract
  inputs:
    compute: {node: sample, output: minimisation}
  options:
    merge_workers: true
    use_archive: false
```

Here `sample` is an `explore` step. Each extraction contains all saved frames
from its source, including intermediate minimisation frames. The SA example in
`examples/workflows/active-learning-dpa4c/sa.yaml` scores, selects, and labels
the two sources separately before adding both labeled contributions to the
training dataset.
