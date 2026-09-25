(sessions)=

# declarative workflows

A workflow is a YAML description of named resources, executable steps, and the
data dependencies between them. The current format is always used, so workflow
files do not declare a `schema_version`.

Use the `workflow` command group to check a file before running it:

```console
gdp workflow validate workflow.yaml
gdp workflow plan workflow.yaml
gdp workflow graph workflow.yaml
gdp workflow run workflow.yaml
```

## Complete example

This workflow reads structures, relaxes them with EMT, and extracts the
trajectories:

```yaml
parameters:
  structures: ./structures.xyz

workflow:
  mode: once
  targets: trajectories

resources:
  emt:
    __type__: potential
    options:
      provider: emt

  minimizer:
    __type__: executor
    options:
      provider: ase
      method: min
      parameters:
        fmax: 0.05
        steps: 100

  relaxation:
    __type__: runtime
    inputs:
      potential: emt
      executor: minimizer
    options:
      dispatch:
        worker: batch
        batch_size: 8

steps:
  structures:
    __type__: read_stru
    options:
      fname: {$param: structures}

  relax:
    __type__: compute
    inputs:
      structures: structures
      runtime: relaxation

  trajectories:
    __type__: extract
    inputs:
      compute: relax
```

`resources` construct reusable values and provider components. `steps` perform
work. Every node selects a registered class with `__type__`; literal constructor
arguments belong under `options`, while `inputs` maps constructor arguments to other node names.
Input values may also be lists of node names. Resource nodes can depend only on
other resources, and workflow targets must be steps.

The run is stored under `<directory>/<workflow-file-stem>`. Select another root
with the global option `gdp -d PATH workflow run workflow.yaml`.

## Parameters and profiles

Use an exact `{$param: name}` value to insert a declared parameter while
preserving its type. Override declared parameters from the command line with
YAML values:

```console
gdp workflow run workflow.yaml --set structures=other.xyz
gdp workflow run workflow.yaml --profile production --set batch.size=32
```

Nested parameter names use dots. Overrides cannot introduce undeclared
parameters. A profile can override parameters and existing node `options`, but
cannot change types or dependency wiring:

```yaml
parameters:
  batch:
    size: 8

profiles:
  production:
    parameters:
      batch:
        size: 32
    resources:
      minimizer:
        options:
          parameters:
            steps: 500
```

## Includes and external values

Split a workflow into reusable YAML fragments with `includes`. Paths are
resolved relative to the file that declares them, and duplicate names are an
error:

```yaml
includes:
  - resources.yaml
  - common-steps.yaml
```

An exact `{$file: path.yaml}` value loads YAML or JSON from a separate file.
Its path is likewise relative to the declaring file. Includes combine workflow
sections; `$file` inserts the loaded value at its location.

## Execution modes

`mode: once` is the default and executes the dependency graph one time.
`mode: repeat` executes the same graph repeatedly, up to `max_iterations`.
This complete minimal example runs the graph for three iterations:

```yaml
parameters:
  input: ./structures.xyz

workflow:
  mode: repeat
  targets: save
  max_iterations: 3
  reset_random_state: false

steps:
  load:
    __type__: read_stru
    options:
      fname: {$param: input}

  save:
    __type__: write_stru
    inputs:
      structures: load
    options:
      dump_last: false
```

Run the checked-in example with:

```console
gdp workflow validate examples/workflows/repeat.yaml
gdp workflow run examples/workflows/repeat.yaml --set input=other.xyz
```

Each iteration gets its own `iter.0000`, `iter.0001`, and `iter.0002`
directory. When a step implements `report_convergence()`, the repeated workflow
stops as soon as all reporting steps converge; otherwise it runs through
`max_iterations`.

Workflows containing stochastic builders can also rewind their random state:

```yaml
workflow:
  mode: repeat
  targets: explore
  max_iterations: 10
  reset_random_state: true
  reset_random_config: [init, 1]
```

`reset_random_config` contains the reset mode (`init` or `zero`) and the first
iteration index at which resetting applies. In production active-learning
workflows, the target is typically the final validation or training step, and
all exploration, selection, labeling, and training dependencies are reached
through its `inputs` graph.

## Migrating older session files

The previous session format is not loaded implicitly. Rename `variables` to
`resources` and `operations` to `steps`, rename each node's `type` to
`__type__`, move constructor values into `options`, and replace interpolated
node references with named `inputs`. Replace the `sessions` entry-point mapping
with `workflow.targets`. Placeholders become declared `parameters` referenced
by `$param` and overridden with `--set`.

The old `gdp session` command has been replaced by `gdp workflow run`.

```{toctree}
:maxdepth: 1

operations
```
