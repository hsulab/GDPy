# broadcast runtime parameters

Use a component-local `broadcast` to run independent calculations that differ
only in selected parameters. This NVT example creates simulations at three
temperatures:

```yaml
potential:
  provider: emt
executor:
  provider: ase
  method: md
  parameters:
    setup:
      ensemble: nvt
      timestep: 1.0
      regulator:
        name: berendsen
        targets:
          temperature: 300
        parameters: {}
    stop:
      steps: 1000
  broadcast:
    setup.regulator.targets.temperature: [300, 600, 900]
```

With `worker: batch`, compatible runtimes are batched automatically. For one
input structure, these temperatures use `cand0`, `cand1`, and `cand2` under the
run directory. In a workflow, a `compute` step accepts the resulting runtime
collection directly.

### Extract and select a temperature

Workflow `compute` and `extract` restore results as
`[variant, input structure, trajectory frame]`, even when several variants
share one physical worker. For a temperature-only sweep with distinct targets,
the first axis is named `temperature` and carries the target values. With
`[400, 500, 600, 700]`, index `1` selects all 500 K trajectories:

```python
result[1]                         # all structures and frames at 500 K
result.sel(temperature=500)        # selection by target temperature
```

The existing locate selector can select the same axis in a workflow:

```yaml
selection:
  method: locate
  group_by: 0
  indices: '1'
```

This result order does not depend on `batch_size` or `concurrent_tasks`.
Cached trajectories are regrouped during extraction as well. One variant
still reduces to `[structure, frame]` by default; `merge_workers: true`
explicitly combines variants into the candidate axis. Sweeps with repeated
temperature targets use a generic `variant` axis.

## Batch executor variants in one allocation

`worker: batch` puts compatible executor variants in one driver worker.
`batch_size` counts simulations across all variants and input structures;
`concurrent_tasks` controls how many simulations execute at once inside each
job. For four temperatures, use:

```yaml
dispatch:
  worker: batch
  batch_size: 4
scheduler:
  provider: slurm
  parameters:
    nodes: 1
    ntasks: 4
    cpus-per-task: 32
    gpus-per-task: 1
    mem: 220G
    concurrent_tasks: 4
    machine_prefix: >-
      srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=32
      --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores --mem=55G
```

Together with an executor broadcast over `[400, 500, 600, 700]`, one input
structure creates one job containing four independent simulations. Each
simulation runs in its own `cand0` through `cand3` directory and gets one GPU.
With additional structures, all variants of each structure stay adjacent in
the task order, and tasks are split into batches of up to four.

Compatible variants share the potential, modifiers, scheduler, dispatch,
and executor provider and method. Executor parameters may differ. They require
`worker: batch` and `share_workdir: false`. Either specify `random_seed` in every
variant or omit it in every variant to batch them together; realized per-task
seeds are persisted for restart. An explicit list of runtimes follows the same
batching rules. Incompatible variants are assigned to separate workers.

GDPy chooses the folder layout automatically: one worker uses `cand0`,
`cand1`, etc. directly under the run directory; multiple workers use
`w0/cand0`, `w1/cand0`, etc. Job batches do not add another directory level.
The frozen plan records variant indices, structure indices, and candidate
paths. Existing plans retain their original layout on restart even when
automatic batching policy changes. Use a new directory when changing inputs.

## Multiple parameters

Broadcast keys are paths relative to `executor.parameters`. Multiple paths
form a Cartesian product in declaration order, with the rightmost path varying
fastest:

```yaml
executor:
  provider: ase
  method: md
  parameters:
    setup:
      ensemble: nvt
      regulator:
        name: berendsen
        targets:
          temperature: 300
        parameters:
          Tdamp: 50.0
  broadcast:
    setup.regulator.targets.temperature: [300, 600]
    setup.regulator.parameters.Tdamp: [50.0, 100.0]
```

The combinations are `(300, 50)`, `(300, 100)`, `(600, 50)`, and
`(600, 100)`. A broadcast may supply a missing final parameter, but every
parent mapping or list index must already exist. Use nested lists when one
alternative is itself a list.

Ordinary lists remain literal executor parameters. Only fields named under
`broadcast` are expanded. Empty, malformed, unknown, or overlapping paths are
rejected before workers are created.

## Modifier windows

Place `broadcast` beside a modifier's `parameters` to create independent
restraint windows. Its keys are paths relative to that modifier's parameters:

```yaml
modifiers:
  - provider: builtin
    method: distance_harmonic
    parameters:
      group: "`index 8 10`"
      kspring: 5.0
    broadcast:
      center: [0.95, 1.15, 1.35, 1.55]
```

This produces four workers, `w0` through `w3`, with one restraint center per
worker. They are independent replicas/windows: broadcast does not exchange
configurations between workers or reconstruct a free-energy profile.
For equilibrated windows, persisted per-replica seeds, and production metadata,
use the {doc}`umbrella-sampling exploration <../explorations/umbrella-sampling>`.

Executor and modifier broadcasts can be combined. They form one Cartesian
product, with executor dimensions first and modifier dimensions following in
modifier-list order. Within each component, YAML declaration order is
preserved and the rightmost dimension varies fastest.

Broadcast is supported for executor and modifier parameters. Use an explicit
flat list of complete runtimes for alternative potentials, schedulers, or
dispatch policies. Each member of that list may define its own broadcasts;
gdpx flattens the resolved runtimes in source order.

For an ordered sequence of calculations, see {doc}`runtime-chain`.
