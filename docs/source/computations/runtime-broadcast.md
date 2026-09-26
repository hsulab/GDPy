# broadcast runtime parameters

Use a component-local `broadcast` to run independent calculations that differ
only in selected parameters. This NVT example creates workers at three
temperatures:

```yaml
potential:
  provider: emt
executor:
  provider: ase
  method: md
  parameters:
    ensemble: nvt
    timestep: 1.0
    steps: 1000
  broadcast:
    temp: [300, 600, 900]
```

With `gdp compute`, the resolved runtimes use `w0`, `w1`, and `w2` in the same
order as the values. In a workflow, a `compute` step accepts the resulting
runtime collection directly.

## Multiple parameters

Broadcast keys are paths relative to `executor.parameters`. Multiple paths
form a Cartesian product in declaration order, with the rightmost path varying
fastest:

```yaml
executor:
  provider: ase
  method: md
  parameters:
    ensemble: nvt
    controller:
      name: berendsen
      params: {}
  broadcast:
    temp: [300, 600]
    controller.params.Tdamp: [50.0, 100.0]
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
