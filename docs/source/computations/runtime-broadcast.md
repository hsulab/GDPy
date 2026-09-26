# broadcast runtime parameters

Use `executor.broadcast` to run independent calculations that differ only in
executor parameters. This NVT example creates workers at three temperatures:

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

Broadcast is shorthand only for executor parameters. Use an explicit flat list
of complete runtimes for alternative potentials, schedulers, or dispatch
policies. Each member of that list may define its own executor broadcast; gdpx
flattens the resolved runtimes in source order.

For an ordered sequence of calculations, see {doc}`runtime-chain`.
