# runtime dispatch

The optional `dispatch` section controls how structures are assigned to
workers and batches. It is independent of the {doc}`scheduler <schedulers>`, which controls
where those workers run.

Defaults are `worker: batch`, `batch_size: 1`, `share_workdir: false`, and
`retain_info: false`. A simulation is one runtime applied to one input
structure. Compatible batch runtimes share a worker automatically, so
`batch_size` counts simulations across both structures and runtime variants.
This example groups up to 16 simulations in each batch and
preserves input metadata on collected structures:

```yaml
potential:
  provider: emt
executor:
  provider: ase
  method: min
  parameters:
    stop:
      fmax: 0.05
      steps: 300
dispatch:
  worker: batch
  batch_size: 16
  retain_info: true
```

Set `share_workdir: true` when every task should run in one shared batch and
write to the shared result catalog. In this mode, the number of tasks overrides
`batch_size`:

```yaml
dispatch:
  worker: batch
  share_workdir: true
```

Use `worker: single` when every structure needs an independent local worker,
as in Monte Carlo exploration:

```yaml
dispatch:
  worker: single
```

`single` is supported only by driver runtimes. Reactor runtimes require
`worker: batch` and do not support `share_workdir` or `retain_info`.

The scheduler's `concurrent_tasks` controls how many simulations in a batch
run simultaneously; it does not change which simulations belong to the batch.
See {doc}`runtime-broadcast` for compatibility rules and a four-simulation,
four-GPU Slurm example. There is no separate variant-grouping option.
