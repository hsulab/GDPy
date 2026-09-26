# runtime configurations

A runtime combines a `potential` with an `executor`. It may also define
`modifiers`, a `scheduler`, and a worker `dispatch` policy. The potential
supplies energies and forces; the executor selects the calculation and the
software interface used to run it.

Start with one complete runtime:

```yaml
potential:
  provider: emt
executor:
  provider: ase
  method: min
  parameters:
    fmax: 0.05
    steps: 300
```

Choose the guide that matches how the runtime will be used:

| Configuration | Use it for |
| --- | --- |
| {doc}`runtime-single` | One potential/executor pairing, including modifiers and provider compatibility. |
| {doc}`runtime-dispatch` | Batching structures and selecting batch or single-worker execution. |
| {doc}`runtime-broadcast` | Independent calculations created by varying executor parameters. |
| {doc}`runtime-chain` | Ordered workflow stages such as equilibration followed by production MD. |
| {doc}`schedulers` | Local execution, queue schedulers, SSH, and machine resources. |

```{toctree}
:hidden:
:maxdepth: 1

runtime-single.md
runtime-dispatch.md
runtime-broadcast.md
runtime-chain.md
```

In standalone runtime YAML, these fields are top-level; do not wrap them in a
`runtime` key. User-authored files omit `schema_version` and use the current
schema automatically. Saved runtime snapshots and compute plans record a
version so incompatible persisted data can be rejected.

Use the shared {doc}`units <../units>` unless a parameter explicitly documents
another unit.
