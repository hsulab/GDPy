# Train

Training is a provider capability. A trainer and the potential it produces must
belong to the same provider. The `train` operation consumes a dataset,
trainer component, potential component, and scheduler, and returns an updated
potential component containing the trained model artifacts.

A trainer variable is declared with `provider`, optional `method`, and
`parameters`:

```
trainer:
  type: trainer
  provider: nnp
  method: default
  parameters:
    config:
      n_epochs: 500
```

Use `size` to train a committee. The resulting potential can be combined
with any compatible executor through a new runtime; training does not own or
select the execution software.
