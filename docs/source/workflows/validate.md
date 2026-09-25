# validate

A `validate` step consumes a validator resource, a structure-producing step,
and optionally a worker supplied by another node. The dependencies are
explicit under `inputs`:

```yaml
workflow:
  targets: validation

resources:
  validator:
    __type__: validator
    options:
      method: minima

steps:
  structures:
    __type__: read_stru
    options:
      fname: ./dataset.xyz

  validation:
    __type__: validate
    inputs:
      validator: validator
      structures: structures
```

Validator-specific parameters vary by registered implementation. Run
`gdp workflow validate workflow.yaml` to check node types and constructor
arguments without constructing the nodes.
