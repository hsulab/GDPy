(potential-emt)=

# emt

The `emt` provider exposes ASE’s effective-medium-theory calculator.

## Requirements

EMT is included with ASE and needs no external model files.

## Configuration

```yaml
potential:
  provider: emt
  parameters:
    asap_cutoff: false
```

The `parameters` field is optional; omit it to use ASE’s `EMT` defaults.
When supplied, its settings pass to ASE’s `EMT` calculator.
Use this provider for lightweight metal examples such as copper clusters.
The calculator has a fixed element parameter table, so it is not a universal
potential for arbitrary chemistry.

See {doc}`../global_optimisation/examples/cluster` for a cluster search example.
