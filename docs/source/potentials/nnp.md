(potential-nnp)=

# nnp

The `nnp` provider exposes gdpx’s element-wise ACSF neural-network calculator through ASE.

## Requirements

Use a version-2 model archive produced by the current gdpx NNP trainer. Inference uses gdpx’s NumPy implementation.

## Backends

| Potential backend | Executor | Default | Description |
| --- | --- | --- | --- |
| `ase` | `ase` | Yes | ASE-compatible calculator. |

Backend defaults depend on the executor. The comments in each configuration
show whether `potential.backend` can be omitted.

## Configurations

### ase + ase

```yaml
schema_version: 3
potential:
  provider: nnp
  backend: ase  # Optional; default for the ase executor.
  parameters:
    model_file: ./nnp_model.npz
    type_map: [H, O]
executor:
  provider: ase
  method: spc
```

The required path key is `model_file`, not `model`. Optional `type_map` is a
list of permitted elements; it defaults to the elements saved in the model.
Every requested element must occur in the archive.

The calculator restores symmetry functions, cutoff, network architecture,
normalisation, and atomic offsets from the model. Legacy archives without a
version marker, and versions other than 2, are rejected. Retrain them with
the current trainer.

Energy and forces are implemented; stress is not. See {doc}`../trainers/nnp`
for training configuration.
