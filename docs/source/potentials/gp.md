(potential-gp)=

# gp

The `gp` provider loads gdpx’s full or sparse Gaussian-process models into an ASE calculator.

## Requirements

Use a model archive written by gdpx’s Gaussian-process trainer or `gdpx.providers.gp.save_model`. This calculator uses gdpx’s NumPy-based implementation.

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
  provider: gp
  backend: ase  # Optional; default for the ase executor.
  parameters:
    model: [./gp_model.npz]
executor:
  provider: ase
  method: spc
```

Pass `model` as a nonempty **list**: the manager indexes its first entry
directly. Only that model is loaded. The archive stores the model class (`FGP`
or `SGP`), hyperparameters, descriptors, and fitted state; an arbitrary `.npz`
file is not sufficient.

Model parameters come from the archive rather than additional calculator
keywords. Committee selection is not implemented in this manager.
