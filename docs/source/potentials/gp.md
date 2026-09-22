(potential-gp)=

# gp

The `gp` provider loads GDPy’s full or sparse Gaussian-process models into an ASE calculator.

## Requirements

Use a model archive written by GDPy’s Gaussian-process trainer or `gdpx.providers.gp.save_model`. This calculator uses GDPy’s NumPy-based implementation.

## Configuration

```yaml
potential:
  provider: gp
  parameters:
    model: [./gp_model.npz]
```

Pass `model` as a nonempty **list**: the manager indexes its first entry
directly. Only that model is loaded. The archive stores the model class (`FGP`
or `SGP`), hyperparameters, descriptors, and fitted state; an arbitrary `.npz`
file is not sufficient.

Model parameters come from the archive rather than additional calculator
keywords. Committee selection is not implemented in this manager.
