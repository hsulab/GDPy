(potential-deepmd)=

# deepmd

The `deepmd` provider loads Deep Potential models.

## Requirements

The ASE interface requires `deepmd-kit` (the adapter handles major versions 2 and 3). LAMMPS execution requires a binary with the DeepMD pair style and a model compatible with that build.

## Configuration

```yaml
potential:
  provider: deepmd
  parameters:
    model: ./graph.pb
    type_list: [H, O]
    estimate_uncertainty: false
```

`model` accepts one existing checkpoint or a list. `models` is also accepted
as an alias when `model` is absent. `type_list` defines the model’s element
mapping and must agree with its training configuration. Optional `head` is
passed to the ASE DeepMD calculator for models that support it.

For ASE, multiple models with `estimate_uncertainty: true` create a committee;
otherwise only the first is evaluated. For the LAMMPS interface, optional
`command` supplies the executable (default `lmp`).

Checkpoint formats depend on the installed DeepMD backend; the `.pb` example
is not a format requirement for every backend. See {doc}`../trainers/deepmd`
for training configuration.
