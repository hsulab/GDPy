(potential-deepmd)=

# deepmd

The `deepmd` provider loads Deep Potential models.

## Requirements

From the repository root, choose the extra matching your model, for example:

```shell
python -m pip install -e '.[deepmd3-torch]'
```

| Installation option | DeepMD 2 | DeepMD 3 |
| --- | --- | --- |
| Installed separately | `deepmd2` | `deepmd3` |
| PyTorch | Not supported | `deepmd3-torch` |
| TensorFlow CPU | `deepmd2-cpu` | `deepmd3-cpu` |
| TensorFlow GPU, existing CUDA/cuDNN | `deepmd2-gpu` | `deepmd3-gpu` |
| TensorFlow GPU, install CUDA 12 runtime | `deepmd2-cu12` | `deepmd3-cu12` |

Every extra includes `dpdata`. Use separate environments for DeepMD 2 and 3;
combine `deepmd3-torch` with a DeepMD 3 TensorFlow extra to install support for
both frameworks.
Both versions use `provider: deepmd` in configuration.

See {doc}`../installation` for the recommended GPU installation and
{ref}`gpu-verification` for checks. LAMMPS requires a separate binary with the
DeepMD pair style and a compatible exported model.

## Backends

| Potential backend | Executor | Default | Description |
| --- | --- | --- | --- |
| `ase` | `ase` | Yes | DeepMD Python calculator. |
| `lammps` | `lammps` | Yes | LAMMPS with the DeepMD pair style. |
| `lammps` | `ase` | No | LAMMPS evaluates energies and forces; ASE drives the calculation. |

Backend defaults depend on the executor. The comments in each configuration
show whether `potential.backend` can be omitted.

## Configurations

### ase + ase

```yaml
potential:
  provider: deepmd
  backend: ase  # Optional; default for the ase executor.
  parameters:
    model: ./graph.pb
    type_list: [H, O]
    estimate_uncertainty: false
executor:
  provider: ase
  method: spc
```

### lammps + lammps

```yaml
potential:
  provider: deepmd
  backend: lammps  # Optional; default for the lammps executor.
  parameters:
    model: ./graph.pb
    type_list: [H, O]
    command: lmp
executor:
  provider: lammps
  method: spc
```

### lammps + ase

```yaml
potential:
  provider: deepmd
  backend: lammps  # Required; overrides the default backend for the ase executor.
  parameters:
    model: ./graph.pb
    type_list: [H, O]
    command: lmp
executor:
  provider: ase
  method: spc
```

### Parameter notes

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
