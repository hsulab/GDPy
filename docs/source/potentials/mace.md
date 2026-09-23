(potential-mace)=

# mace

The `mace` provider loads local MACE checkpoints.

## Installation

From the repository root:

```shell
python -m pip install -e '.[mace]'
```

The extra installs `mace-torch` and PyTorch for inference and training,
including `mace_run_train`. See {doc}`../installation` for CPU/GPU setup,
verification, and dependency conflicts with other providers.

LAMMPS requires a separate binary with the MACE pair style and a compatible
exported model.

## Configuration

```yaml
potential:
  provider: mace
  parameters:
    model: ./mace.model
    type_list: [H, O]
    precision: float32
    estimate_uncertainty: false
```

`model` accepts an existing path or list of paths. The ASE adapter passes
`precision` (default `float32`) as `default_dtype`, selecting CUDA if available
and CPU otherwise. Multiple models with `estimate_uncertainty: true` enable
a committee; otherwise only the first is used. This adapter expects local
files rather than foundation-model names.

The LAMMPS interface requires exactly one exported model; `command` defaults
to `lmp`.

See {doc}`../trainers/mace` for training.
