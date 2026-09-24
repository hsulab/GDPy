(potential-reann)=

# reann

The `reann` provider loads REANN TorchScript models with gdpx’s ASE calculator.

## Requirements

From the repository root:

```shell
python -m pip install -e '.[reann]'
```

The extra installs PyTorch and `opt_einsum`. Supply an exported TorchScript
potential such as `PES.pt`; training checkpoints such as `REANN.pth` must be
exported first. gdpx uses ASE neighbour lists, so inference does not require
upstream REANN source or its Fortran extension. See {doc}`../installation`
for CPU/GPU setup and verification. This provider supports ASE only.

### Training and exporting models

Obtain the training code separately (Git is required):

```shell
git clone --branch v_1.0 --depth 1 https://github.com/zhangylch/REANN.git /path/to/REANN
```

Replace `/path/to/REANN` with your source directory and follow that checkout's
README and manual to set up training and export. The tag has no pip packaging;
the `reann` extra installs its Python dependencies only.

Set the gdpx trainer's `command` and `freeze_command` to invoke your scripts.
The defaults, `train` and `freeze`, are not installed by gdpx. The trainer
expects `REANN.pth` as the training checkpoint and `PES.pt` as the exported
potential.

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
  provider: reann
  backend: ase  # Optional; default for the ase executor.
  parameters:
    model: ./PES.pt
    type_list: [H, O]
    precision: float32
    compute_stress: false
    estimate_uncertainty: false
executor:
  provider: ase
  method: spc
```

`type_list` is passed as the model’s atom-type ordering. `precision` must be
`float32` (default) or `float64`. `compute_stress` defaults to `false`; enable
it for calculations that require stress. CUDA is selected when available,
otherwise CPU; Apple MPS is not selected.

`model` accepts one path or a list of existing paths. Multiple models with
`estimate_uncertainty: true` form a committee; otherwise only the first model
is evaluated.
