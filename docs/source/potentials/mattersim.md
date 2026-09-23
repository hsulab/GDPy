(potential-mattersim)=

# mattersim

The `mattersim` provider loads MatterSim pretrained models and local checkpoints.

## Installation

From the repository root:

```shell
python -m pip install -e '.[mattersim]'
```

The extra installs MatterSim and its PyTorch dependencies. See
{doc}`../installation` for the recommended GPU installation and verification.
gdpx uses CUDA when available and otherwise uses CPU.

## Configuration

```yaml
potential:
  provider: mattersim
  parameters:
    model: MatterSim-v1.0.0-1M
    compute_stress: false
```

`compute_stress: false` avoids unnecessary stress evaluation for fixed-cell
molecules and clusters. Keep stress enabled when the calculation requires it.

## Models

The `model` parameter accepts either a MatterSim pretrained-model name or a
local checkpoint path:

| Model | Use |
| --- | --- |
| `MatterSim-v1.0.0-1M` | Smallest and fastest pretrained model; gdpx's default. |
| `MatterSim-v1.0.0-5M` | Larger pretrained model with greater computational cost. |
| `/path/to/model.pth` | A local MatterSim checkpoint. |

The `.pth` suffix is optional for either pretrained name. MatterSim downloads a
named checkpoint automatically on first use; local paths must already exist.

## Usage notes

- Use a sufficiently large periodic vacuum box for an isolated molecule or
  cluster. This avoids interactions with periodic images while following the
  periodic representation expected by materials-oriented models.
- MatterSim relaxes every atomic coordinate; molecular tags guide gdpx's
  genetic operations but do not constrain intramolecular geometry during
  minimisation.
- Validate the selected checkpoint for the target chemistry and property before
  treating results as production data. A universal pretrained potential and a
  small demonstration search do not by themselves establish accuracy.

See {ref}`ga-water-cluster-example` for a complete four-water global-
optimisation example using the 1M model.
