(potential-reann)=

# reann

The `reann` provider loads REANN TorchScript models with GDPy’s ASE calculator.

## Requirements

Install PyTorch and supply exported REANN TorchScript checkpoints. GDPy’s calculator uses ASE neighbour lists rather than the upstream Fortran neighbour-list extension.

## Configuration

```yaml
potential:
  provider: reann
  parameters:
    model: ./REANN.pt
    type_list: [H, O]
    precision: float32
    compute_stress: false
    estimate_uncertainty: false
```

`type_list` is passed as the model’s atom-type ordering. `precision` must be
`float32` (default) or `float64`. `compute_stress` defaults to `false`; enable
it for calculations that require stress. CUDA is selected when available,
otherwise CPU.

`model` accepts one path or a list of existing paths. Multiple models with
`estimate_uncertainty: true` form a committee; otherwise only the first model
is evaluated.
