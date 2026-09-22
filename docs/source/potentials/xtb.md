(potential-xtb)=

# xtb

The `xtb` provider exposes the xTB Python ASE calculator for semiempirical calculations.

## Requirements

Install the Python package exposing `xtb.ase.calculator.XTB` and its native library in the GDPy environment.

## Configuration

```yaml
potential:
  provider: xtb
  parameters:
    method: GFN2-xTB
    accuracy: 1.0
    electronic_temperature: 300.0
    max_iterations: 250
```

Calculator options pass directly to `XTB`. `parameters.method` selects the
xTB Hamiltonian; the potential component’s method remains `default`.
Other calculator options include `solvent` and `cache_api`.

The registered provider uses the `xtb` interface. TBLite is not selectable
through this potential configuration.
