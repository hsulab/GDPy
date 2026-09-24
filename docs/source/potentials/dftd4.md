(potential-dftd4)=

# dftd4

The `dftd4` provider wraps the `DFTD4` ASE dispersion calculator.

## Requirements

Install the package providing `dftd4.ase` and its native dispersion library in the gdpx environment.

## Configuration

```yaml
potential:
  provider: dftd4
  parameters:
    method: PBE
```

`parameters.method` identifies the exchange-correlation functional used
for the dispersion parameters. All potential parameters pass to the upstream
ASE calculator.

This calculator evaluates the **dispersion contribution only**. Selecting it
as the potential does not also run an electronic-structure calculation.
DFT-D4 modifier registration is not provided by this change. The old VASP
combined interface supported DFT-D3 only; see {doc}`dftd3` for the supported
modifier configuration.
