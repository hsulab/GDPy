(potential-dftd3)=

# dftd3

The `dftd3` provider wraps the `DFTD3` ASE dispersion calculator.

## Requirements

Install the package providing `dftd3.ase` and its native dispersion library in the GDPy environment.

## Configuration

```yaml
potential:
  provider: dftd3
  parameters:
    method: PBE
    damping: d3bj
```

`parameters.method` identifies the exchange-correlation functional used
for the dispersion parameters. All potential parameters pass to the upstream
ASE calculator.

This calculator evaluates the **dispersion contribution only**. Selecting it
as the potential does not also run an electronic-structure calculation.
For a combined calculation, use an explicitly supported combined interface,
such as VASP’s `vasp_interactive_disp` option described in {doc}`vasp`.
The removed `mixer` potential is not a schema-version-3 provider.
