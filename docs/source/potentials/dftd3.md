(potential-dftd3)=

# dftd3

The `dftd3` provider wraps the `DFTD3` ASE dispersion calculator.

## Requirements

Install the package providing `dftd3.ase` and its native dispersion library in the gdpx environment.

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
For a combined ASE calculation, add DFT-D3 as a runtime modifier:

```yaml
modifiers:
  - provider: dftd3
    backend: ase
    parameters:
      method: PBE
      damping: d3bj
```

The optional `backend` selects the implementation (default: `ase`);
`parameters.method` specifies the functional. See {doc}`vasp` for a complete example.
