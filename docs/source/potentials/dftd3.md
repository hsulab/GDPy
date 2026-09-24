(potential-dftd3)=

# dftd3

The `dftd3` provider wraps the `DFTD3` ASE dispersion calculator.

## Requirements

Install the package providing `dftd3.ase` and its native dispersion library in the gdpx environment.

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
  provider: dftd3
  backend: ase  # Optional; default for the ase executor.
  parameters:
    method: PBE
    damping: d3bj
executor:
  provider: ase
  method: spc
```

`parameters.method` identifies the exchange-correlation functional used
for the dispersion parameters. All potential parameters pass to the upstream
ASE calculator.

This calculator evaluates the **dispersion contribution only**. Selecting it
as the potential does not also run an electronic-structure calculation.
### DFT-D3 modifier

For a combined ASE calculation, add DFT-D3 as a runtime modifier:

```yaml
modifiers:
  - provider: dftd3
    backend: ase  # Optional; default for the dftd3 modifier.
    parameters:
      method: PBE
      damping: d3bj
```

The optional `backend` selects the implementation (default: `ase`);
`parameters.method` specifies the functional. See {doc}`vasp` for a complete example.
