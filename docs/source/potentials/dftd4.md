(potential-dftd4)=

# dftd4

The `dftd4` provider wraps the `DFTD4` ASE dispersion calculator.

## Requirements

Install the package providing `dftd4.ase` and its native dispersion library in the gdpx environment.

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
  provider: dftd4
  backend: ase  # Optional; default for the ase executor.
  parameters:
    method: PBE
executor:
  provider: ase
  method: spc
```

`parameters.method` identifies the exchange-correlation functional used
for the dispersion parameters. All potential parameters pass to the upstream
ASE calculator.

This calculator evaluates the **dispersion contribution only**. Selecting it
as the potential does not also run an electronic-structure calculation.
DFT-D4 is available as a standalone potential only. See {doc}`dftd3` for
the supported DFT-D3 modifier configuration.
