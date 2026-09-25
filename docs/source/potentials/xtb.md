(potential-xtb)=

# xtb

The `xtb` provider exposes the xTB Python ASE calculator for semiempirical calculations.

## Requirements

For backend `xtb`, install the Python package exposing `xtb.ase.calculator.XTB`
and its native library. For backend `tblite`, install the package exposing
`tblite.ase.TBLite` and its native library.

## Backends

| Potential backend | Executor | Default | Description |
| --- | --- | --- | --- |
| `xtb` | `ase` | Yes | xTB Python calculator. |
| `tblite` | `ase` | No | TBLite Python calculator. |

Backend defaults depend on the executor. The comments in each configuration
show whether `potential.backend` can be omitted.

## Configurations

### xtb + ase

```yaml
potential:
  provider: xtb
  backend: xtb  # Optional; default for the ase executor.
  parameters:
    method: GFN2-xTB
    accuracy: 1.0
    electronic_temperature: 300.0
    max_iterations: 250
executor:
  provider: ase
  method: spc
```

### tblite + ase

```yaml
potential:
  provider: xtb
  backend: tblite  # Required; overrides the default backend for the ase executor.
  parameters:
    method: GFN2-xTB
executor:
  provider: ase
  method: spc
```

### Parameter notes

Calculator options pass directly to the selected `XTB` or `TBLite` calculator. `parameters.method` selects the
xTB Hamiltonian; the potential component’s method remains `default`.
Other calculator options include `solvent` and `cache_api`.

The default is `xtb`; select `backend: tblite` explicitly for TBLite.
The two calculators accept different optional parameters; the TBLite example
uses only the shared Hamiltonian setting.
