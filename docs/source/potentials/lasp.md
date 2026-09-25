(potential-lasp)=

# lasp

The `lasp` provider loads LASP neural-network potential files.

## Requirements

Provide the LASP executable and trained potential files for every element in the calculation.

## Backends

| Potential backend | Executor | Default | Description |
| --- | --- | --- | --- |
| `lasp` | `lasp` | Yes | Native LASP execution. |
| `lasp` | `ase` | Yes | LASP calculator driven by ASE. |

Backend defaults depend on the executor. The comments in each configuration
show whether `potential.backend` can be omitted.

## Configurations

### lasp + lasp

```yaml
potential:
  provider: lasp
  backend: lasp  # Optional; default for the lasp executor.
  parameters:
    command: lasp
    type_list: [Cu, O]
    model: [./Cu.pot, ./O.pot]
executor:
  provider: lasp
  method: spc
```

### lasp + ase

```yaml
potential:
  provider: lasp
  backend: lasp  # Optional; default for the ase executor.
  parameters:
    command: lasp
    type_list: [Cu, O]
    model: [./Cu.pot, ./O.pot]
executor:
  provider: ase
  method: spc
```

### Parameter notes

`model` accepts a path or list of existing files. When the number of models
matches `type_list`, gdpx maps files to elements in that order. Otherwise the
first model file is assigned to every element, so use that form only for a
file that covers all requested species.
