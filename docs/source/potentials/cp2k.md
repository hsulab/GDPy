(potential-cp2k)=

# cp2k

The `cp2k` provider configures CP2K input templates and electronic settings.

## Requirements

Provide a CP2K executable, input template, and the basis/pseudopotential data
referenced by the input. Backend `interactive` requires a shell-capable CP2K
executable; replace `cp2k_shell.psmp` below with your shell command.

## Backends

| Potential backend | Executor | Default | Description |
| --- | --- | --- | --- |
| `cp2k` | `cp2k` | Yes | Native file-based CP2K execution. |
| `cp2k` | `ase` | Yes | File-based CP2K calculator driven by ASE. |
| `interactive` | `ase` | No | CP2K shell calculator driven by ASE. |

Backend defaults depend on the executor. The comments in each configuration
show whether `potential.backend` can be omitted.

## Configurations

### cp2k + cp2k

```yaml
potential:
  provider: cp2k
  backend: cp2k  # Optional; default for the cp2k executor.
  parameters:
    command: cp2k.psmp -i cp2k.inp -o cp2k.out
    template: ./cp2k-template.inp
    basis_set_file: /path/to/BASIS_MOLOPT
    potential_file: /path/to/GTH_POTENTIALS
executor:
  provider: cp2k
  method: spc
```

### cp2k + ase

```yaml
potential:
  provider: cp2k
  backend: cp2k  # Optional; default for the ase executor.
  parameters:
    command: cp2k.psmp -i cp2k.inp -o cp2k.out
    template: ./cp2k-template.inp
    basis_set_file: /path/to/BASIS_MOLOPT
    potential_file: /path/to/GTH_POTENTIALS
executor:
  provider: ase
  method: spc
```

### interactive + ase

```yaml
potential:
  provider: cp2k
  backend: interactive  # Required; overrides the default backend for the ase executor.
  parameters:
    command: cp2k_shell.psmp
    template: ./cp2k-template.inp
    basis_set_file: /path/to/BASIS_MOLOPT
    potential_file: /path/to/GTH_POTENTIALS
executor:
  provider: ase
  method: spc
```

### Parameter notes

`template` supplies the CP2K input text. When present, the adapter leaves
`cutoff`, `max_scf`, and `xc` to the template unless explicitly supplied as
calculator parameters. Basis and potential file paths are resolved absolutely.

Set `potential.backend: interactive` alongside `provider: cp2k` to select
the shell calculator with an ASE executor and a shell-capable CP2K command.
Both ASE and native CP2K executors default to backend `cp2k`.
The old `parameters.interface: cp2k_shell` setting must migrate to
`potential.backend: interactive`.
