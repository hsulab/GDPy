(potential-abacus)=

# abacus

The `abacus` provider configures ABACUS electronic-structure calculations.

## Requirements

Install ABACUS and an ASE distribution that provides `ase.calculators.abacus` and `ase.io.abacus`. Supply an INPUT template, pseudopotentials, and basis files.

## Backends

| Potential backend | Executor | Default | Description |
| --- | --- | --- | --- |
| `abacus` | `abacus` | Yes | Native ABACUS execution. |
| `abacus` | `ase` | Yes | ABACUS calculator driven by ASE. |

Backend defaults depend on the executor. The comments in each configuration
show whether `potential.backend` can be omitted.

## Configurations

### abacus + abacus

```yaml
schema_version: 3
potential:
  provider: abacus
  backend: abacus  # Optional; default for the abacus executor.
  parameters:
    command: mpirun -n 4 abacus
    template: ./INPUT_ABACUS
    pseudo_dir: /path/to/pseudopotentials
    basis_dir: /path/to/orbitals
    kpts: [1, 1, 1]
    type_info:
      Si:
        pseudo: Si.upf
        basis: Si.orb
executor:
  provider: abacus
  method: scf
```

### abacus + ase

```yaml
schema_version: 3
potential:
  provider: abacus
  backend: abacus  # Optional; default for the ase executor.
  parameters:
    command: mpirun -n 4 abacus
    template: ./INPUT_ABACUS
    pseudo_dir: /path/to/pseudopotentials
    basis_dir: /path/to/orbitals
    kpts: [1, 1, 1]
    type_info:
      Si:
        pseudo: Si.upf
        basis: Si.orb
executor:
  provider: ase
  method: spc
```

### Parameter notes

`template`, `pseudo_dir`, and `basis_dir` are required by the manager.
Each `type_info` entry maps an element to its pseudopotential and basis filename.
`kpts` defaults to `[1, 1, 1]`.

The input template must specify `calculation scf` (or omit it to use that
default); other template calculation types are rejected during materialization.
