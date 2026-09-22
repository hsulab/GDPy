(potential-abacus)=

# abacus

The `abacus` provider configures ABACUS electronic-structure calculations.

## Requirements

Install ABACUS and an ASE distribution that provides `ase.calculators.abacus` and `ase.io.abacus`. Supply an INPUT template, pseudopotentials, and basis files.

## Configuration

```yaml
potential:
  provider: abacus
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
```

`template`, `pseudo_dir`, and `basis_dir` are required by the manager.
Each `type_info` entry maps an element to its pseudopotential and basis filename.
`kpts` defaults to `[1, 1, 1]`.

The input template must specify `calculation scf` (or omit it to use that
default); other template calculation types are rejected during materialization.
