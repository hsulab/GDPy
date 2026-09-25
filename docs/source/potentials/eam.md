(potential-eam)=

# eam

The `eam` provider reads embedded-atom potential files.

## Requirements

ASE supplies the Python calculator. For LAMMPS execution, install a binary with the chosen EAM pair style. Supply an existing potential file.

## Backends

| Potential backend | Executor | Default | Description |
| --- | --- | --- | --- |
| `ase` | `ase` | Yes | ASE EAM calculator. |
| `lammps` | `lammps` | Yes | LAMMPS with the selected EAM pair style. |

Backend defaults depend on the executor. The comments in each configuration
show whether `potential.backend` can be omitted.

## Configurations

### ase + ase

```yaml
potential:
  provider: eam
  backend: ase  # Optional; default for the ase executor.
  parameters:
    model: ./Cu.eam.alloy
    flavour: eam/alloy
    type_list: [Cu]
executor:
  provider: ase
  method: spc
```

### lammps + lammps

```yaml
potential:
  provider: eam
  backend: lammps  # Optional; default for the lammps executor.
  parameters:
    model: ./Cu.eam.alloy
    flavour: eam/alloy
    type_list: [Cu]
    command: lmp
executor:
  provider: lammps
  method: spc
```

### Parameter notes

`model` accepts a path or list of paths; only the first file is used.
ASE reads the file using its `EAM` calculator. For the LAMMPS interface,
`command` supplies the executable (default `lmp`).

The allowed `flavour` values are `eam` (default), `eam/alloy`, `eam/cd`,
`eam/fs`, and `eam/he`. These select the LAMMPS pair style; ASE’s own file-format
support still applies when using ASE. LAMMPS uses `metal` units and `atomic`
atom style, with element mapping appended to `pair_coeff` (default `* *`).
