(potential-eam)=

# eam

The `eam` provider reads embedded-atom potential files.

## Requirements

ASE supplies the Python calculator. For LAMMPS execution, install a binary with the chosen EAM pair style. Supply an existing potential file.

## Configuration

```yaml
potential:
  provider: eam
  parameters:
    model: ./Cu.eam.alloy
    flavour: eam/alloy
    type_list: [Cu]
```

`model` accepts a path or list of paths; only the first file is used.
ASE reads the file using its `EAM` calculator. For the LAMMPS interface,
`command` supplies the executable (default `lmp`).

The allowed `flavour` values are `eam` (default), `eam/alloy`, `eam/cd`,
`eam/fs`, and `eam/he`. These select the LAMMPS pair style; ASE’s own file-format
support still applies when using ASE. LAMMPS uses `metal` units and `atomic`
atom style, with element mapping appended to `pair_coeff` (default `* *`).
