(potential-reax)=

# reax

The `reax` provider configures a ReaxFF reactive force field.

## Requirements

Provide a ReaxFF force-field file and a LAMMPS binary that accepts the adapter’s `reax/c` pair style.

## Configuration

```yaml
potential:
  provider: reax
  parameters:
    command: lmp
    model: ./ffield.reax
    type_list: [H, O]
```

The adapter resolves `model` to an absolute path and sets
`pair_style reax/c NULL`, `units real`, and `atom_style charge`.
`type_list` identifies the elements in the calculation.

This is a legacy adapter: it does not configure a charge-equilibration fix in
its manager. Check the generated LAMMPS input and the requirements of your
force field before running. A binary that only provides a differently named
ReaxFF style is not compatible with this hard-coded style.
