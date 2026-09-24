(potential-nequip)=

# nequip

The `nequip` provider declares ASE and LAMMPS interfaces for NequIP, with an Allegro flavour for LAMMPS.

## Requirements

The ASE adapter expects PyTorch and the NequIP API `NequIPCalculator.from_deployed_model`. LAMMPS requires the matching `nequip` or `allegro` pair style and an exported model.

## Backends

| Potential backend | Executor | Default | Description |
| --- | --- | --- | --- |
| `ase` | `ase` | Yes | NequIP deployed-model Python calculator. |
| `lammps` | `lammps` | Yes | LAMMPS with the NequIP or Allegro pair style. |
| `lammps` | `ase` | No | LAMMPS evaluates energies and forces; ASE drives the calculation. |

Backend defaults depend on the executor. The comments in each configuration
show whether `potential.backend` can be omitted.

## Configurations

### ase + ase

```yaml
schema_version: 3
potential:
  provider: nequip
  backend: ase  # Optional; default for the ase executor.
  parameters:
    model: ./deployed_model.pth
    type_list: [H, O]
    estimate_uncertainty: false
executor:
  provider: ase
  method: spc
```

### lammps + lammps

```yaml
schema_version: 3
potential:
  provider: nequip
  backend: lammps  # Optional; default for the lammps executor.
  parameters:
    model: ./deployed_lammps_model.pth
    type_list: [H, O]
    command: lmp
executor:
  provider: lammps
  method: spc
```

### lammps + ase

```yaml
schema_version: 3
potential:
  provider: nequip
  backend: lammps  # Required; overrides the default backend for the ase executor.
  parameters:
    model: ./deployed_lammps_model.pth
    type_list: [H, O]
    command: lmp
executor:
  provider: ase
  method: spc
```

### Parameter notes

`type_list` maps chemical symbols to the same model type names. The ASE branch
selects CUDA when available and supports committee construction through
`estimate_uncertainty`. Its loader uses the deployed-model API, so arbitrary
newer checkpoint formats cannot be assumed compatible.

For the declared LAMMPS interface, `flavour` selects `nequip` (default) or
`allegro`, and `command` supplies the executable. Only the first model is used.
The adapter requests `newton off` for NequIP and `newton on` for Allegro.
Allegro is a flavour of `nequip`, not a separate provider name.
