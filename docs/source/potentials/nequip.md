(potential-nequip)=

# nequip

The `nequip` provider evaluates NequIP models through ASE or LAMMPS.
For Allegro models, use the separate {doc}`allegro` provider.

## Requirements

The ASE adapter expects PyTorch and the NequIP API `NequIPCalculator.from_deployed_model`. LAMMPS requires the `nequip` pair style and an exported model.

## Backends

| Potential backend | Executor | Default | Description |
| --- | --- | --- | --- |
| `ase` | `ase` | Yes | NequIP deployed-model Python calculator. |
| `lammps` | `lammps` | Yes | LAMMPS with the NequIP pair style. |
| `lammps` | `ase` | No | LAMMPS evaluates energies and forces; ASE drives the calculation. |

Backend defaults depend on the executor. The comments in each configuration
show whether `potential.backend` can be omitted.

## Configurations

### ase + ase

```yaml
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

For the LAMMPS backend, `command` supplies the executable (default `lmp`).
Only the first model is used. The adapter requests `newton off` for NequIP.

To migrate `provider: nequip` with `parameters.flavour: allegro`, use
`provider: allegro` and remove `flavour`. Keep `backend: lammps` explicitly when using Allegro with an ASE executor;
Allegro now defaults to direct ASE inference for that executor. Remove a redundant
`flavour: nequip` from NequIP configurations as well.
