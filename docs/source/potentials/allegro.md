(potential-allegro)=

# allegro

The `allegro` provider evaluates Allegro models directly through ASE or through
LAMMPS using its own `AllegroManager`. NequIP models use the separate {doc}`nequip` provider.

## Requirements

For backend `ase`, install NequIP with
`nequip.integrations.ase.NequIPCalculator.from_compiled_model` and the
dependencies required by your compiled Allegro model. Compile the model for
ASE, matching the device used for inference; see the
[upstream ASE guide](https://nequip.readthedocs.io/en/latest/integrations/ase.html).
This route needs no LAMMPS executable.

For backend `lammps`, provide a LAMMPS executable with the `allegro` pair style
and a compatible exported model. ASE-target and LAMMPS-target model artifacts
are not interchangeable.

## Backends

| Potential backend | Executor | Default | Description |
| --- | --- | --- | --- |
| `ase` | `ase` | Yes | Direct inference through NequIPCalculator. |
| `lammps` | `lammps` | Yes | Native LAMMPS execution with the Allegro pair style. |
| `lammps` | `ase` | No | LAMMPS evaluates energies and forces; ASE drives the calculation. |

## Configurations

### ase + ase

```yaml
schema_version: 3
potential:
  provider: allegro
  backend: ase  # Optional; default for the ase executor.
  parameters:
    model: ./allegro-ase.nequip.pt2
    device: cpu
    type_list: [H, O]
executor:
  provider: ase
  method: min
  parameters:
    steps: 100
    fmax: 0.05
```

The ASE backend loads models with `NequIPCalculator.from_compiled_model`.
`device` defaults to `cpu`; set it to match the compiled artifact. `type_list`
provides an identity mapping from chemical symbols to model type names. For
custom names, supply `chemical_species_to_atom_type_map` explicitly instead.
Other options, including `neighborlist_backend` and unit conversion factors,
pass to the upstream loader. Multiple models with `estimate_uncertainty: true`
create a committee; otherwise only the first model is evaluated.

### lammps + lammps

```yaml
schema_version: 3
potential:
  provider: allegro
  backend: lammps  # Optional; default for the lammps executor.
  parameters:
    model: ./allegro-deployed.pth
    command: lmp
executor:
  provider: lammps
  method: spc
```

### lammps + ase

```yaml
schema_version: 3
potential:
  provider: allegro
  backend: lammps  # Required; the ase executor defaults to ase.
  parameters:
    model: ./allegro-deployed.pth
    command: lmp
executor:
  provider: ase
  method: min
  parameters:
    steps: 100
    fmax: 0.05
```

The LAMMPS model must be an existing exported file. A list is accepted, but
only its first model is evaluated. The LAMMPS adapter uses `metal` units, `atomic` atom style,
and `newton on`. With the ASE executor, LAMMPS runs single-point evaluations
and ASE controls ionic motion.

Migration: replace `provider: nequip` plus `parameters.flavour: allegro` with
`provider: allegro`, and remove `flavour`. Use `backend: lammps` explicitly to preserve LAMMPS evaluation with an ASE
executor. NequIP's trainer remains registered under `nequip`; this provider
does not add a separate Allegro training integration.
