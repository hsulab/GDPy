(potential-nequip)=

# nequip

The `nequip` provider declares ASE and LAMMPS interfaces for NequIP, with an Allegro flavour for LAMMPS.

## Requirements

The ASE adapter expects PyTorch and the NequIP API `NequIPCalculator.from_deployed_model`. LAMMPS requires the matching `nequip` or `allegro` pair style and an exported model.

## Configuration

```yaml
potential:
  provider: nequip
  parameters:
    model: ./deployed_model.pth
    type_list: [H, O]
    estimate_uncertainty: false
```

## Current limitation

The current manager constructs a calculator but does not assign it to
`self.calc`. Materialization therefore retains the placeholder calculator.
The configuration above documents the intended interface; calculations need
this implementation issue resolved before use.

`type_list` maps chemical symbols to the same model type names. The ASE branch
selects CUDA when available and supports committee construction through
`estimate_uncertainty`. Its loader uses the deployed-model API, so arbitrary
newer checkpoint formats cannot be assumed compatible.

For the declared LAMMPS interface, `flavour` selects `nequip` (default) or
`allegro`, and `command` supplies the executable. Only the first model is used.
The adapter requests `newton off` for NequIP and `newton on` for Allegro.
Allegro is a flavour of `nequip`, not a separate provider name.
