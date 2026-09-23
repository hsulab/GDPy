(sampling-operator-swap-type)=

# `swap_type`

Change the chemical identity of one atom while keeping the total atom count
fixed. This provides composition changes without insertion or removal.

## Configuration

Place this fragment under `recipe.operators` for MC or `strategy.operators` for basin hopping,
or under top-level `operators` for hybrid MC:

```yaml
- method: swap_type
  particles: [Cu, Ni]
  chempots: [0.0, -0.5]
  temperature: 1000.0
```

## Settings

| Setting | Meaning | Default |
| --- | --- | --- |
| `particles` | At least two distinct atomic chemical symbols | Required |
| `chempots` | Chemical potentials in eV, in the same order as `particles` | Required |

See {ref}`sampling-operator-shared-settings` for temperature, relative selection
weights, regions, particle tags, and distance checks.

## Proposal and acceptance

An initial species is chosen from the eligible species currently present.
The operator selects one atom of that species and changes it to a different
configured species. The target species need not already be present. Each
selected particle must contain exactly one atom, so tag independent atoms
separately.

Coordinates are unchanged and the proposal does not perform geometric distance
checks. It changes atomic identity; velocities, charges, and magnetic moments
are not reinitialized for the new species. Ensure the runtime potential supports
all configured elements.

Acceptance uses the energy change with the chemical-potential offset
`mu_before - mu_after`, at the configured temperature. The values above are
illustrative search settings, not calibrated reservoir chemical potentials.
For fixed-composition positional swaps, use {doc}`swap`.
