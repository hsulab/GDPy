(sampling-operator-swap-type)=

# `swap_type`

Change the chemical identity of one atom while keeping the total atom count
fixed. This provides composition changes without insertion or removal.

## Configuration

Place this fragment under `strategy.operators` for MC or basin hopping, or
under top-level `operators` for hybrid MC. Preset MC reads temperature and
chemical potentials from `system.ensemble`; custom MC and the other methods add
them to the operator:

```yaml
- method: swap_type
  particles: [Cu, Ni]
```

## Settings

| Setting | Meaning | Default |
| --- | --- | --- |
| `particles` | At least two distinct atomic chemical symbols | Required |
| `chempots` | Chemical potentials for custom MC, BH, and hybrid MC; preset MC uses the ensemble mapping | Required outside preset MC |

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
`mu_before - mu_after`, at the configured temperature. It also includes the
reverse/forward proposal ratio from selecting a species and then one atom of
that species. Chemical potentials must use the same energy reference as the
runtime potential.
For fixed-composition positional swaps, use {doc}`swap`.
