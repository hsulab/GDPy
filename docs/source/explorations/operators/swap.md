(sampling-operator-swap)=

# `swap`

Rearrange two particle types while preserving their total counts.

## Configuration

Place this fragment under `strategy.operators`. Preset MC and hybrid MC set
temperature under `system.ensemble`; custom MC/HMC and basin hopping add it to
the operator:

```yaml
- method: swap
  particles: [Cu, Ni]
  swap_mode: atomic
  check_used_pairs: true
```

## Settings

| Setting | Meaning | Default |
| --- | --- | --- |
| `particles` | Exactly two distinct particle types | Required |
| `swap_mode` | `atomic` or `cop_z` | `atomic` |
| `check_used_pairs` | Avoid retrying the same pair within a proposal | `false` |

See {ref}`sampling-operator-shared-settings` for temperature, relative selection
weights, regions, particle tags, and distance checks.

## Proposal and acceptance

The operator chooses one tagged particle of each type. Both types must be
present in the selection region. In `atomic` mode it exchanges their position
arrays, including rotation of molecular groups. Use this mode for atomic swaps
or groups with compatible atom counts; it is not a general swap of unequal-size
molecules.

In `cop_z` mode, each group is translated so that its lowest-z atom lies at the
other group's original center of positions. This gives a z-oriented placement
rule for molecular particles rather than a direct exchange of atom coordinates.

Pairs are retried up to `max_random_attempts` while checking distances.
`check_used_pairs` tracks attempted pairs only within the current proposal.
The swap distance check requires non-isolated placement even if the shared
`allow_isolated` option is enabled. Missing species or exhausted attempts gives
an invalid proposal.

Acceptance uses energy and temperature; no chemical potentials are needed
because the composition is unchanged. To change elemental counts, use
{doc}`swap_type`.
