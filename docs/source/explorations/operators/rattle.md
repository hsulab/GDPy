(sampling-operator-rattle)=

# `rattle`

Translate a random subset of eligible particles in one proposal, allowing
several local environments to change before relaxation.

## Configuration

Place this fragment under `strategy.operators`. Preset MC and hybrid MC set
temperature under `system.ensemble`; custom MC/HMC and basin hopping add it to
the operator:

```yaml
- method: rattle
  particles: [Cu, Ni]
  rattle_strength: 0.8
  rattle_prop: 0.4
```

## Settings

| Setting | Meaning | Default |
| --- | --- | --- |
| `particles` | Eligible symbols or molecular formulas | Required |
| `rattle_strength` | Half-width of each Cartesian displacement interval, in Å; finite and positive | 0.8 |
| `rattle_prop` | Independent selection probability per eligible particle; in `(0, 1]` | 0.4 |

See {ref}`sampling-operator-shared-settings` for temperature, relative selection
weights, regions, particle tags, and distance checks.

## Proposal and acceptance

Each eligible tagged group is selected independently with probability
`rattle_prop`. Each selected group receives its own displacement, with x, y,
and z components sampled uniformly from `[-rattle_strength, rattle_strength]`.
All atoms within a group receive the same translation; groups are not rotated.
The displacement length can therefore exceed `rattle_strength`.

An attempt selecting no groups is retried. Distance checks include interactions
between the moved particles. Attempts continue up to `max_random_attempts`;
no eligible groups or no successful attempt gives an invalid proposal.

This preserves composition and uses energy/temperature acceptance. Set
`rattle_prop: 1.0` to translate every eligible group in each attempt. This
particle-selection probability is separate from the operator's `probability`
weight in a mixed operator list.
