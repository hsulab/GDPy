(sampling-operator-move)=

# `move`

Translate one randomly selected particle. This is a simple starting point for
a fixed-composition cluster search.

## Configuration

Place this fragment under `recipe.operators` for MC or basin hopping,
or under top-level `operators` for hybrid MC:

```yaml
- method: move
  particles: [Cu]
  max_disp: 0.8
  temperature: 500.0
```

## Settings

| Setting | Meaning | Default |
| --- | --- | --- |
| `particles` | Eligible symbols or molecular formulas | Required |
| `max_disp` | Translation distance in Å | 2.0 |

See {ref}`sampling-operator-shared-settings` for temperature, relative selection
weights, regions, particle tags, and distance checks.

## Proposal and acceptance

The operator selects one eligible tagged particle in the region and draws a
random direction. In the current implementation, its center moves by exactly
`max_disp`; this is not a uniform draw between zero and that distance. Molecular
particles are also randomly rotated. Atoms sharing a tag move as one particle.

The proposal retries placement until distance checks pass or
`max_random_attempts` is exhausted. Intramolecular pairs are excluded from
these checks. No eligible particle or exhausted attempts produces an invalid
proposal. `skip_distance_check: true` bypasses the geometric check.

Composition is unchanged. After runtime evaluation, acceptance uses the energy
change and `temperature`. Increase `max_disp` to explore farther from the current
structure; smaller values make more local proposals.

See {ref}`bh-cu8-example` for a complete EMT run.
