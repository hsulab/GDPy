(sampling-operator-cavity-exchange)=

# `cavity_exchange`

Screen several trial positions for atomic insertion and account for their
cavity count in exchange acceptance.

## Configuration

Place this fragment under `strategy.operators`. Preset MC and hybrid MC read
temperature and chemical potentials from `system.ensemble`; custom MC/HMC and
basin hopping add them to the operator:

```yaml
- method: cavity_exchange
  particles: [Ni]
  num_trials: 50
  cavity_distance: [2.0, null]
  region:
    method: sphere
    origin: [10.0, 10.0, 10.0]
    radius: 3.0
```

## Settings

| Setting | Meaning | Default |
| --- | --- | --- |
| `particles` | One species per operator | Required |
| `chempots` | Chemical potential for custom MC/HMC and BH; preset MC/HMC uses the ensemble mapping | Required outside preset MC/HMC |
| `use_ads` | Build the particle with the adsorbate representation | `false` |
| `num_trials` | Number of trial points; supply a positive integer | Required |
| `cavity_distance` | Absolute `[minimum, maximum]` distances in Å; `null` maximum disables the isolation check | Uses `covalent_ratio` |

See {ref}`sampling-operator-shared-settings` for temperature, relative selection
weights, regions, particle tags, and distance checks.

## Proposal and acceptance

Only atomic chemical symbols are supported. Insertion samples `num_trials`
positions and screens their distances to existing atoms, ignoring interactions
between trial points. With qualifying cavities, the first qualifying point is
used and the proposal records a factor `num_trials / num_cavities`.

If no cavity qualifies, the current implementation uses the last sampled point
with a factor of one. Thus cavity screening does not guarantee that every
submitted insertion passes the cavity criterion. Choose a runtime that can
handle the resulting trial structures.

Removal also evaluates trial cavities, including the removed atom's position,
to obtain its corresponding factor. The acceptance rule combines this factor
with the full region volume and the usual exchange energy, chemical-potential,
particle-count, and temperature terms.

Without `cavity_distance`, screening uses the covalent-distance ratios and
requires a nearby atom. The example specifies only a minimum distance, allowing
isolated insertion sites. This operator uses its own screening rather than the
ordinary exchange retry loop.
