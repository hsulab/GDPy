(sampling-operator-exchange)=

# `exchange`

Insert or remove one particle to search across compositions.

## Configuration

Place this fragment under `strategy.operators` for MC or basin hopping, or
under top-level `operators` for hybrid MC. Preset MC reads temperature and
chemical potentials from `system.ensemble`; custom MC and the other methods add
them to the operator:

```yaml
- method: exchange
  particles: [Ni]
  region:
    method: sphere
    origin: [10.0, 10.0, 10.0]
    radius: 3.0
```

## Settings

| Setting | Meaning | Default |
| --- | --- | --- |
| `particles` | One species per operator | Required |
| `chempots` | Chemical potential for custom MC, BH, and hybrid MC; preset MC uses the ensemble mapping | Required outside preset MC |
| `use_ads` | Build the particle with the adsorbate representation | `false` |

See {ref}`sampling-operator-shared-settings` for temperature, relative selection
weights, regions, particle tags, and distance checks.

## Proposal and acceptance

When eligible particles are present, insertion and removal are chosen with
equal probability. With none present, insertion is selected. Removal deletes
one randomly selected tagged particle; insertion samples positions in the
configured region and checks distances, retrying up to `max_random_attempts`.
Failed insertion gives an invalid proposal. Inserted particles receive a new tag.

Acceptance includes the energy difference, chemical potential, region volume,
particle count, and thermal wavelength at the configured temperature. The
chemical-potential energy offset is `-mu` for insertion and `+mu` for removal.
The volume is the full geometric region volume. At the zero-particle boundary,
acceptance also accounts for forced insertion and the half-probability reverse
deletion branch.

Use a separate operator entry for each exchangeable species. Builder composition
ranges apply only to initialization and do not constrain later exchange moves.
The runtime must support every species that can be inserted, and
formation-energy ranking should use consistent chemical potentials.

See {ref}`bh-variable-composition-example` for a complete Cu₆Niₓ search with
these illustrative chemical-potential settings.
