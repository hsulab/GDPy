(bh-operator-exchange)=

# `exchange`

Insert or remove one particle to search across compositions.

## Configuration

This is an operator fragment to place in an exploration recipe:

```yaml
recipe:
  operators:
    - method: exchange
      particles: [Ni]
      chempots: [-0.5]
      temperature: 1000.0
      region:
        method: sphere
        origin: [10.0, 10.0, 10.0]
        radius: 3.0
```

## Settings

| Setting | Meaning | Default |
| --- | --- | --- |
| `particles` | One species per operator | Required |
| `chempots` | One chemical potential in eV per particle | Required |
| `use_ads` | Build the particle with the adsorbate representation | `false` |

See {ref}`bh-operator-shared-settings` for temperature, relative selection
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
The volume is the full geometric region volume.

Use a separate operator entry for each exchangeable species. Builder composition
ranges apply only to initialization and do not constrain later exchange moves.
The runtime must support every species that can be inserted, and
formation-energy ranking should use consistent chemical potentials.

See {ref}`bh-variable-composition-example` for a complete Cu₆Niₓ search with
these illustrative chemical-potential settings.
