(sampling-operator-biased-volume-exchange)=

# `biased_volume_exchange`

Insert or remove particles using an estimated unoccupied volume in the
acceptance factor.

## Configuration

Place this fragment under `strategy.operators` for MC or basin hopping, or
under top-level `operators` for hybrid MC. Preset MC reads temperature and
chemical potentials from `system.ensemble`; custom MC and the other methods add
them to the operator:

```yaml
- method: biased_volume_exchange
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

Selection, insertion attempts, particle tags, and removal follow
{doc}`exchange`. The difference is the volume used for acceptance: the operator
calls `region.get_empty_volume(atoms)` for the current structure.

The shared region implementation subtracts the sum of covalent-radius sphere
volumes from the region volume for particles whose centers lie inside it. This
is an estimate: overlapping atomic volumes are not merged, and a particle
crossing the region boundary is counted as a whole. Check that the estimated
volume remains positive for the intended density and region.

Insertions still sample the region using its ordinary position generator;
using empty volume in acceptance does not itself restrict sampling to a cavity.
For explicit trial-point screening, see {doc}`cavity_exchange`.

Acceptance otherwise includes the same energy, chemical-potential, particle-count,
and temperature factors as ordinary exchange. Initial builder composition ranges
do not limit subsequent insertions or removals.
