(sampling-operator-biased-volume-exchange)=

# `biased_volume_exchange`

Insert or remove particles using an estimated unoccupied volume in the
acceptance factor.

## Configuration

Place this fragment under `executor.parameters.setup.operators` for MC/HMC
or `strategy.operators` for basin hopping. Preset MC and hybrid MC read
temperature and chemical potentials from `parameters.ensemble`;
basin hopping adds them to the operator:

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
| `chempots` | Chemical potential for BH; preset MC/HMC uses the ensemble mapping | Required outside preset MC/HMC |
| `use_ads` | Build the particle with the adsorbate representation | `false` |

See {ref}`sampling-operator-shared-settings` for temperature, relative selection
weights, regions, particle tags, and distance checks.

## Proposal and acceptance

When eligible particles are present, insertion and removal are selected with
equal probability. With none present, insertion is forced. Removal selects one
eligible tagged particle uniformly. Insertion draws a position from the region
and assigns a new tag; when ordinary distance checks are enabled, it retries
up to `max_random_attempts`. Failed placement gives an invalid proposal.

The acceptance volume is `region.get_empty_volume(atoms)` for the current
structure. Acceptance also includes the energy difference, chemical potential,
particle count, and thermal wavelength. The chemical-potential offset is `-mu`
for insertion and `+mu` for removal. At the zero-particle boundary, the rule
accounts for forced insertion and the half-probability reverse deletion branch.

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

## Equilibrium-sampling limitation

Replacing the geometric volume with an estimated empty volume does not by
itself correct the proposal probability: insertion still uses the ordinary
region generator. Do not assume detailed balance from this acceptance-volume
substitution. Quantitative MC use requires a consistent derivation and
validation of the forward and reverse proposal factors for the chosen region.
