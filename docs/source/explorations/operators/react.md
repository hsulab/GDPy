(sampling-operator-react)=

# `react`

Replace reactant particles with product particles according to a configured
stoichiometric reaction, allowing both forward and reverse proposals.

## Configuration

Place this fragment under `recipe.operators` for MC or `strategy.operators` for basin hopping,
or under top-level `operators` for hybrid MC:

```yaml
- method: react
  reaction:
    particles: [H2, O2, H2O]
    chempot_0: [0.0, 0.0, 0.0]
    coefficients: [-2, -1, 2]
  temperature: 1000.0
  use_bias: false
  region:
    method: sphere
    origin: [10.0, 10.0, 10.0]
    radius: 3.0
```

## Settings

| Setting | Meaning | Default |
| --- | --- | --- |
| `reaction.particles` | Ordered species participating in the reaction | Required |
| `reaction.chempot_0` | Standard chemical potentials in eV per particle, in the same order | Required |
| `reaction.coefficients` | Signed integer coefficients: negative for reactants, positive for products | Required |
| `region` | Particle selection and acceptance-volume region | Required |
| `temperature` | Acceptance temperature in K | Required |
| `pressure` | Operator pressure setting in bar | 1.0 |
| `use_bias` | Use estimated empty volume instead of geometric volume | `true` |

See {ref}`sampling-operator-shared-settings` for temperature, relative selection
weights, regions, particle tags, and distance checks.

## Proposal and acceptance

The example expresses `2 H2 + O2 -> 2 H2O`. Keep the three reaction lists
aligned and choose physically appropriate standard chemical potentials; the
zeros above only illustrate the input format. Initial structures need separately
tagged molecular particles and a potential that supports all reaction species.

If only reactant species are present, a forward proposal is attempted; if only
product species are present, a reverse proposal is attempted. If both sides
are present, the direction is chosen with equal probability. A missing complete
side or insufficient particle counts for the chosen stoichiometry makes the
proposal invalid.

Consumed particles are removed. New particles are placed at the first removed
particle's center (or a sampled point if nothing is removed), with rotations
retried to pass distance checks. This is a structure-search proposal, not a
reaction path or a kinetic model. Multiple products can compete for the same
placement center and fail the geometric checks.

Acceptance includes energy, standard chemical-potential changes, stoichiometric
particle-count factors, temperature, and volume. With `use_bias: true`, the
volume is the region's estimated empty volume, as described for
{doc}`biased_volume_exchange`; the example uses geometric volume instead.
The current reaction acceptance calculation uses a fixed standard-pressure
factor, so changing `pressure` does not tune a reservoir pressure in that rule.
