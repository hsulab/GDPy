(sampling-operator-rotate)=

# `rotate`

Rotate one tagged molecule without translating its pivot. This searches
adsorbate orientations while retaining the molecule's internal geometry.

```yaml
- method: rotate
  particles: [H2O]
  center: O
  max_angle: 75.0
  temperature: 500.0
```

Place this under `strategy.operators`. Preset MC/HMC put temperature under
`system.ensemble`; custom MC/HMC and BH put it on the operator.

| Setting | Meaning | Default |
| --- | --- | --- |
| `particles` | Eligible tagged molecular formulas | Required |
| `max_angle` | Maximum absolute angle in degrees, finite and in `(0, 180]` | `180.0` |
| `center` | `com` (center of mass), `cop` (mean position), or a unique element symbol in the molecule | `com` |
| `axis` | Fixed, nonzero three-vector in Cartesian coordinates; normalized internally | `null` (isotropic random axis) |

The operator selects one eligible molecular tag and samples a signed angle
uniformly from `[-max_angle, max_angle]`. The isotropic axis is a normalized
Gaussian vector. This is a bounded rotation proposal, not a uniform draw over
all possible orientations. Opposite angles about the same axis give inverse
proposals. `center: O` keeps a water's oxygen exactly stationary during the
proposal; relaxation can subsequently move it unless the runtime fixes it.
`axis: [0, 0, 1]` restricts proposals to rotations about the Cartesian z axis.

Atoms sharing a tag form one molecule. Single atoms are skipped. An element
pivot requires exactly one atom of that element in the selected group; groups
with missing or ambiguous anchors are skipped. No eligible molecule gives an
invalid proposal. Compact molecules crossing a periodic boundary are
reconstructed using minimum-image vectors before rotation, retaining each
atom's original lattice image. This assumes the molecule fits within the
minimum-image cell; extended polymers need a connectivity-based unwrap.

Intramolecular distances are preserved. With distance checking enabled,
intermolecular and substrate clashes are checked, and rotations are retried
from the original coordinates up to `max_random_attempts`. Exhausted attempts
are invalid and restore all edited coordinates. `allow_isolated` applies to
the same external-neighbor check as other operators; use `true` when outward
water hydrogens need not each have an external neighbor. Composition, tags,
and constraints remain unchanged. Acceptance uses the ordinary energy change
and temperature. Rejection restores the original coordinates and calculator.

See {ref}`sampling-operator-shared-settings` for regions, weights and checks,
and {ref}`bh-tio2-water-rotation` for anatase (101) examples with two or three
waters and xreac.
