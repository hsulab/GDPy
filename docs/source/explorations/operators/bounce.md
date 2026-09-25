(sampling-operator-bounce)=

# `bounce`

Bias an atomic displacement along a chosen Cartesian axis and push nearby
atoms away when the displacement brings them too close.

## Configuration

Place this fragment under `strategy.operators`. Preset MC and hybrid MC set
temperature under `system.ensemble`; custom MC/HMC and basin hopping add it to
the operator:

```yaml
- method: bounce
  particles: [Cu]
  direction: +z
  bias_ratio: 0.8
  max_disp: 0.8
  repulsion_strength: 1.0
```

## Settings

| Setting | Meaning | Default |
| --- | --- | --- |
| `particles` | Eligible single-atom species | Required |
| `direction` | One of `+x`, `-x`, `+y`, `-y`, `+z`, `-z` | Required |
| `bias_ratio` | Directional bias, in `(0, 1]` | 0.8 |
| `max_disp` | Displacement scale in Å | 2.0 |
| `repulsion_strength` | Scale applied to neighboring overlap corrections | 1.0 |

See {ref}`sampling-operator-shared-settings` for temperature, relative selection
weights, regions, particle tags, and distance checks.

## Proposal and acceptance

One eligible atom is selected. Its displacement combines a component along
`direction` with a random perpendicular component according to `bias_ratio`.
Neighbors closer than the lower covalent-distance threshold after this move
are displaced away by the shortfall multiplied by `repulsion_strength`.
Consequently, atoms outside the selected `particles` list can also move.

This is a local neighbor correction, not a guarantee that all overlaps in the
resulting structure have been removed. Only single-atom particles are supported;
no eligible atom gives an invalid proposal.

Composition is unchanged and acceptance uses energy and temperature. The
axis bias makes this useful for directed search proposals; it does not make
a trajectory an equilibrium sample.
