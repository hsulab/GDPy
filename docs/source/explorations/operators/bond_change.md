(sampling-operator-bond-change)=

# `bond_change`

Propose bond breaking or formation by translating or rotating a connected
fragment. This geometry operator preserves atom count, atom order and chemical
elements. It is intended for basin-hopping searches of molecular and dissociated
minima; it does not implement equilibrium reaction proposal corrections.
The calculator evaluates coordinates and determines the relaxed bonding.

```yaml
- method: bond_change
  pairs: [[O, H]]
  mode: both
  form_distance: 1.0
  break_distance: 2.5
  bond_cutoffs:
    O-H: 1.25
  allow_isolated: true
  temperature: 500.0
```

Place this under `strategy.operators`. Element pairs are unordered and can
include pairs other than O–H. `bond_ratio` times the sum of covalent radii
defines connectivity unless `bond_cutoffs` overrides that pair. These are
geometric classifications, not ReaxFF bond orders. They are separate from
the ordinary `covalent_ratio` collision checks.

| Setting | Meaning | Default |
| --- | --- | --- |
| `pairs` | Nonempty list of eligible element pairs | Required |
| `mode` | `break`, `form`, or `both` | `both` |
| `form_distance` | Proposed formed-bond distance, below each eligible pair's bond cutoff when formation is enabled | 1.0 Å |
| `break_distance` | Proposed broken-bond distance, above each eligible pair's bond cutoff when breaking is enabled | 2.5 Å |
| `max_pair_distance` | Largest initial distance for a formation pair | 3.5 Å |
| `break_angle` | Maximum angular deviation from the original bond direction, in `(0, 180]` | 10° |
| `bond_ratio` | Covalent-radius multiplier for connectivity | 1.2 |
| `bond_cutoffs` | Positive pair-specific distances, keyed as `O-H` | `{}` |
| `break_fragments` | Optional source-fragment formulas; either endpoint can match | `null` |
| `form_fragments` | Optional pairs of endpoint-fragment formulas | `null` |

Formula filters use Hill notation internally, so `OH` and `HO` are equivalent
in these filters. `substrate` selects an endpoint with tag 0. Filters match
current fragments rather than original molecule identities. For water on a
surface, `break_fragments: [H2O, H]` permits water O–H breaking and removal of a
surface-bound H; `form_fragments: [[HO, H], [substrate, H]]` permits OH + H
recombination and surface H attachment. Filters do not impose general valence
rules on a force field.

## Geometry and fixed components

Breaking requires a bridge: removing the selected bond must disconnect its
endpoints. Ring bonds and alternative paths through the substrate are skipped.
The moving component is translated to the target distance with a bounded
angular perturbation. Its internal geometry is preserved.

Formation requires two separate connected components. The moving component is
rigidly rotated about its selected endpoint and translated to the target
distance in a random direction. Proposals within an already connected component
are skipped. Minimum-image geometry is used for compact fragments crossing
periodic boundaries; extended periodic networks are outside this version.
Attempts whose target distance aliases to a different minimum-image distance
in a small periodic cell are rejected.

All tag-0 atoms form one immovable substrate body. A component containing any
substrate or fixed atom cannot move. Other constraints must expose atom indices;
constraints without `get_indices()` are rejected. If both components can move,
either is chosen with equal probability. If both modes have eligible pairs,
`both` chooses each mode with equal probability. Otherwise it uses the available
mode. Geometric retries begin from the original coordinates; exhausted attempts
are invalid proposals. Use `allow_isolated: true` when disconnected fragments
are intentionally allowed.

## Fragment tags and history

Positive tags identify current connected components after excluding substrate
atoms from the fragment graph. Substrate atoms always retain tag 0. A
surface-bound H has its own positive tag and is not grouped with the slab.
Atoms must carry nonnegative tags before proposing moves.

On a break, the stationary fragment retains its tag and the moving fragment
gets an unused positive tag. On formation, the stationary fragment's tag
survives. Atom indices identify reaction participants; tags are fragment labels
and are not permanent molecule provenance. Position, tag and history edits are
transactional: invalid proposals and rejection restore the original state.

When BH includes this operator, it reconciles tags after initial relaxation and
every relaxed trial, including trials from other operators. Largest atom overlap
retains existing labels, with deterministic ties and fresh labels for splits.
This captures relaxation-driven recombination and transfer between molecules.
Use Hill formulas such as `HO` when selecting hydroxyls with `rotate` or `move`.

Candidate database `data.bond_change` records the proposed operation, selected
pair, moving atoms, proposed distance and tag edits. `data.fragment_changes`
records final tag edits and bonds involving mobile atoms. Both accepted and
rejected evaluated trials are recorded; the accepted parent keeps its own tags.
Checkpoint recovery requires identical bond-change settings. All bond-change
operators in one search must share their connectivity settings. BH currently
combines them with atom-order-preserving geometry operators (`move`, `rattle`,
`rotate`, and `bounce`).

See {ref}`bh-tio2-water-bond-change` for water dissociation and recombination.
