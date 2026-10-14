(sampling-operator-bond-change)=

# `bond_change`

Change one bond by moving a connected fragment. Atom count, atom order and
chemical elements are preserved. This operator targets basin-hopping searches;
the calculator determines the bonding after relaxation. These biased proposals
do not implement equilibrium reaction proposal corrections.

```yaml
- method: bond_change
  particles: [H2O, HO, H]
  bond: [O, H]
  cutoffs: [0.8, 3.5]
  covalent_ratio: [0.8, 1.2]
  temperature: 500.0
```

Place this under `strategy.operators`. At least one endpoint must belong to a
selected positive-tag fragment in the operator's region. The other endpoint
can belong to any suitable fragment, including the substrate or a partner
outside the region. Formula aliases such as `OH` and `HO` are normalized to
Hill notation. Filters refer to current fragments, not original molecules.

| Setting | Meaning | Default |
| --- | --- | --- |
| `particles` | Eligible fragment formulas | Required |
| `bond` | One unordered element pair, `[O, H]` or `O-H` | Required |
| `cutoffs` | Inclusive candidate-distance window `[min, max]`, in Å | `[0.0, 3.5]` |
| `covalent_ratio` | Relative overlap and connectivity bounds `[lower, upper]` | `[0.8, 1.2]` |

`cutoffs` must contain two finite numbers with `0 <= min < max`.
`covalent_ratio` must satisfy `0 < lower < 1 < upper`. Shared MC settings such
as temperature, probability, region and collision checks remain available.
Temperature affects acceptance, not the geometry functions.

## Finding and classifying pairs

First find pairs of the selected elements whose current minimum-image distance
lies inside `cutoffs`. Then let `R` be their covalent-radius sum:

| Current pair distance | Classification |
| --- | --- |
| `d < lower * R` | Too close; exclude |
| `lower * R <= d <= upper * R` | Bonded; consider breaking |
| `d > upper * R` | Unbonded; consider formation |

Topology determines whether the classified operation is possible. Breaking
requires a bridge: removing the bond must disconnect its endpoints. Formation
requires separate connected components. Ring bonds, alternative paths through
the substrate and joins inside one component are skipped. The upper ratio also
controls connectivity and fragment reconciliation throughout the BH search.

The formed distance is `R`; the broken distance is `1.25 * upper * R`. For O–H
with default ratios, these are 0.97 and 1.455 Å. These proposed distances need
not lie inside `cutoffs`, which only filters the starting pair. To restrict a
search to currently bonded or unbonded pairs, set the search window accordingly.

## Motion and fixed components

Breaking translates the moving component with an angular perturbation of up
to 10 degrees. Formation rigidly rotates and translates it in a random direction.
Fragment internal geometry is preserved. Compact fragments crossing periodic
boundaries use minimum-image geometry; extended periodic networks are outside
this implementation. Target distances that alias in a small cell are rejected.

All tag-0 atoms represent one immovable substrate. Components containing any
substrate or fixed atom cannot move. Constraints must expose `get_indices()`.
When both components can move, either orientation is equally likely. Breaking
and formation are equally likely when both have candidates; otherwise the
available operation is used.

Collision checks use the lower covalent ratio for minimum separations to other
atoms. Detached fragments are allowed by default, including the intended broken
bond; `allow_isolated: false` enables the usual neighbor requirement. Geometric
retries start from the original coordinates. Exhausted attempts are invalid.

## Geometry functions

The MC operator wraps `gdpx.structures.geometry.bond_change` functions:

```python
from gdpx.structures.geometry.bond_change import change_bond

trial = atoms.copy()
event, diagnostic = change_bond(
    trial, rng, particles=["H2O"], bond="O-H",
    cutoffs=[0.8, 1.1], covalent_ratio=[0.8, 1.2],
)
if event is not None:
    trial.info["bond_change"] = event
```

`change_bond` edits coordinates and fragment tags in place. Failure or exceptions
restore those arrays. Success returns the operation, selected pair, moved
indices, tag changes and proposed distance. Atom order, constraints, calculator
and metadata remain unchanged. The caller decides whether to keep the edits
and where to store the event.

`find_bond_change_candidates` enumerates possible operations on reconciled tags;
`bond_change_settings` validates and normalizes the four domain arguments.
Connectivity lives in `gdpx.structures.geometry.fragments`. The MC wrapper owns
region selection, transactions and proposal history.

## Tags, history and recovery

Positive tags identify current connected components after excluding substrate
atoms from the fragment graph. Substrate atoms retain tag 0. A surface-bound H
keeps its own positive tag. Input atoms must carry nonnegative tags.

On breaking, the stationary fragment retains its tag and the separated fragment
gets an unused tag. On formation, the stationary fragment's tag survives.
Relaxation can reverse a proposal or transfer atoms between fragments, so BH
reconciles tags after every relaxation, including trials from other operators.
Largest atom overlap retains existing labels, with deterministic tie breaking.
Invalid or rejected proposals restore coordinates, tags and history exactly.

`candidates.db` stores proposed operations in `data.bond_change` and actual final
connectivity in `data.fragment_changes`. Accepted and rejected evaluated trials
are both recorded. All bond-change operators in one search must use the same
upper covalent ratio. Mixtures require atom-order-preserving operators:
`bond_change`, `rotate`, `move`, `rattle` or `bounce`.

Presets and the former detailed bond-specific arguments are no longer accepted.
Replace them with `particles`, `bond`, `cutoffs` and `covalent_ratio`. Start a
fresh BH directory: old checkpoints are incompatible with this classification
algorithm. New checkpoints record normalized arguments and the geometry
algorithm version; changed settings require a fresh search.

See {ref}`bh-tio2-water-bond-change` for water dissociation and recombination.
