(sampling-operator-bond-change)=

# Bond breaking and formation

Use {doc}`bond_break` to separate one atom and {doc}`bond_form` to attach an
isolated atom to a stationary acceptor. They are separate search operators with
independent selection probabilities. Neither falls back to the other operation.
They preserve composition and atom order and target basin-hopping searches;
these biased moves do not implement equilibrium reaction proposal corrections.

```yaml
- method: bond_break
  particles: [H]
  bond: [O, H]
  cutoffs: [0.8, 1.2]
  probability: 0.5
- method: bond_form
  particles: [H]
  bond: [O, H]
  acceptor_group: "`index 45 123`"
  cutoffs: [1.2, 3.5]
  probability: 0.5
```

Place operators under `strategy.operators`. Add temperature for BH acceptance
as needed. Breaking only separates one atom: it does not place that atom
onto an acceptor in the same move. Relaxation and acceptance occur between hops.
A subsequent formation move is possible only if an isolated eligible atom remains.

## Distance rules

Both operators accept eligible moving element symbols (`particles`), an
unordered element pair (`bond: [O, H]` or `bond: O-H`),
an inclusive candidate-distance window (`cutoffs`, default `[0.0, 3.5]` Å), and
`covalent_ratio` (default `[0.8, 1.2]`). Require finite bounds with
`0 <= min_cutoff < max_cutoff` and `0 < lower_ratio < 1 < upper_ratio`.

First find pairs inside `cutoffs`. With `R` equal to their covalent-radius sum:

| Current distance | Eligible operation |
| --- | --- |
| `d < lower_ratio * R` | Neither: too close |
| `lower_ratio * R <= d <= upper_ratio * R` | Breaking only |
| `d > upper_ratio * R` | Formation only |

Connectivity and fragment reconciliation use `upper_ratio * R`. A moving atom
must have exactly one neighbor for breaking and no neighbors for formation.
An atom with additional bonds is skipped; multiatom fragments never move.

Form at `R`; break at `1.25 * upper_ratio * R`. For default O–H settings these
are 0.97 and 1.455 Å. Target distances need not lie in the search window.
Collision checks use the lower ratio for minimum separations to other atoms.
Detached fragments are allowed by default; `allow_isolated: false` enables the
usual neighbor requirement. Shared operator controls remain available.

## Motion, constraints and tags

Breaking translates one atom with an angular perturbation of up to 10 degrees.
Formation places one isolated atom at the formed distance from a stationary
acceptor in a random direction. All other coordinates remain unchanged.
Periodic distances use the minimum image; aliased target distances are rejected.

Tag-0 atoms represent the immovable substrate. Fixed atoms cannot move;
constraints must expose `get_indices()`. Selecting an atom that cannot move
never causes another element or a multiatom component to move instead.

Positive tags identify current fragments, excluding substrate atoms from their
connectivity graph. A surface-bound H keeps its own positive tag. Reconcile
tags after every relaxation, including trials from other operators. Invalid or
rejected proposals restore coordinates, tags, calculator and history exactly.

## Standalone geometry and history

The MC wrappers delegate to functions in `gdpx.structures.geometry.bond_change`:

```python
from gdpx.structures.geometry.bond_change import break_bond, form_bond

trial = atoms.copy()
event, diagnostic = break_bond(
    trial, rng, particles=["H"], bond="O-H",
)
# Separately, on a structure containing a detached H:
event, diagnostic = form_bond(
    detached.copy(), rng, particles=["H"], bond="O-H",
    acceptor_group="`index 45 123`",
)
```

Both functions edit coordinates and tags in place and return an event plus a
diagnostic. Failed attempts or exceptions restore their edits. Success leaves
edits in place; atom order, constraints, calculator and metadata stay unchanged.
The caller chooses whether to keep the trial and where to record its event.
`find_bond_break_candidates` and `find_bond_form_candidates` enumerate candidates
on reconciled tags. Connectivity helpers live in `structures.geometry.fragments`.

Both operators keep the existing history format: `data.bond_change` records
`operation: break` or `form`, the oriented `[stationary, moving]` pair, moved
indices, distance and tag edits. `data.fragment_changes` records actual final
connectivity. Accepted and rejected evaluated trials are both stored.

All bond operators in a search must use the same upper covalent ratio. Mixtures
require atom-order-preserving operators: `bond_break`, `bond_form`, `rotate`,
`move`, `rattle` or `bounce`.

## Migration

`method: bond_change` has been removed. Replace it with explicit `bond_break`
and/or `bond_form` entries. For both operators, `particles` now lists
the elements of the single atom allowed to move. Molecular formulas and the
former `moving_particles` argument are not supported. Start a fresh
search directory. Checkpoints include operator kind, arguments and geometry
version; changing these settings requires a fresh run.

See {ref}`bh-tio2-water-bond-change` for water dissociation and recombination.
