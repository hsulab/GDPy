(sampling-operator-bond-break)=

# `bond_break`

Break one bond by moving a single selected atom:

```yaml
- method: bond_break
  particles: [H]
  bond: [O, H]
  cutoffs: [0.8, 1.2]
  covalent_ratio: [0.8, 1.2]
```

`particles` lists eligible element symbols. Each proposal moves exactly one
atom; molecular formulas such as `H2O`, `HO` or `H2` are not accepted.
For HO–H, selecting `[H]` moves one H and leaves HO stationary. No additional
moving-fragment argument is needed.

The selected atom must have exactly one connected neighbor: the atom at the
other end of the bond. Breaking must leave the moving atom isolated. An atom
with additional bonds is skipped, so the operator never carries a fragment
or stretches its remaining bonds. Fixed atoms and substrate atoms cannot move.
If both endpoints are eligible terminal atoms, either orientation is equally
likely; only one atom moves.

This operator only separates the selected atom. Use {doc}`bond_form` for a
later attachment. See {doc}`bond_change` for distance rules, geometry functions,
constraints, history and checkpoint handling.
