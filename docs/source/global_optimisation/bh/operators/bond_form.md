(sampling-operator-bond-form)=

# `bond_form`

Move one isolated atom onto a stationary acceptor:

```yaml
- method: bond_form
  particles: [H]
  bond: [O, H]
  acceptor_group: "`index 45 123`"
  cutoffs: [1.2, 3.5]
  covalent_ratio: [0.8, 1.2]
```

`particles` lists eligible element symbols. Each proposal moves exactly one
positive-tag atom in the operator's region. It must be isolated under the
current connectivity policy. A bonded H is skipped, including surface-bound H;
the operator never pulls a whole molecule or fragment along. Molecular formulas
such as `HO` or `H2` are not accepted. The acceptor and every other atom stay
stationary. Acceptors may lie outside the moving atom's region.

`acceptor_group` uses existing atom-group expressions: zero-based `index`,
one-based `id`, `symbol`, `tag`, and their supported combinations. For example,
`` "`symbol O` and `index 45 123`" `` selects those oxygen atoms. Groups are
evaluated on the current reconciled structure. `None` allows all compatible
stationary atoms. An empty selection permits no formation moves; there is no
fallback to another site. Invalid expressions or out-of-range indices raise
an error without retaining trial edits.

The selected bond elements, search window and connectivity still restrict
eligible pairs. The acceptor cannot be the moving atom. This operator never
breaks an existing bond. Use {doc}`bond_break` for atom separation, and
{doc}`bond_change` for shared distance rules, geometry functions and history.
