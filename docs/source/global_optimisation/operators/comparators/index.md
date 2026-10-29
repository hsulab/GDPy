(ga-comparators)=

# comparators

Comparators decide whether two relaxed candidates represent the same minimum.
This prevents duplicate structures from dominating the population.

Structural comparison in global optimisation uses atoms with **tag != 0** by
default. Tag 0 identifies substrate atoms, regardless of their order in the
structure. Set `group_expression` to compare a subset of the mobile atoms:

```yaml
system:
  comparator:
    method: interatomic_distance
    group_expression: "`symbol O`"
    dE: 0.02
```

The expression is evaluated on the original structure, then intersected with
`tag != 0`; index/ID selectors therefore refer to original atom indices. An
empty intersection raises an error. Omit `group_expression` to compare all
mobile atoms. `n_top` is no longer accepted in population comparator inputs.
Energy gates always use the complete candidate energy. In a `combined`
comparator, set groups on individual structural children, not on the energy
criterion or the combination itself. These tag restrictions apply to global
optimisation; standalone analysis comparators retain their own group defaults.

| Method | Description |
| --- | --- |
| `interatomic_distance` | Compares energy and sorted interatomic-distance fingerprints. `pair_cor_cum_diff` and `pair_cor_max` control the cumulative and maximum fingerprint differences; `dE` controls the energy tolerance. |
| `oganov_fingerprint` | Compares species-resolved radial environments using weighted cosine distance, independently of energy. |
| `displacement` | Compares RMS atomic displacement, optionally matching same-species atoms and using MIC, without alignment. |
| `energy` | Compares total potential energies; similarity means absolute difference strictly below `ene_diff_tol` (eV). |
| `combined` | Requires every comparator in its `comparators` list to consider the candidates similar. |
| `energetic_oganov_fingerprint` | Preset expanding flat energy and Oganov fingerprint parameters into a combined comparator. |
| `nnmat` | Compares nearest-neighbour matrices to detect differences in atomic distribution and structure. |

`interatomic_distance` derives its minimum-image behavior from
`system.periodic` and supports parallel fingerprint generation with
`n_jobs`. Its default thresholds are
`pair_cor_cum_diff: 0.015`, `pair_cor_max: 0.7`, and `dE: 0.02` eV.

`oganov_fingerprint` is implemented in the analysis comparator package as
`OganovFingerprintComparator`. Use `group_expression` to select comparison
atoms; omit it to compare all mobile atoms. Energy differences are not considered.
Population derives its periodicity from `system.periodic`; do not set `pbc`
in this mapping or in nested structural comparators.
The former `ofp` name and `n_top` parameter are removed.

`displacement` compares RMS atomic displacement in the input coordinate frame.
It supports `group_expression`, `rmsd_max` (Angstrom), and
`permutation_invariant` (default `false`). When enabled, same-species assignment
minimizes the total squared displacement. Periodic comparison applies MIC
before assignment, using the structures' actual periodic directions.
Population supplies `mic` from `system.periodic`; do not set it in the
comparator mapping. No alignment is performed. This replaces
`cartesian_coordinate` and `permutation_rmsd` without presets.

## Combining energy and structure

`CombinedComparator` accepts a nonempty list of independent comparators and
requires every criterion to consider a pair similar (AND semantics):

```yaml
system:
  periodic: false
  comparator:
    method: combined
    comparators:
      - method: energy
        ene_diff_tol: 1.0
      - method: oganov_fingerprint
        group_expression: "`id 257:296`"
        cos_dist_max: 0.005
```

The preset expands to the same `CombinedComparator`, using a flat union of
both criteria's parameters:

```yaml
system:
  periodic: false
  comparator:
    method: energetic_oganov_fingerprint
    ene_diff_tol: 1.0
    group_expression: "`id 257:296`"
    cos_dist_max: 0.005
```

The preset accepts `ene_diff_tol` plus the Oganov parameters `group_expression`,
`cos_dist_max`, `rcut`, `binwidth`, `sigma`, `nsigma`, `pbc`, and `maxdims`, using
the standalone comparators' defaults. In global optimisation, `pbc` is supplied
by `system.periodic` and must not be specified in the comparator mapping.
Component options such as `random_seed` and `n_jobs` are forwarded to both
children and the combined comparator.

Group selection affects only fingerprints. Energy always uses the complete
input's total potential energy. The IDs above are specific to Cu08Pt24/O08.
To combine energy with displacement, put `method: displacement` as the second
criterion in the general `combined` definition and set `permutation_invariant`
to the intended matching policy.

In Python, construct `CombinedComparator(comparators=[...])` from
`gdpx.analysis.comparators.combined`, or pass the comparator list directly to
`create_comparator`. Both factories also accept a list as shorthand for a
combined definition. Lists can contain comparator configurations or instances,
and combined definitions can be nested.

Direct comparisons short-circuit in configuration order; putting energy first
avoids structural comparisons when energy rejects the pair. Prepared
comparisons retain all criteria, including energy. Energy comparison requires
finite available potential energies and treats the threshold boundary as
dissimilar. Geometry-only comparison does not require energies.

The former `dE` parameter on `oganov_fingerprint` is removed. To preserve its
old combined behavior, use `energetic_oganov_fingerprint` and set `ene_diff_tol`
to the old `dE` value. The legacy `interatomic_distance` comparator is unchanged.
