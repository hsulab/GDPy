(ga-comparators)=

# comparators

Comparators decide whether two relaxed candidates represent the same minimum.
This prevents duplicate structures from dominating the population.

| Method | Description |
| --- | --- |
| `interatomic_distance` | Compares energy and sorted interatomic-distance fingerprints. `pair_cor_cum_diff` and `pair_cor_max` control the cumulative and maximum fingerprint differences; `dE` controls the energy tolerance. |
| `oganov_fingerprint` | Compares species-resolved radial environments using weighted cosine distance, independently of energy. |
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
atoms; omit it to compare all atoms. Energy differences are not considered.
Population derives its periodicity from `system.periodic`; do not set `pbc`
in this mapping or in nested structural comparators.
The former `ofp` name and `n_top` parameter are removed.

`permutation_rmsd` is also available through the analysis registry. It compares
contiguous clusters with species-preserving fitted RMSD and supports
`group_expression` and `rmsd_max` (Angstrom). This comparator uses nonperiodic
cluster geometry and has no energy gate.

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
To combine energy with RMSD, put `method: permutation_rmsd` as the second
criterion in the general `combined` definition.

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
