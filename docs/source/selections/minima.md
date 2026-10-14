# Distinct minima

`minima` visits force-converged candidates in ascending total energy and
retains a candidate when its configured comparator considers it dissimilar
to every previously retained structure. The comparator is explicit:

```yaml
method: minima
number: 8
fmax: 0.05
random_seed: 0
comparator:
  method: permutation_rmsd
  group_expression: "`id 257:296`"
  rmsd_max: 0.5
```

For Cu08Pt24/O08, the one-based inclusive IDs 257–296 select exactly Cu8Pt24O8
and exclude the MgO support. A symbol-only oxygen selection also includes
support oxygen. Other datasets can use tags, symbols, regions, or combinations
from {doc}`../groups-and-constraints`. Omit `group_expression` to compare all
atoms. Empty selections and incompatible selected compositions are rejected.

## Permutation-aware RMSD

`permutation_rmsd` removes rigid translation and rotation and matches atoms
only within the same species. Similarity means fitted RMSD strictly below
`rmsd_max`, in Angstrom. The algorithm alternates Hungarian assignment and
Kabsch fitting, starting from the identity and all 24 proper principal-axis
orientations. This multiple-start procedure estimates the best match; it
is not exhaustive permutation enumeration. Compare contiguous, nonperiodic
clusters. Placement relative to an excluded support is not compared.

## Oganov fingerprints

An alternative comparator measures weighted cosine distance between
species-resolved radial environments:

```yaml
method: minima
number: 8
fmax: 0.05
comparator:
  method: oganov_fingerprint
  group_expression: "`id 257:296`"
  pbc: false
  cos_dist_max: 0.005
  rcut: 20.0
  binwidth: 0.05
  sigma: 0.02
  nsigma: 4
```

Similarity means distance strictly below `cos_dist_max`. Calibrate smoothing
and distance thresholds against known similar structures: narrow smoothing
can distinguish small bond-length distortions rather than different structural
families. With `pbc: false`, clusters must be contiguous across cell boundaries.
These illustrative fingerprint settings do not reproduce the RMSD
selection above.

Structural comparators expose `prepare_data(structures)`,
`compare_fingerprints(first, second)`, and `fingerprint_distance(first, second)`.
Energy and combined comparators provide preparation and comparison without
requiring a structural distance metric.
Preparation computes fresh, operation-owned representations without using or
changing fingerprints stored in input metadata. `oganov_fingerprint` and
`permutation_rmsd` compare geometry alone. The
independent `energy` comparator uses total potential energies. Use `combined` or
the `energetic_oganov_fingerprint` preset to require both energy and structural
similarity. Prepared comparisons preserve every composed criterion, so minima
selection treats geometrically similar candidates as distinct when their
energy difference meets or exceeds the configured energy threshold.

## Convergence and outputs

The maximum atomic force **vector norm**, after applying constraints, must be
at most `fmax`. Energies and force checks include the complete structure even
when the comparator selects a group. This is a relaxation convergence check;
no Hessian calculation is performed. Minima comparators must implement
`prepare_data` and `compare_fingerprints`.

The selector requires stored finite energies and forces and raises an error
if fewer than `number` distinct converged candidates are available. Output
retains input order even though selection uses energy order. Use a fresh output
directory when changing inputs or selection parameters because existing
selection caches are reused.

## Migration

Replace selector-level `n_top`, `ofp`, and `rmsd` with an explicit `comparator`
mapping. Use `group_expression` in place of `n_top`, and `rmsd_max` in place of
`rmsd`. The former `ofp` method is now `oganov_fingerprint`; its implementation
is `gdpx.analysis.comparators.oganov_fingerprint.OganovFingerprintComparator`.
The old population import path is removed. Fingerprints are prepared afresh,
so the former metadata-cache `recalculate` option is also removed.

The former Oganov `dE` energy gate is removed. Request it explicitly with
`comparator.method: energetic_oganov_fingerprint`, `ene_diff_tol`, and the
fingerprint parameters in the same mapping; see {doc}`../global_optimisation/operators/comparators/index`.
