# Structure comparison

`compare` retains a candidate when its configured comparator considers it
dissimilar to every previously retained structure. By default it visits every
candidate in input order. Set `fmax` to filter force-converged candidates,
`rank_by: energy` to visit them in ascending total energy, and `number` to stop
after a fixed number of distinct structures. The comparator is explicit:

```yaml
method: compare
number: 8
fmax: 0.05
rank_by: energy
random_seed: 0
comparator:
  method: displacement
  group_expression: "`id 257:296`"
  rmsd_max: 0.5
  permutation_invariant: true
  mic: false
```

For Cu08Pt24/O08, the one-based inclusive IDs 257–296 select exactly Cu8Pt24O8
and exclude the MgO support. A symbol-only oxygen selection also includes
support oxygen. Other datasets can use tags, symbols, regions, or combinations
from {doc}`../groups-and-constraints`. Omit `group_expression` to compare all
atoms. Empty selections and incompatible selected compositions are rejected.

## Atomic displacement

`displacement` compares root-mean-square atomic displacement in the input
coordinate frame. Similarity means RMS displacement strictly below
`rmsd_max`, in Angstrom. No translation or rotation fitting is performed.

`permutation_invariant: false` (the default) pairs atoms by their order within
the selected group and requires the same species at corresponding indices.
With `permutation_invariant: true`, Hungarian assignment minimizes total
squared displacement independently within each species. Atoms of different
elements are never exchanged. This is an exact assignment for the supplied
coordinate frame; atom ordering is preserved during group preparation.

`mic: true` uses the minimum-image distance for each candidate pair before
assignment, respecting the input structures' periodic directions. MIC requires
matching periodic boundary conditions and, when periodic, matching cells with
valid periodic vectors. The default `mic: false` measures ordinary Cartesian
displacements. Global optimisation supplies `mic` from `system.periodic`.

Use fixed-index comparison for structures whose atom order represents atom
identity, and enable permutation invariance when order may differ between
structures. Translation and orientation differences remain part of the
comparison, including movement relative to a support.

## Oganov fingerprints

An alternative comparator measures weighted cosine distance between
species-resolved radial environments:

```yaml
method: compare
number: 8
fmax: 0.05
rank_by: energy
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
These illustrative fingerprint settings use a different similarity criterion
from the displacement example above.

Structural comparators expose `prepare_data(structures)`,
`compare_fingerprints(first, second)`, and `fingerprint_distance(first, second)`.
Energy and combined comparators provide preparation and comparison without
requiring a structural distance metric.
Preparation computes fresh, operation-owned representations without using or
changing fingerprints stored in input metadata. `oganov_fingerprint` and
`displacement` compare geometry alone. The
independent `energy` comparator uses total potential energies. Use `combined` or
the `energetic_oganov_fingerprint` preset to require both energy and structural
similarity. Prepared comparisons preserve every composed criterion, so
comparison treats geometrically similar candidates as distinct when their
energy difference meets or exceeds the configured energy threshold.

## Convergence and outputs

The maximum atomic force **vector norm**, after applying constraints, must be
at most `fmax`. Energies and force checks include the complete structure even
when the comparator selects a group. This is a relaxation convergence check;
no Hessian calculation is performed. Comparison comparators must implement
`prepare_data` and `compare_fingerprints`.

When configured, energy ranking and force filtering require stored finite
values. The selector raises an error if fewer than `number` distinct eligible
candidates are available. Output retains input order even when comparison uses
energy order. Use a fresh output directory when changing inputs or selection
parameters because existing selection caches are reused.

## Migration

The former `minima` selector is folded into `compare`. Replace
`method: minima` with `method: compare` and add `rank_by: energy`; `number`,
`fmax`, and `comparator` retain their meanings. Omit `rank_by`, `fmax`, or
`number` independently for input-order comparison, no force filter, or no
limit. Comparator configurations use `method`, not the former `name` field.

Replace selector-level `n_top`, `ofp`, and `rmsd` with an explicit `comparator`
mapping. Use `group_expression` in place of `n_top`, and `rmsd_max` in place of
`rmsd`. The former `ofp` method is now `oganov_fingerprint`; its implementation
is `gdpx.analysis.comparators.oganov_fingerprint.OganovFingerprintComparator`.
The old population import path is removed. Fingerprints are prepared afresh,
so the former metadata-cache `recalculate` option is also removed.

The former Oganov `dE` energy gate is removed. Request it explicitly with
`comparator.method: energetic_oganov_fingerprint`, `ene_diff_tol`, and the
fingerprint parameters in the same mapping; see {doc}`../global_optimisation/operators/comparators/index`.

`cartesian_coordinate` and `permutation_rmsd` are replaced by `displacement`,
without compatibility presets. Replace `group` with `group_expression` and set
`permutation_invariant` and `mic` explicitly for the intended comparison.
`rmsd_max` is the RMS displacement threshold, replacing the Cartesian
comparator's separate `dtol_avg` and `dtol_std` criteria; their numeric values
are not interchangeable. The former permutation RMSD performed rotation and
translation fitting; `displacement` retains the input coordinate frame and
does not accept `align`.
