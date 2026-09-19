(ga-operations)=

# Operations

GA operations control how GDPy recognises duplicate structures, combines
parents, and modifies offspring. They are configured in three categories:
`comparator`, `crossover`, and `mutation`.

```yaml
operators:
  comparator:
    method: interatomic_distance
  crossover:
    method: periodic_cut_and_splice
  mutation:
    - method: rattle
      probability: 1.0
    - method: cluster_rotation
      probability: 0.5
```

When several mutations are configured, `probability` gives their relative selection
weights. Builder-derived values such as minimum bond distances, the substrate,
and the number of optimised atoms are supplied to compatible operations
automatically. Periodicity and fragment preservation are configured once as
`population.periodic` and `population.preserve_fragments`.

## Comparators

Comparators decide whether two relaxed candidates represent the same minimum.
This prevents duplicate structures from dominating the population.

| Method | Implementation | Description |
| --- | --- | --- |
| `interatomic_distance` | GDPy | Compares energy and sorted interatomic-distance fingerprints. `pair_cor_cum_diff` and `pair_cor_max` control the cumulative and maximum fingerprint differences; `dE` controls the energy tolerance. |
| `ofp` | GDPy | Uses Oganov fingerprints and an energy threshold. It is useful when radial environments provide a better similarity measure than direct distance-list comparison. |
| `nnmat` | GDPy | Compares nearest-neighbour matrices to detect differences in atomic distribution and structure. |

`interatomic_distance` derives its minimum-image behavior from
`population.periodic` and supports parallel fingerprint generation with
`n_jobs`. Its default thresholds are
`pair_cor_cum_diff: 0.015`, `pair_cor_max: 0.7`, and `dE: 0.02` eV.

## Crossovers

Crossovers create an offspring from two selected parents.

| Method | Implementation | Description |
| --- | --- | --- |
| `periodic_cut_and_splice` | GDPy | Divides two parents with a random plane and joins material from opposite sides. It supports fixed or variable cells and can preserve tagged molecular fragments. |
| `cluster_cut_and_splice` | GDPy | Applies cut-and-splice crossover to isolated particles or clusters. It preserves composition by default and separates halves when atoms would otherwise be too close. |

Use `periodic_cut_and_splice` for supported structures and periodic systems. Use
`cluster_cut_and_splice` for free clusters where there is no substrate.

## Mutations

Mutations introduce variation into a single candidate. GDPy checks the
resulting geometry where supported and discards an operation when it cannot
produce a valid structure.

| Method | Implementation | Description |
| --- | --- | --- |
| `mirror` | GDPy | Keeps one side of a randomly oriented cutting plane and mirrors it to replace the other side. This currently supports atomic structures only. |
| `rattle` | GDPy | Randomly displaces a fraction of the optimised atoms or tagged fragments while enforcing minimum distances. `rattle_prop` selects the fraction and `rattle_strength` controls displacement. |
| `soft` | GDPy | Displaces the structure along a smooth low-frequency mode generated from its local geometry. |
| `strain` | GDPy | Applies a random strain to the cell. It is intended for variable-cell searches and respects configured cell bounds. |
| `bounce` | GDPy | Selects a tagged atom and moves it using neighbour repulsion. The move can be directionally biased and can target either mobile particles or a selected buffer group. |
| `cluster_rattle` | GDPy | Finds connected clusters using the atomic graph and translates selected clusters as rigid units in random directions. |
| `cluster_rotation` | GDPy | Finds graph-connected clusters and rotates selected clusters as rigid units about a fixed or random axis. |
| `exchange` | GDPy | Inserts or removes atoms or molecular fragments. It supports composition bounds, spatial regions, and predefined adsorption sites. |
| `group_rattle` | GDPy | Selects atoms from a group expression and randomly displaces either a fixed number or a fraction of them, rejecting overlaps. |
| `swap` | GDPy | Exchanges the positions of different tagged particle types while preserving the internal geometry of molecular fragments. |

### Geometry-aware mutations

The GDPy-native `bounce`, `cluster_rattle`, `cluster_rotation`, `exchange`, and
`swap` mutations use the builder's bond-distance information. Their
`covalent_ratio` defines both lower and upper acceptable distance ratios, so
they can reject candidates that are either overlapping or disconnected.

### Tags and molecular fragments

GDPy uses ASE tags internally to distinguish the substrate and individual
particles. A substrate has tag 0; generated atoms or molecular fragments have
positive tags. With `population.preserve_fragments: true`, atoms sharing a
positive tag must be treated as one particle and their internal geometry must
be preserved. GDPy rejects an incompatible crossover or mutation while loading
the configuration.

The following restrictions apply:

- `bounce` and `swap` require tagged structures.
- `swap` requires at least two particle types.
- `periodic_cut_and_splice`, `rattle`, `soft`, `strain`, `exchange`, and `swap`
  support fragment-preserving searches.
- `cluster_cut_and_splice`, `mirror`, `bounce`, `cluster_rattle`,
  `cluster_rotation`, and `group_rattle` do not guarantee preservation of
  configured tag-defined fragments.
- `cluster_rattle` and `cluster_rotation` act on graph-connected groups formed
  from positively tagged atoms.

### Variable composition

`exchange` is the mutation for variable-composition searches. `species`
selects what may be inserted or removed, and `num_min_max` bounds the number of
each species. Insertions can be sampled inside a configured region or placed
at graph-derived adsorption sites using `anchors`.

The standard GA operation interfaces are GDPy-owned implementations inspired
by the algorithms and configuration surface in ASE-GA 1.0.3. They use explicit
NumPy `Generator` streams; GDPy does not import the legacy `ase.ga` package at
runtime.
