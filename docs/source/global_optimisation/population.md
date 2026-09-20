(global-optimisation-population)=

# Population

Basin hopping (BH) and genetic algorithms (GA) share the same definition of a
population: the best distinct, eligible candidates retained from the search
history for parent selection. The comparator determines whether candidates are
similar; extinction rules exclude candidates that are no longer eligible.

## Three sizes

| Setting | Meaning |
| --- | --- |
| `population.initial.total_size` | Number of candidates generated for initialization |
| `population.retained_size` | Maximum number of distinct candidates retained for parent selection |
| `population.generation.total_size` | Number of new candidates per generation; independent chains for BH |

All three sizes must be positive integers, but need not be equal or ordered.
`retained_size` defaults to `generation.total_size`. The retained pool may be
smaller than its capacity if too few distinct candidates are available.

BH samples chain starts with replacement, weighted by fitness. Multiple chains
can therefore start from the same retained candidate, and each chain evolves
independently. An empty surviving pool ends the search as extinct. GA uses its
own reproduction and mutation policies to produce the requested generation.

## Builders and initialization

Both methods use named `builders` and exact `initial.builder_allocations`:

```yaml
population:
  retained_size: 2
  periodic: false
  builders:
    random:
      method: random_structure_improved
      composition: {Cu: 8}
      box: [12.0, 12.0, 12.0]
      region:
        method: sphere
        origin: [6.0, 6.0, 6.0]
        radius: 3.0
  initial:
    total_size: 4
    builder_allocations:
      - builder: random
        size: 4
  generation:
    total_size: 2
  comparator:
    method: interatomic_distance
```

Allocation sizes must sum to `initial.total_size`. Each allocation can set
`maximum_attempts`; the default is ten times its requested size. Exhausting the
attempt limit without generating enough valid structures raises an error.
`reference_builder` defaults to `random` and identifies the builder supplying
system metadata to the search operators.

`periodic` defaults to `true` and controls builder and comparator periodicity.
Set it to `false` for gas-phase clusters; do not set `pbc` inside builders or
`pbc`/`mic` inside the comparator. Molecular tags remain available for BH moves;
GA's `preserve_fragments` setting controls its genetic operators.

## Comparison and selection

`population.comparator.method` defaults to `interatomic_distance`. Other search
comparators are `ofp`, `nnmat`, and `atoms` (exact ASE Atoms equality). Existing
analysis comparator methods are also available.

Candidates are ranked by descending objective score and duplicates removed.
Fitness includes similarity counts across eligible search history. GA additionally
uses pairing participation; BH does not. Equal scores receive equal base fitness
before history weighting. Optional `population.thanos` callbacks apply extinction
rules in both methods.

GA keeps `generation.reproduction`, `generation.mutation`, and
`generation.completion`. BH requires only `generation.total_size`; its move
operators and `num_mcmoves` define how each chain produces a candidate.

## Migrating existing input

| Previous setting | Replacement |
| --- | --- |
| BH `population.initial_size` | `population.initial.total_size` |
| BH `population.population_size` | `population.retained_size` |
| BH `population.generation_size` | `population.generation.total_size` |
| BH `population.random_offspring_generator` or recipe `builder` | `population.builders` and `initial.builder_allocations` |
| GA `operators.comparator` or `operators.mobile.comparator` | `population.comparator` |
| BH comparator `name` | comparator `method` |

Old YAML fields raise migration errors rather than silently changing meaning.
Use `comparator: {method: atoms}` to retain BH's former exact-equality comparison.
BH now launches the exact requested number of chains even when its retained pool
is underfilled. Named random streams and the changed selection policy mean that
migrated BH runs need not reproduce old trajectories bit for bit.
