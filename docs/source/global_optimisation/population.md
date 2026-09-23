(global-optimisation-population)=

# population

Basin hopping (BH) and genetic algorithms (GA) share the same definition of a
population: the best distinct, eligible candidates retained from the search
history for parent selection. The comparator determines whether candidates are
similar; extinction rules exclude candidates that are no longer eligible.

## Shared search configuration

Both algorithms use `method: global_optimisation`. Search settings are top-level;
there is no `recipe` wrapper. `population` describes initialization and the
retained candidate pool, while `strategy.method` selects `genetic_algorithm` or
`basin_hopping`. `objective`, `convergence`, `random_seed`, and `use_archive` are
shared search settings. Calculation `runtime` and exploration `scheduler` remain
separate top-level execution settings.

GA puts `operators`, `reproduction`, `mutation`, `completion`, and optional
`substrate` compatibility settings inside `strategy`. BH puts `operators`,
`steps_per_chain`, and `selection` there. Switching strategies does not require
changing the population schema. See the complete Cu₈ examples for both methods.

Crossover compatibility follows the operator automatically. Composition-preserving
operators require matching ordered atomic species and tags, plus the configured
substrate tolerance. The first parent is fitness-weighted among candidates with
at least one compatible partner; the second is fitness-weighted among its
compatible partners. Operators explicitly supporting variable composition may
pair different compositions. Missing capability flags default to composition
preservation. If no pair is compatible, GA uses its single-parent mutation
fallback and builder completion. Standalone mutations select from the full pool.
There is no composition-group balancing or population-wide constant/variable mode.

## Three sizes

| Setting | Meaning |
| --- | --- |
| `population.initial.total_size` | Number of candidates generated for initialization |
| `population.retained_size` | Maximum number of distinct candidates retained for parent selection |
| `population.generation.total_size` | New candidates per generation for GA; independent chains per generation for BH |

All three sizes must be positive integers, but need not be equal or ordered.
`retained_size` defaults to `generation.total_size`. The retained pool may be
smaller than its capacity if too few distinct candidates are available.

BH samples chain starts weighted by fitness, without replacement when enough
candidates are available. Otherwise it samples with replacement. Set
`strategy.selection.replace: true` to always allow replacement. Each chain
evolves independently. An empty surviving pool ends the search as extinct. GA uses its
own reproduction and mutation policies to produce the requested generation.

## Builders and initialization

Both methods use named `builders` and exact `initial.builder_allocations`:

```yaml
population:
  retained_size: 2
  builders:
    random:
      method: random_structure_improved
      composition: {Cu: 8}
      box: [20.0, 20.0, 20.0]
      region:
        method: sphere
        origin: [10.0, 10.0, 10.0]
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

GA uses `strategy.reproduction`, `strategy.mutation`, and
`strategy.completion`. BH requires only `generation.total_size`; its move
operators and `steps_per_chain` define the trials along each chain. Every evaluated
trial minimum enters the database, including rejected trials. A BH generation
therefore adds up to `generation.total_size × steps_per_chain` evaluated candidates;
invalid proposals add none. Population refresh considers all eligible stored
minima at the start of the next generation.

## Migrating existing input

| Previous setting | Replacement |
| --- | --- |
| `method: genetic_algorithm` or `method: basin_hopping` | `method: global_optimisation` and `strategy.method` |
| `recipe` wrapper | Move its shared settings to the top level |
| `recipe.operators` | `strategy.operators` |
| GA `population.generation.reproduction/mutation/completion` | Corresponding sections under `strategy` |
| GA `population.substrate` | `strategy.substrate` |
| GA `population.name` | Remove; crossover compatibility is automatic |
| BH `recipe.num_mcmoves` or `strategy.num_mcmoves` | `strategy.steps_per_chain` |
| BH `recipe.selection` | `strategy.selection` |
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

## Population and algorithm policies

GA and BH inherit `gdpx.exploration.population.PopulationBasedExploration` for
common setup and serialization, and use the same
`gdpx.exploration.population.Population` class. Each
engine exposes the retained state as `engine.population`, while
`engine.population_config` owns configuration, builders, initialization, and
serialization.

`Population.refresh(database)` ranks eligible relaxed candidates, removes
duplicates, and rebuilds similarity counts. `population.candidates` is a tuple
of borrowed `Atoms` references; `population.similarity_counts` stores statistics
by candidate ID. Refresh preserves candidate metadata, including existing
fingerprint caches. The population does not own random streams or generate new
structures.

Selection belongs to each algorithm:

- GA's `GeneticParentSelector` selects one parent or a distinct compatible pair and applies
  pairing-participation penalties. `GeneticGenerationManager` handles
  reproduction, mutation, compatibility checks, and builder completion.
- BH's `HoppingStartSelector` selects chain starts using `strategy.selection.replace`. The BH
  engine runs the moves, relaxation, and acceptance steps independently for
  each selected start.

Selection returns references without copying structures. Each algorithm creates
an independent mutable copy only when starting an offspring or hopping chain.
The shared population has no GA-specific subclasses, and algorithm selection
statistics are kept outside `Atoms.info`.

## Generation progress and restart

Both engines use `gdpx.exploration.generation.GenerationInfo` and
`GenerationState`. The database determines whether a generation is beginning,
in progress, complete, or the search is extinct. GA and initialization require
fixed counts of committed evaluations. BH hopping generations finalize an
explicit list of evaluated candidate IDs after all rounds; completion requires
all listed results, even when there are more evaluations than chains or no
valid trials at all. An output directory alone does not establish completion. Worker evaluation
uses a separate `EvaluationStatus` (`PENDING` or `FINISHED`).

Generation plans persist construction progress and random-stream states.
Restart with the same recipe and output directory to resume pending evaluations
or partial result ingestion without creating duplicate candidates. Generation
sizes and the extinction policy cannot change on restart.

GA retains its reproduction and mutation plan. BH additionally checkpoints each
completed batch round under `tmp_folder/gen*/rounds/round-*`, including all
accepted structures and random state in JSON and non-pickled NumPy arrays.
Only two committed snapshots are retained, plus the current pending batch.
Pending rounds persist trials before submission and resume through the worker.
A compact event journal references candidate structures in the database; full
snapshots are removed after generation finalization. These checkpoints are
independent of optional XYZ trajectory exports used for inspection.

Legacy BH runs with pending inputs that lack generation metadata cannot be
resumed; start a new run for those inputs.

For BH, an accepted extinct trial also terminates its chain segment. The retained
population is refreshed after the complete round to select a replacement for the
remaining moves. Rejected extinct trials only affect population eligibility.
Replacement choices and all selection RNG streams are included in the round
checkpoint; interruption cannot redraw a committed replacement.

The unified configuration keeps database and checkpoint formats unchanged. Migrate
saved submission inputs as well as user configurations before resuming old runs.
Committed candidates and parent choices are reused. Automatic compatible-parent
selection changes GA sampling and RNG consumption, so future choices may differ
from earlier versions even with the same seed. Within this implementation,
interrupted and uninterrupted runs remain reproducible.
