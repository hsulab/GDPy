(global-optimisation-population)=

# population

The population is the working pool of promising structures used to continue a
global optimisation search. It contains the best distinct, eligible candidates
from the search history, rather than only the results of the latest generation.
Genetic algorithms, basin hopping, and simulated annealing share this model.

The search database records evaluated candidates; the population retains a
limited selection of them. A candidate can remain in the database without being
selected for the population.

## how the population evolves

1. **Initialise:** builders generate the starting structures, which are evaluated
   with the configured runtime.
2. **Retain:** eligible results are ranked by the search objective. Similar
   structures are filtered out, and the best distinct candidates fill the
   population up to its capacity.
3. **Explore:** the chosen strategy selects candidates from the population and
   uses them to produce new structures for evaluation.
4. **Refresh:** new results are considered alongside earlier candidates when
   rebuilding the population. Good structures can survive across generations.

The population supplies candidates; the strategy determines how to select and
use them. See the {doc}`genetic algorithm <genetic-algorithm>`,
{doc}`basin hopping <basin_hopping>`, and
{doc}`simulated annealing <simulated-annealing>` guides for those search cycles.

## population sizes

Three settings under `system` control different parts of the search:

| Setting | Purpose |
| --- | --- |
| `initial.total_size` | Number of structures generated to start the search |
| `retained_size` | Maximum number of distinct candidates kept in the population |
| `generation.total_size` | Amount of new search work launched per generation, interpreted by the chosen strategy |

These are positive integers and need not be equal. `retained_size` defaults to
`generation.total_size`. The actual population can be smaller than its capacity
when too few distinct, eligible candidates are available.

For example, a search can start from 20 structures and retain the best 8 distinct
candidates. Later generations may improve or replace those candidates without
discarding good results just because they came from an earlier generation.

## quality, diversity, and eligibility

Three choices determine which structures are retained:

- **Objective** (`strategy.objective`): defines which candidates are better,
  for example by energy or formation energy.
- **Comparator** (`system.comparator`): decides whether structures are similar,
  preventing duplicates from filling the population. See the
  {doc}`comparator guide <operators/comparators/index>` for available methods.
- **Extinction rules** (`system.thanos`): optionally exclude structures that
  violate physical or geometric requirements. Excluded candidates remain in the
  database but cannot enter the population. See {doc}`population/thanos`.

A larger retained pool leaves room for more distinct structures, but cannot
create diversity when the search has found only similar candidates. The
comparator and extinction rules therefore matter as much as the capacity.

## configuring a search

Define named builders and their initial allocations under `system.builders` and
`system.initial`. Allocation sizes must sum to `initial.total_size`. Set the
population capacity and comparator under `system`; choose the search method
and objective under `strategy`.

The {doc}`global optimisation guide <index>` provides the shared configuration
and run commands. The {doc}`examples <examples/index>` provide complete inputs
for different systems.

```{toctree}
:maxdepth: 1
:hidden:

population/thanos
```
