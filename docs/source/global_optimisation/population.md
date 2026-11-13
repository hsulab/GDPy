(global-optimisation-population)=

# population

The population retains the best distinct, eligible structures from the search
history. All global optimisation methods share the following `system`
configuration:

```{code-block} yaml
:caption: YAML

system:
  periodic: false
  preserve_fragments: false
  retained_size: 8
  initial:
    total_size: 20
    builder_allocations:
      - builder: init
        size: 20
  generation:
    total_size: 4
  builders:
    init:
      method: random_structure_improved
      composition: {Cu: 8}
      box: [20.0, 20.0, 20.0]
      region:
        method: sphere
        origin: [10.0, 10.0, 10.0]
        radius: 3.0
  comparator:
    method: interatomic_distance

# Optional: exclude evaluated structures with Cu–Cu contacts below 1.5 Å.
#   thanos:
#     - name: restraints
#       restraints:
#         - type: contact_count
#           pair: [Cu, Cu]
#           max: 0
#           distance: {min: 0.0, max: 1.5}
#           scope: all_atoms
```

This example starts from 20 Cu₈ clusters, retains up to 8 distinct candidates,
and sets the generation size to 4. Combine this section with
`method: global_optimisation`, a `strategy`, and a calculation `runtime`;
see the {doc}`global optimisation guide <index>` for a complete input.

## system properties

### periodic

`periodic` controls whether builders and population comparisons use periodic
boundary conditions. It defaults to `true`. Set it to `false` for isolated
clusters, as in this example.

### preserve_fragments

`preserve_fragments` describes whether tagged groups of atoms should be treated
as fragments by compatible search operators. It defaults to `true`, which
requires explicit ASE tags: positive tags identify mobile fragments, and tag 0
is reserved for substrate atoms. Give independent atoms distinct positive tags
and atoms in the same fragment a shared tag.

This atomic Cu₈ example sets it to `false` because there are no molecular
fragments to preserve. Fragment handling belongs to the search operators;
this setting does not constrain bonds during relaxation or dynamics.

## population sizes

### retained_size

`retained_size` is the maximum number of distinct, eligible candidates kept
for selection. Here, the population retains up to 8 candidates, ranked by
`strategy.objective`, for example by energy or formation energy.

The population draws from the search history, so good structures can survive
across generations. It can be smaller than its capacity when too few distinct,
eligible candidates are available. Evaluated candidates remain in the database
even when they are not retained.

### initial

`initial.total_size` is the number of starting structures. Each entry in
`initial.builder_allocations` names a `builder` and the number of structures
(`size`) it supplies. Allocation sizes must sum to `total_size`; here, all 20
structures come from `init`. These structures are evaluated before filling the
first retained population.

### generation

`generation.total_size` controls the amount of new search work launched per
generation. The strategy determines how this size is used and how candidates
are selected from the retained population. New results are considered alongside
earlier candidates when the population is refreshed.

The initial, retained, and generation sizes must be positive integers but need
not be equal. If omitted, `retained_size` defaults to `generation.total_size`.
For the search cycles, see {doc}`genetic algorithm <genetic-algorithm>`,
{doc}`basin hopping <basin_hopping>`, or
{doc}`simulated annealing <simulated-annealing>`.

## builders

`builders` defines named sources of initial structures. Here, `init` uses
`method: random_structure_improved` to generate random structures with
`composition: {Cu: 8}`. The `box` is a 20 Å cell in each direction; `region`
places the atoms in a sphere of radius 3 Å centred at `[10, 10, 10]` Å.
See the {doc}`builder guide <../builders/index>` for other structure generators.

## comparator

`comparator` determines whether structures are similar, preventing duplicates
from filling the population. This example selects `interatomic_distance`.
A larger retained pool leaves room for more distinct structures, but cannot
create diversity if the search has found only similar candidates.
See the {doc}`comparator guide <operators/comparators/index>` for available
methods and their settings.

Population membership is rebuilt using this simplified Python logic:

```{code-block} python
:caption: Python

retained = []
# Higher objective scores are better; history contains eligible candidates.
ranked = sorted(history, key=objective_score, reverse=True)
for candidate in ranked:
    duplicate = any(
        same_composition(candidate, other)
        and comparator.looks_like(candidate, other)
        for other in retained
    )
    if not duplicate:
        retained.append(candidate)
    if len(retained) == retained_size:
        break
```

The objective determines the order; the comparator decides similarity. Because
better candidates are visited first, a later similar candidate is skipped.
Different compositions are treated as distinct. Skipped candidates remain in
the database, and the retained pool can be smaller than `retained_size` if there
are not enough distinct candidates.

## optional extinction

`system.thanos` excludes evaluated candidates that violate physical or geometric
requirements. Uncomment the example rule to mark structures with Cu–Cu contacts
below 1.5 Å as extinct. This distance is illustrative; choose a cutoff appropriate
for your system.

Extinct structures remain in the database but cannot enter the retained
population, regardless of their energy. The rule filters results after evaluation;
it does not constrain relaxation. Leave it commented out to disable extinction
checks. See {doc}`population/thanos` for other rules and configuration.

```{toctree}
:maxdepth: 1
:hidden:

population/thanos
```
