(global-optimisation-ga)=

# genetic algorithm (ga)

A genetic algorithm searches for low-energy structures by evolving a population
of candidates. It is useful when the configuration space is too large for
systematic enumeration and structural features from good candidates can be
recombined to discover better ones.

```{toctree}
:maxdepth: 2

operations
examples/index
```

See {ref}`exploration-output-layout` for output directories and restart metadata.

GA prints a setup box and a box for each generation, with worker output nested
inside. Each generation ends with its status, committed evaluation and extinction
counts, best eligible objective, and elapsed time for the current invocation.
Resumed generations show the number of evaluations already committed. Routine
diagnostics remain available at DEBUG level, and warnings remain visible.

Two text files retain the candidate details outside the console boxes:

- `tmp_folder/genN/history.log` lists committed reproduction, mutation, and
  random/builder steps, including intermediate mutations, candidate IDs, parents,
  origins, builder names, and operation descriptions. The header includes the
  production stage and reproduction/mutation attempt totals. Individual failed
  attempt diagnostics are not stored in the database and cannot be reconstructed.
- `candidates.log` in the exploration directory collects evaluated candidates
  across all generations, with timestamp, candidate ID, fitness (`raw_score`),
  extinction flag, and species counts. Fitness is maximised; it is distinct from
  the minimised objective shown in the generation box. Searches with
  `population.preserve_fragments: true` report fragment/species counts; other
  searches report elemental atom counts, regardless of crossover provenance tags.

These files are regenerated from committed database records at generation and
invocation boundaries, including partial results when an invocation fails.
Timestamps come from the original database records. Resuming a search refreshes
the files without duplicating entries and also reconstructs history for older runs.

Completed searches also write a compact 1200 × 600 `results/family_tree.png`.
The family tree places candidates in generation rows
and connects parents to offspring. Only candidate IDs, generation numbers, and
the objective color bar are labeled, with the same `coolwarm` palette as BH.
When initialization uses multiple builders, the initial row is grouped and
directly labeled by builder name to compare their energies and descendants.
Candidate IDs are shown
when space permits, including 20 candidates per generation across an initial
population and 10 evolved generations. Larger searches use smaller markers and
omit IDs when necessary. Intermediate mutations retain the original crossover parents
instead of creating self-links. The figure is rebuilt from database history when
reporting a completed search, including after a restart.

## Search cycle

A typical GA search repeats the following steps:

1. **Initialise** a diverse population of valid structures.
2. **Evaluate** each candidate by relaxing it and calculating its objective,
   usually energy.
3. **Select** parent structures, favouring strong candidates while retaining
   enough diversity to avoid premature convergence.
4. **Reproduce** candidates with crossover and mutation operators.
5. **Relax** the offspring to nearby local minima.
6. **Update** the population and test the convergence criterion.

```{image} ../../images/ga.png
:align: center
:alt: Genetic algorithm search cycle
:width: 800
```

The shared {ref}`global-optimisation-population` reference defines population
sizes, builders, and comparison for both GA and BH.

## Components in gdpx

gdpx separates the search policy from candidate construction and evaluation:

- A builder generates the initial population and any fresh random candidates.
- Crossover combines parts of two parent structures.
- Mutation introduces structural variation into offspring.
- A comparator identifies similar candidates and helps maintain diversity.
- The configured runtime relaxes and scores every candidate.
- A convergence rule determines when the search ends.

The quality of a search depends on balancing exploitation of low-energy
structures with exploration of new regions. An overly similar population can
converge early, while excessive random variation can prevent useful structural
features from being retained.

## Running a GA search

The {ref}`genetic-algorithm` guide contains gdpx's YAML configuration reference,
available operators, command-line example, and application links.
