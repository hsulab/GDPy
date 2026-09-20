(global-optimisation-ga)=

# Genetic Algorithm (GA)

A genetic algorithm searches for low-energy structures by evolving a population
of candidates. It is useful when the configuration space is too large for
systematic enumeration and structural features from good candidates can be
recombined to discover better ones.

```{toctree}
:maxdepth: 2

operations
examples/index
```

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

## Components in GDPy

GDPy separates the search policy from candidate construction and evaluation:

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

The {ref}`genetic-algorithm` guide contains GDPy's YAML configuration reference,
available operators, command-line example, and application links.
