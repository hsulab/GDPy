(ga-cluster-example)=

# Cluster

This minimal example searches for low-energy structures of a 13-atom copper
cluster. Random Cu13 candidates are generated inside a spherical region, then
relaxed with ASE's Effective Medium Theory (EMT) calculator. The example uses
four candidates in the initial population and two offspring in one subsequent
generation, so it is intended as a quick demonstration rather than a converged
global optimisation.

## Input

The complete example is available at
`examples/global_optimisation/explorations/genetic_algorithm/cu13.yaml`:

```{literalinclude} ../../../../examples/global_optimisation/explorations/genetic_algorithm/cu13.yaml
:language: yaml
```

Pair this exploration with the following runtime (passed with `--runtime`):

```{literalinclude} ../../../../examples/global_optimisation/runtimes/emt.yaml
:language: yaml
```

The search uses interatomic-distance comparison, cluster cut-and-splice
crossover, and rattle mutation. A fixed random seed makes candidate generation
reproducible. `population.periodic: true` uses a 20 × 20 × 20 Å periodic vacuum cell for the
cluster. `population.preserve_fragments: false` permits the atom-wise cluster
crossover.

## Run

From the repository root, run:

```shell
gdp -d ./run-cu13-emt \
    --runtime ./examples/global_optimisation/runtimes/emt.yaml explore \
    ./examples/global_optimisation/explorations/genetic_algorithm/cu13.yaml
```

The search is stored under `run-cu13-emt/expedition-0`. When it completes,
`results/all_candidates.xyz` contains the relaxed candidates ordered by the GA
score, while `results/pop.png` summarises the population energies by
generation. The example produces six relaxed Cu13 candidates: four initial
structures and two offspring.

:::{note}
EMT is inexpensive and convenient for demonstrating the workflow, but this
small population and single-generation search are not sufficient for a
scientific Cu13 global optimisation.
:::
