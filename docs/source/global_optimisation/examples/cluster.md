(ga-cluster-example)=

# cluster

This minimal example searches for low-energy structures of an 8-atom copper
cluster. Random Cu8 candidates are generated inside a spherical region, then
relaxed with ASE's Effective Medium Theory (EMT) calculator. The example uses
four candidates in the initial population and two offspring in one subsequent
generation, so it is intended as a quick demonstration rather than a converged
global optimisation.

This is the same Cu₈ system and EMT runtime used by the
{ref}`BH cluster example <bh-cu8-example>`, with matching builder settings, seed,
and initial population size. The reproduction and hopping budgets are specific
to each method.

## Input

The complete example is available at
`examples/global_optimisation/explorations/genetic_algorithm/cu8.yaml`:

```{literalinclude} ../../../../examples/global_optimisation/explorations/genetic_algorithm/cu8.yaml
:language: yaml
```

Pair this exploration with the following runtime (passed with `--runtime`):

```{literalinclude} ../../../../examples/global_optimisation/runtimes/emt.yaml
:language: yaml
```

The search uses interatomic-distance comparison, `cut_and_splice`
crossover, and rattle mutation. A fixed random seed makes candidate generation
reproducible. Isolated clusters always use `system.periodic: true` and
`system.preserve_fragments: true`. This example uses a 20 × 20 × 20 Å periodic
vacuum cell and assigns each Cu atom its own positive tag, so individual atoms
can move and recombine while retaining consistent species tags.

## Run

From the repository root, run:

```shell
gdp -d ./run-cu8-emt \
    --runtime ./examples/global_optimisation/runtimes/emt.yaml explore \
    ./examples/global_optimisation/explorations/genetic_algorithm/cu8.yaml
```

The search is stored under `run-cu8-emt`. When it completes,
`results/all_candidates.xyz` contains the relaxed candidates ordered by the GA
score, while `results/pop.png` summarises the population energies by
generation. The example produces six relaxed Cu8 candidates: four initial
structures and two offspring.

:::{note}
EMT is inexpensive and convenient for demonstrating the workflow, but this
small population and single-generation search are not sufficient for a
scientific Cu8 global optimisation.
:::
