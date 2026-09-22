(ga-alloy-cluster-example)=

# Alloy cluster with swap mutation

This example searches the chemical ordering of a Cu7Ni6 alloy cluster. Four
random cluster geometries are generated inside a spherical region and relaxed
with ASE's Effective Medium Theory (EMT) calculator. The following generation
uses only swap mutation, making the example a focused demonstration of
exchanging unlike atomic species without changing the composition.

## Input

The complete example is available at
`examples/global_optimisation/explorations/genetic_algorithm/cu7ni6.yaml`:

```{literalinclude} ../../../../examples/global_optimisation/explorations/genetic_algorithm/cu7ni6.yaml
:language: yaml
```

Pair this exploration with the following runtime (passed with `--runtime`):

```{literalinclude} ../../../../examples/global_optimisation/runtimes/emt.yaml
:language: yaml
```

The default periodic setting places the isolated cluster in a 20 × 20 × 20 Å periodic vacuum cell. The random
builder assigns every Cu and Ni atom a distinct positive tag, so the default
fragment-preserving mode treats each atom as an independently movable
particle.

The generation requests no crossover offspring and two mutation offspring.
`particles: [Cu, Ni]` restricts swap to Cu/Ni pairs, while `swap_ratio: 0.2`
gives one successful exchange for this composition. The builder and mutation
use the same `covalent_ratio`, ensuring that generated and swapped structures
are checked with consistent distance limits. If the requested mutations cannot
be produced within the attempt limit, the random builder completes the
generation.

## Run

From the repository root, run:

```shell
gdp -d ./run-cu7ni6-emt \
    --runtime ./examples/global_optimisation/runtimes/emt.yaml explore \
    ./examples/global_optimisation/explorations/genetic_algorithm/cu7ni6.yaml
```

The search is stored under `run-cu7ni6-emt`, with its restart state
in `candidates.db`. When it completes, `results/all_candidates.xyz` contains
the relaxed candidates ordered by energy, and `results/pop.png` summarises the
population energies by generation.

:::{note}
EMT and the small population keep this example inexpensive. They are suitable
for demonstrating swap-driven chemical ordering, not for a converged or
quantitatively predictive Cu-Ni cluster study.
:::
