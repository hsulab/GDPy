(ga-water-cluster-example)=

# (H<sub>2</sub>O)<sub>4</sub> molecular cluster

This example searches for low-energy structures of a four-water cluster. The
builder inserts four intact H2O molecules in a spherical region at the centre
of a large periodic vacuum box and assigns one tag to each molecule. Structures
are relaxed with xreac and the bundled H/O ReaxFF parameters. See the
{ref}`potential-reax` guide for installation and runtime settings. No
neural-network model download is needed.

## Input

The complete example is available at
`examples/global_optimisation/explorations/genetic_algorithm/water4.yaml`:

```{literalinclude} ../../../../examples/global_optimisation/explorations/genetic_algorithm/water4.yaml
:language: yaml
```

Pair this exploration with the following runtime (passed with `--runtime`):

```{literalinclude} ../../../../examples/global_optimisation/runtimes/xreac.yaml
:language: yaml
```

As for all isolated clusters, use `system.periodic: true` and
`system.preserve_fragments: true`. The `cut_and_splice` crossover preserves tagged molecular
fragments and their inherited orientations, while rattle translates whole
water molecules. The ReaxFF relaxation may change intramolecular coordinates
because the model evaluates and relaxes all atoms.

## Run

After installing `gdpx[reax]` as described in the {ref}`potential-reax`
guide, run from the repository root:

```shell
gdp -d ./run-water4-reax \
    --runtime ./examples/global_optimisation/runtimes/xreac.yaml explore \
    ./examples/global_optimisation/explorations/genetic_algorithm/water4.yaml
```

The search is stored under `run-water4-reax`. The small
population and single generation keep this example quick; increase both for a
production search.

:::{note}
This compact example demonstrates the workflow rather than providing a
converged or validated water-cluster study.
:::
