(ga-water-cluster-example)=

# Molecular cluster

This example searches for low-energy structures of a four-water cluster. The
builder inserts four intact H2O molecules in a spherical region at the centre
of a large periodic vacuum box and assigns one tag to each molecule. Structures
are relaxed with the 1-million-parameter MatterSim checkpoint, the fastest
pretrained MatterSim model. See the {ref}`potential-mattersim` guide for
installation, model selection, and runtime settings.

## Input

The complete example is available at
`examples/global_optimisation/explorations/genetic_algorithm/water4.yaml`:

```{literalinclude} ../../../../examples/global_optimisation/explorations/genetic_algorithm/water4.yaml
:language: yaml
```

Pair this exploration with the following runtime (passed with `--runtime`):

```{literalinclude} ../../../../examples/global_optimisation/runtimes/mattersim.yaml
:language: yaml
```

The periodic cut-and-splice implementation preserves tagged molecular
fragments and their inherited orientations, while rattle translates whole
water molecules. The MatterSim relaxation may change intramolecular coordinates
because the model evaluates and relaxes all atoms.

## Run

After installing MatterSim as described in the {ref}`potential-mattersim`
guide, run from the repository root:

```shell
gdp -d ./run-water4-mattersim \
    --runtime ./examples/global_optimisation/runtimes/mattersim.yaml explore \
    ./examples/global_optimisation/explorations/genetic_algorithm/water4.yaml
```

The search is stored under `run-water4-mattersim/expedition-0`. The small
population and single generation keep this example quick; increase both for a
production search.

:::{note}
This compact example demonstrates the workflow rather than providing a
converged or validated water-cluster study.
:::
