(ga-bulk-example)=

# Bulk crystal

This minimal example searches for low-energy periodic cells containing four Cu
atoms. The `random_bulk` builder samples both atomic positions and cell shapes,
and ASE's Effective Medium Theory (EMT) calculator relaxes each candidate. The
small population and single generation make this a workflow demonstration, not
a converged Cu crystal-structure prediction.

## Input

The complete example is available at
`examples/global_optimisation/explorations/genetic_algorithm/cu4_bulk.yaml`:

```{literalinclude} ../../../../examples/global_optimisation/explorations/genetic_algorithm/cu4_bulk.yaml
:language: yaml
```

Pair this exploration with the following runtime (passed with `--runtime`):

```{literalinclude} ../../../../examples/global_optimisation/runtimes/emt_min_100.yaml
:language: yaml
```

The default periodic setting applies periodic boundary conditions in all three
directions. The builder fixes the cell volume at 48 Å³ while sampling cell
lengths and angles within the declared bounds. Periodic cut-and-splice combines
parent structures, while rattle and strain mutations vary atomic positions and
cell shape. Each Cu atom has its own tag, so default fragment preservation still
treats the atoms as independently movable particles.

## Run

From the repository root, run:

```shell
gdp -d ./run-cu4-bulk-emt \
    --runtime ./examples/global_optimisation/runtimes/emt_min_100.yaml explore \
    ./examples/global_optimisation/explorations/genetic_algorithm/cu4_bulk.yaml
```

The search is stored under `run-cu4-bulk-emt/expedition-0`. When it completes,
`results/all_candidates.xyz` contains the relaxed candidates ordered by their
GA score, and `results/pop.png` summarises the energies by generation. This
configuration produces four initial structures and two offspring.

:::{note}
EMT and the compact search settings keep the example inexpensive. Increase the
population, number of generations, and structural diversity before using this
workflow for a scientific bulk-structure search.
:::
