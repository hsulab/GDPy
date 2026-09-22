(ga-surface-oxide-example)=

# Surface oxide

This example searches Cu–O reconstructions on a two-layer Cu(111)-p(2×2)
substrate. The right-angled surface cell contains eight substrate Cu atoms. Four
additional Cu atoms and four O atoms are placed above it, giving Cu12O4
candidates that are relaxed with the 1-million-parameter MatterSim model.

## Substrate and search region

The substrate is provided as
`examples/global_optimisation/assets/cu111_p2x2_2layer.xyz`. Its orthogonal
in-plane cell is 5.11 × 4.43 Å. The lower and upper Cu layers are placed at
z = 2.00 and 4.09 Å in a 22 Å-tall cell, leaving most of the cell as vacuum
above the surface.

:::{note}
Substrate atoms must have ASE tag 0. The builder assigns a distinct positive
tag to every added Cu or O atom, allowing GA operations to distinguish the
fixed substrate from the searchable overlayer.
:::

The insertion region spans the complete right-angled periodic surface cell from
z = 4.60 to 9.10 Å, leaving about 12.9 Å of clear vacuum above it.
The default periodic setting applies periodic boundary conditions in all three
directions. The shared demo runtime allows all atoms to relax. For a surface
study, add `constraint: lowest 4` to `executor.parameters` to fix the lower Cu
layer while relaxing the upper layer and Cu–O overlayer.

## Input

The exploration configuration is available at
`examples/global_optimisation/explorations/genetic_algorithm/cu4o4_cu111.yaml`:

```{literalinclude} ../../../../examples/global_optimisation/explorations/genetic_algorithm/cu4o4_cu111.yaml
:language: yaml
```

Pair this exploration with the following runtime (passed with `--runtime`):

```{literalinclude} ../../../../examples/global_optimisation/runtimes/mattersim.yaml
:language: yaml
```

The search uses periodic cut-and-splice crossover and rattle mutation. Every
added Cu and O atom has its own tag, so default fragment preservation treats
them as independent particles. MatterSim is used instead of EMT because this
example requires a potential that describes both Cu and O; see the
{ref}`potential-mattersim` guide for installation and model details.

## Run

After installing MatterSim, run from the repository root:

```shell
gdp -d ./run-cu4o4-cu111 \
    --runtime ./examples/global_optimisation/runtimes/mattersim.yaml explore \
    ./examples/global_optimisation/explorations/genetic_algorithm/cu4o4_cu111.yaml
```

The search is stored under `run-cu4o4-cu111`. When it completes,
`results/all_candidates.xyz` contains the relaxed structures ordered by their
GA score, while `results/pop.png` summarises their energies by generation.

:::{note}
The two-layer slab, four-candidate population, and single generation keep this
example compact. Use a thicker slab, test the vacuum and cell size, and expand
the population and convergence settings for a scientific surface-oxide study.
:::
