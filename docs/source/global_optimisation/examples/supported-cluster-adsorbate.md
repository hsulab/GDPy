(ga-supported-cluster-adsorbate-example)=

# Supported cluster with an adsorbate

This example searches for low-energy structures of CO adsorbed on a supported
Cu4 cluster. Four Cu atoms and one intact CO molecule are generated above an
orthogonal α-Al₂O₃(0001) slab, and the complete Cu4–CO/support structure is
relaxed with the 1-million-parameter MatterSim model.

## System and search region

The example reuses
`examples/global_optimisation/assets/alpha_alumina111_ortho.xyz` from the
{ref}`ga-supported-nanoparticle-example`. The 180-atom Al72O108 slab has a
14.28 × 16.49 Å orthogonal surface and 12 Al atoms in its exposed top layer.
Its bottom Al–O–Al repeat unit is fixed during relaxation with
`constraint: lowest 60`.

The Cu atoms and the centre of mass of CO are generated inside a sphere of
radius 2.4 Å centred above the support. Keeping all five mobile particles in
one compact region samples CO binding to different Cu4 structures as well as
different cluster–support geometries.

:::{note}
All support atoms have ASE tag 0. Each Cu atom receives its own positive tag,
while C and O share one positive tag identifying CO as a molecular fragment.
Default fragment preservation therefore makes crossover and mutation move CO
as one particle instead of separating its atoms.
:::

## Input

The exploration configuration is available at
`examples/global_optimisation/explorations/genetic_algorithm/cu4_co_alumina111.yaml`:

```{literalinclude} ../../../../examples/global_optimisation/explorations/genetic_algorithm/cu4_co_alumina111.yaml
:language: yaml
```

Pair this exploration with the following runtime (passed with `--runtime`):

```{literalinclude} ../../../../examples/global_optimisation/runtimes/mattersim_alumina_min_100.yaml
:language: yaml
```

The configuration uses fragment-preserving periodic cut-and-splice crossover
and rattle mutation. MatterSim is used because the potential must describe Cu,
C, O, Al, and their interfaces; see the {ref}`potential-mattersim` guide for
installation and model details.

Fragment preservation applies to structure generation and GA operations. The
local MatterSim relaxation still optimises the C–O coordinates, so it does not
impose a rigid C–O bond.

## Run

After installing MatterSim, run from the repository root:

```shell
gdp -d ./run-cu4-co-alumina111 \
    --runtime ./examples/global_optimisation/runtimes/mattersim_alumina_min_100.yaml explore \
    ./examples/global_optimisation/explorations/genetic_algorithm/cu4_co_alumina111.yaml
```

The search is stored under `run-cu4-co-alumina111/expedition-0`. When it
completes, `results/all_candidates.xyz` contains the relaxed candidates and
`results/pop.png` summarises their energies by generation.

:::{note}
The thin slab, four-candidate population, and single generation make this a
workflow demonstration. Converge the slab, surface area, population, search
length, and MatterSim settings before using the result scientifically.
:::
