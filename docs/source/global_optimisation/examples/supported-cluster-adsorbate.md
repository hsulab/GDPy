(ga-supported-cluster-adsorbate-example)=
(ga-adsorbate-insertion-example)=

# supported cluster with an adsorbate

This example searches for low-energy structures of CO adsorbed on a supported
Cu4 cluster. Two builders contribute four candidates each to one initial
population, comparing random construction with adsorption-site insertion.
Every candidate contains Cu4 and one CO molecule on the same orthogonal
α-Al₂O₃(0001) support and is relaxed with the same MatterSim runtime.

## System and search region

The example reuses
`examples/global_optimisation/assets/alpha_alumina111_ortho.xyz` from the
{ref}`ga-supported-nanoparticle-example`. The 180-atom Al72O108 slab has a
14.28 × 16.49 Å orthogonal surface and 12 Al atoms in its exposed top layer.
The shared demo runtime allows all atoms to relax. To fix the bottom Al–O–Al
repeat unit for a surface study, add `constraint: lowest 60` to
`executor.parameters`.

The `random` builder generates Cu4 and CO together inside a sphere of radius
2.4 Å centred above the bare support. It samples both cluster–support geometry
and CO placement.

The `site_insertion` builder uses `method: adsorbate_insertion` and starts from
`examples/global_optimisation/assets/cu4_alumina111_supported.xyz`, which already
contains a tagged supported Cu4 cluster. It inserts only CO onto Cu adsorption
sites. The group `` `symbol Cu` ``, 3.0 Å connectivity cutoff, and `max_order: 1`
provide atop and bridge sites; C is the contact atom. The Cu atoms remain mobile
during subsequent crossover, mutation, and relaxation.

The named `random` builder remains the default reference builder for substrate
and bond-distance metadata. Both builders produce compatible atom ordering and
fragment tags, so their candidates can participate in the same population.
Completion candidates use `random`; the two-builder comparison is in generation 0.

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

```{literalinclude} ../../../../examples/global_optimisation/runtimes/mattersim.yaml
:language: yaml
```

The configuration uses fragment-preserving `cut_and_splice` crossover
and rattle mutation. MatterSim is used because the potential must describe Cu,
C, O, Al, and their interfaces; see the {ref}`potential-mattersim` guide for
installation and model details.

Fragment preservation applies to structure generation and GA operators. The
local MatterSim relaxation still optimises the C–O coordinates, so it does not
impose a rigid C–O bond.

## Run

After installing MatterSim, run from the repository root:

```shell
gdp -d ./run-cu4-co-alumina111 \
    --runtime ./examples/global_optimisation/runtimes/mattersim.yaml explore \
    ./examples/global_optimisation/explorations/genetic_algorithm/cu4_co_alumina111.yaml
```

The search is stored under `run-cu4-co-alumina111`. When it
completes, `results/all_candidates.xyz` contains the relaxed candidates and
`results/pop.png` summarises their energies by generation.

`results/family_tree.png` labels the two initial builder groups as `random` and
`site_insertion`. Candidate IDs and black parent arrows trace their descendants;
the shared energy color bar uses blue for lower energy and red for higher energy.
Compare the initial groups to see whether site insertion gives better starting
structures in this run. `candidates.log` provides exact relaxed scores, and
`tmp_folder/gen0/history.log` records which builder created each candidate.

This compares two initialization strategies, including their different starting
Cu4 geometries. Site insertion is not guaranteed to yield lower energies for
every sample or after relaxation. The shared 20-step relaxation limit also means
the displayed energies need not correspond to converged minima.

:::{note}
The thin slab, eight-candidate initial population, and single generation make this a
workflow demonstration. Converge the slab, surface area, population, search
length, and MatterSim settings before using the result scientifically.
:::
