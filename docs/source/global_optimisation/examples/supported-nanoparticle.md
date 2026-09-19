(ga-supported-nanoparticle-example)=

# Supported nanoparticle

This example searches for low-energy Cu4 nanoparticles supported on
α-Al₂O₃(0001), which is the basal surface corresponding to `(111)` in the
rhombohedral setting. Random Cu clusters are generated above an orthogonal 2×2
Al₂O₃ surface cell and relaxed with the 1-million-parameter MatterSim model.

## Substrate and search region

The substrate is provided as
`examples/global_optimisation/assets/alpha_alumina111_ortho.xyz`. Its 180 atoms
form a stoichiometric Al72O108 slab in a 14.28 × 16.49 × 34.0 Å orthogonal
cell, obtained with a 3×2 repeat of the orthogonal surface unit. This gives a
more nearly square search area with 12 Al atoms in the exposed top layer. The
demo uses nine atomic planes, spanning z = 2.00–8.01 Å, to reduce its cost while
leaving a large vacuum region above the exposed Al-terminated surface.

An exact 3×3 repeat of the primitive α-Al₂O₃(0001) surface would have equal
in-plane lattice-vector lengths, but those vectors meet at 120°. The 3×2
orthogonal repeat is used here to retain rectangular periodic boundaries while
keeping the two in-plane dimensions similar.

The Cu atoms are generated inside a sphere of radius 1.8 Å centred above the
surface. This confines the atoms to a compact supported nanoparticle instead of
distributing them across the complete surface cell. The top of the generation
region is at z = 11.8 Å, leaving about 22.2 Å of upper vacuum.

:::{note}
All substrate Al and O atoms must have ASE tag 0. The builder assigns a distinct
positive tag to every Cu atom, allowing GA operations to distinguish the fixed
support from the searchable nanoparticle.
:::

During relaxation, `constraint: lowest 60` fixes only the bottom Al–O–Al repeat
unit while allowing the upper six atomic planes and the Cu nanoparticle to
relax.

## Input

The complete search configuration is available at
`examples/global_optimisation/cu4_alumina111_mattersim.yaml`:

```{literalinclude} ../../../../examples/global_optimisation/cu4_alumina111_mattersim.yaml
:language: yaml
```

The search uses periodic cut-and-splice crossover and rattle mutation. Fragment
preservation is disabled because each Cu atom is independently movable.
MatterSim is used because the potential must describe Cu, Al, O, and their
interfaces; see the {ref}`potential-mattersim` guide for installation and model
details.

## Run

After installing MatterSim, run from the repository root:

```shell
gdp -d ./run-cu4-alumina111 explore \
    ./examples/global_optimisation/cu4_alumina111_mattersim.yaml
```

The search is stored under `run-cu4-alumina111/expedition-0`. When it completes,
`results/all_candidates.xyz` contains the relaxed structures ordered by their
GA score, while `results/pop.png` summarises their energies by generation.

:::{note}
The compact population and single generation are intended to demonstrate the
workflow. Converge the slab thickness, surface area, vacuum, population size,
and number of generations before a scientific supported-cluster study.
:::
