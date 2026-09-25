(bh-seeded-cu8-example)=

# Seeded Cu<sub>8</sub> cluster

This example starts one basin-hopping search from **two random structures and
two seed structures**. It uses two named builders in the same population:
`random` generates fresh Cu₈ clusters and `seeds` reads selected frames from a
checked-in structure file. All four initial candidates are relaxed using EMT
before the search selects chain starts.

## Input

Run from the repository root so the seed-file path resolves correctly.
The exploration input is
`examples/global_optimisation/explorations/basin_hopping/cu8_seeded.yaml`:

```{literalinclude} ../../../../../examples/global_optimisation/explorations/basin_hopping/cu8_seeded.yaml
:language: yaml
```

The seed file contains two hand-built starting geometries: a slightly perturbed
cube and a slightly perturbed square antiprism. They are unrelaxed examples, not
claimed low-energy minima. Both frames contain eight Cu atoms, a 20 × 20 × 20 Å
periodic vacuum cell, and distinct positive atom tags (1–8), matching the random
builder's system description. The tags allow moves of individual Cu atoms.

```{literalinclude} ../../../../../examples/global_optimisation/assets/cu8_seeds.xyz
:language: text
:caption: cu8_seeds.xyz (two extended-XYZ frames)
```

`indices: [0, 1]` selects both seed frames in file order. The allocation sizes
must sum to `initial.total_size`: two random candidates plus two seeds gives
four initial evaluations. The `direct` builder returns its selected frames;
its allocation must match that frame count. It does not randomly sample or
truncate a larger file to the requested allocation size.

`random` remains the default reference builder, supplying system metadata to
the move operators. Both sources feed the same retained pool, whose capacity
is four distinct candidates. Similar relaxed candidates can be deduplicated,
and fitness-weighted selection chooses two chain starts from that pool. A seed
is not guaranteed to start a chain simply because it was provided.

Each chain attempts five steps in generation 1. Including initialization, the
search evaluates at most 14 structures; invalid proposals do not trigger an
evaluation. The shared {doc}`population settings <../../population>` explain
initial, retained, and generation sizes.

## Run with EMT

Use the existing relaxation runtime:

```shell
gdp -d ./run-cu8-seeded-bh-emt \
    --runtime ./examples/global_optimisation/runtimes/emt.yaml explore \
    ./examples/global_optimisation/explorations/basin_hopping/cu8_seeded.yaml
```

No model download is needed. The runtime relaxes every initial structure,
including the seeds, and every valid hopping trial. You can also embed the
runtime in the exploration file as shown in the
{doc}`global optimisation overview <../../index>`.

Inspect the initial candidates' builder provenance in the database:

```python
from ase.db import connect

database = connect("run-cu8-seeded-bh-emt/candidates.db")
for row in database.select(relaxed=1, generation=0):
    print(row.confid, row.data["builder"], row.energy)
```

There are two initial rows labeled `random` and two labeled `seeds`, even if
some candidates later relax to similar minima. Reports and the lineage figure
are under `results/`; see the {doc}`basic Cu₈ example <cluster>` for trajectory
export and the output layout.

## Supply your own seeds

Replace `system.builders.seeds.frames` with your structure-file path and
set `indices` to the frames you want to include. Update the `seeds` allocation
and `initial.total_size` together if the number of selected frames changes.
Keep the cell, periodicity, composition, and tags consistent with the intended
search and moves. For this atomic Cu₈ example, give each atom a distinct positive
tag and preserve the periodic vacuum cell. Extended XYZ stores both the cell
and tags; plain XYZ may omit this information.

Seeds are starting geometries, not cached evaluations: their calculators and
previous search metadata are cleared, and the configured runtime evaluates them
again. Use a new output directory when changing the seed file or initialization
allocations. Resume an interrupted run with the same configuration and seed file.
