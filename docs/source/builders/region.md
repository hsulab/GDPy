(region-definitions)=

# regions

Regions define where to generate particles or check their positions. Builders
use them to sample candidate coordinates; selection and extinction rules use
membership checks. All lengths and Cartesian coordinates are in Å, and volumes
are in Å³.

Choose a region by its geometry:

| Region | Geometry and typical use |
| --- | --- |
| {doc}`auto <regions/auto>` | Use the current structure's simulation cell. |
| {doc}`surface <regions/surface>` | Full periodic xy plane with finite Cartesian z bounds. |
| {doc}`cube <regions/cube>` | Axis-aligned box with explicit bounds on all three axes. |
| {doc}`sphere <regions/sphere>` | Ball around a chosen center. |
| {doc}`hemisphere <regions/hemisphere>` | Half-ball oriented by a plane normal. |
| {doc}`cylinder <regions/cylinder>` | Vertical cylinder with a circular xy cross-section. |
| {doc}`lattice <regions/lattice>` | Explicit lattice cell for periodic membership and sampling. |

```{toctree}
:maxdepth: 1

regions/auto
regions/surface
regions/cube
regions/sphere
regions/hemisphere
regions/cylinder
regions/lattice
regions/surface_lattice
```

## Configuration and coordinates

Set `region.method` to the region name and supply its geometry parameters.
Each subpage gives a YAML fragment for an existing builder or operator input.
A region alone does not constrain motion during a calculation.

Region types own their coordinate conventions. Cubes, spheres, hemispheres,
and cylinders check the supplied Cartesian positions without minimum-image
distances. The {doc}`surface <regions/surface>` region provides `region.wrap`
for a configurable fractional interval before its Cartesian height check.
{ref}`Thanos <global-optimisation-thanos>` delegates membership to the region
without adding a separate wrapping step.

Use {doc}`surface <regions/surface>` for a slab with a finite height interval.
A lattice region repeats its membership check along z as well as xy, so it does
not impose a slab cutoff. The old `surface_lattice` implementation is described
on its {doc}`legacy reference page <regions/surface_lattice>`.

See {ref}`monte-carlo` for region-based sampling and
{ref}`global-optimisation-thanos` for excluding evaluated structures.
