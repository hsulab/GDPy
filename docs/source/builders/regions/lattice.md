(region-lattice)=

# lattice

An explicit cell defined by three lattice vectors and a Cartesian `origin`.
Use it to sample positions in a parallelepiped, including a skewed cell.
The cell vectors are rows; a flat list of nine values is also accepted.

```yaml
region:
  method: lattice
  origin: [0, 0, 2]
  cell:
    - [10, 0, 0]
    - [0, 10, 0]
    - [0, 0, 1]
```

Sampling draws fractional coordinates in `[0, 1)` and returns
`origin + fractional @ cell`. In this example, sampled z coordinates lie in
`[2, 3)` Å. Use a nondegenerate, right-handed cell; volume is the scalar triple
product of its vectors.

## Membership and periodicity

Membership wraps fractional coordinates into `[0, 1)` in all three directions.
Coordinates are measured relative to the region's `origin` and `cell`, not the
structure's simulation cell, and structure PBC flags do not disable this
explicitly periodic region. Negative coordinates and translations by any number
of cells are supported. Integer cell boundaries map to the lower face.
The sampled
cell is therefore not a finite containment box: a point above its upper z
face can still be included. For a finite Cartesian z interval with periodic
xy membership, use {doc}`surface`.

Every finite point has an image in the lattice cell, so this region does not
exclude points merely because they are far from the origin. Nonfinite positions
are excluded. The membership interval is fixed; there is no `wrap` option for
this region. Sampling remains confined to the single cell specified above.

The figure illustrates the sampling cell with origin `(0, 0, 2)` Å and height
1 Å; it does not depict a finite z cutoff for membership.

```{image} ../../../images/region-lattice.png
:align: center
:alt: Lattice sampling cell
:width: 800
```
