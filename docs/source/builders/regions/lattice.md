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

Membership uses fractional remainders in all three directions. The sampled
cell is therefore not a finite containment box: a point above its upper z
face can still be included. For a finite Cartesian z interval with periodic
xy membership, use {doc}`surface`.

The current membership implementation uses signed fractional remainders.
Negative, nonintegral fractional coordinates relative to `origin` are rejected
rather than wrapped into positive images. Avoid assuming symmetric handling
of positive and negative images with this legacy behavior. Caller-side wrapping
also depends on the simulation cell, which may differ from this region's cell.

The figure illustrates the sampling cell with origin `(0, 0, 2)` Å and height
1 Å; it does not depict a finite z cutoff for membership.

```{image} ../../../images/region-lattice.png
:align: center
:alt: Lattice sampling cell
:width: 800
```
