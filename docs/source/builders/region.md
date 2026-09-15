(region-definitions)=

# Regions

A region is a space defined in the Cartesian coordinate system, which help operators
access and modify atoms in a much easier way. See its application in {ref}`monte-carlo`.

Define a region in the `yaml` input file as follows (units: Ang):

- `auto`.

  > This region takes the simulation box of the input atoms.

```yaml
region:
  method: auto
```

- `cube`.

```yaml
# Create a cube as ox+xl <= x <= ox+xh, oy+yl <= y <= oy+yh, oz+zl <= z <= oz+zh
region:
  method: cube
  origin: [50, 50, 50] # ox, oy, oz
  boundary: [0, 0, 0, 10, 10, 10] # xl, yl, zl, xh, yh, zh
```

The figure shows a cubic region in a (10x10x10) simulation box. The origin is (2,2,2)
and the boundary is [0,0,0,2,2,2].

```{image} ../../images/region-cube.png
:align: center
:alt: Cubic region
:width: 800
```

- `sphere`.

```yaml
# Create a sphere centre at [50, 50, 50] with the radius 50.
region:
  method: sphere
  origin: [50, 50, 50]
  radius: 50
```

The figure shows a spherical region in a (10x10x10) simulation box. The origin is (5,5,5)
and the radius is 2.

```{image} ../../images/region-sphere.png
:align: center
:alt: Spherical region
:width: 800
```

- `cylinder`.

```yaml
# Create a vertical cylinder.
region:
  method: cylinder
  origin: [50, 50, 50]
  radius: 50
  height: 20
```

The figure shows a vertical cylinderal region in a (10x10x10) simulation box.
The origin is (5,5,2), the radius is 2, and the height is 6.

```{image} ../../images/region-cylinder.png
:align: center
:alt: Cylindrical region
:width: 800
```

- `lattice`.

```yaml
# Create a periodic cubic lattice that is centred at [50, 50, 50].
region:
  method: lattice
  origin: [0, 0, 2]
  cell: [10, 0, 0, 0, 10, 0, 0, 0, 1]
```

The figure shows a lattice region in a (10x10x10) simulation box. The origin is (0,0,2)
and the cell is [10,0,0,0,10,0,0,0,1]. The surface thickness is 1 and atoms with z-coordinate
within \[2, 3) will be considered as in the region. Periodic boundary condition is used for this region.

```{image} ../../images/region-lattice.png
:align: center
:alt: Lattice region
:width: 800
```

- `surface_lattice`

This has the same definition as `lattice`. However, periodic boundary condition
is only applied in x- and y-axis. This is useful when only considering surface atoms.
