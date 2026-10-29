(region-sphere)=

# sphere

A ball centered at `origin`, including its outer surface. Membership requires
the Cartesian distance to the center to be no greater than `radius`. Use a
positive radius; the volume is `4*pi*radius**3/3`.

```yaml
region:
  method: sphere
  origin: [5, 5, 5]
  radius: 2
```

Membership does not use minimum-image distances. A point near the opposite
simulation-cell face is not included solely because it is a periodic image of
a point near the center. Use {doc}`hemisphere` when only one side of a plane
through the center should be included.

The figure shows a spherical region in a (10x10x10) simulation box. The origin is (5,5,5)
and the radius is 2.

```{image} ../../../images/region-sphere.png
:align: center
:alt: Spherical region
:width: 800
```
