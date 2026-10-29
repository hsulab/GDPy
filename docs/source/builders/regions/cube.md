(region-cube)=

# cube

An axis-aligned rectangular box. `origin` shifts the six relative bounds;
`boundary` is ordered as `[xmin, ymin, zmin, xmax, ymax, zmax]`. All faces are
included, and opposite faces need not have equal lengths.

```yaml
region:
  method: cube
  origin: [2, 2, 2]
  boundary: [0, 0, 0, 2, 2, 2]
```

This includes `2 ≤ x ≤ 4`, `2 ≤ y ≤ 4`, and `2 ≤ z ≤ 4` Å. The volume is
`(xmax - xmin) * (ymax - ymin) * (zmax - zmin)`. Membership uses Cartesian
coordinates without periodic repetition. For a full periodic xy plane, use
{doc}`surface` instead of specifying lateral bounds.

The figure shows a cubic region in a (10x10x10) simulation box. The origin is (2,2,2)
and the boundary is [0,0,0,2,2,2].

```{image} ../../../images/region-cube.png
:align: center
:alt: Cubic region
:width: 800
```
