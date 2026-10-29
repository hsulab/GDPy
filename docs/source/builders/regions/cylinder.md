(region-cylinder)=

# cylinder

A cylinder aligned with the Cartesian z axis. `origin` is the center of the
bottom disk. Membership requires an xy distance of at most `radius` from that
center and `origin[2] ≤ z ≤ origin[2] + height`. The curved surface and both
end disks are included. Use positive radius and height; the volume is
`pi*radius**2*height`.

```yaml
region:
  method: cylinder
  origin: [5, 5, 2]
  radius: 2
  height: 6
```

The axis cannot be rotated with a `normal` parameter. Membership uses Cartesian
coordinates without minimum-image distances or periodic repetition.

The figure shows a vertical cylindrical region in a (10x10x10) simulation box.
The origin is (5,5,2), the radius is 2, and the height is 6.

```{image} ../../../images/region-cylinder.png
:align: center
:alt: Cylindrical region
:width: 800
```
