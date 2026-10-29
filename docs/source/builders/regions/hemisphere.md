(region-hemisphere)=

# hemisphere

This solid half-sphere includes points within `radius` of `origin` on the side
pointed to by `normal`: `(position - origin) dot normal >= 0`. Both the flat face
and spherical surface are included. The normal may have any nonzero length;
it defaults to `[0, 0, 1]`. Random positions are sampled uniformly in volume.

```yaml
region:
  method: hemisphere
  origin: [5, 5, 2]
  radius: 2
  normal: [0, 0, 1] # Include the half above z = 2.
```

The volume is `2*pi*radius**3/3`. The radius must be finite and positive,
and the origin and normal must contain finite coordinates.
