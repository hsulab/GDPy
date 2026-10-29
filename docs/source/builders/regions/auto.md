(region-auto)=

# auto

Use the simulation cell of the current structure without repeating its cell
vectors in the input.

```yaml
region:
  method: auto
```

Random positions are sampled in fractional `[0, 1)` along each cell vector and
converted to Cartesian coordinates. The volume is the simulation-cell volume.
Membership uses periodic fractional coordinates in all three directions; this
region does not impose a finite surface-height cutoff. Use {doc}`surface` for
that purpose.

The calling builder or operator must attach the current structure. For direct
Python use, call `region.preprocess(atoms)` before sampling, checking membership,
or evaluating volume. A nondegenerate cell is required for membership and volume.
