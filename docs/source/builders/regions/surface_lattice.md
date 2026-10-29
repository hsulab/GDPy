(region-surface-lattice)=

# surface_lattice (legacy)

`SurfaceLatticeRegion` remains available as a Python compatibility class, but
it is not registered as a YAML region method. Do not use
`method: surface_lattice` in new configurations.

Use {doc}`surface` instead. It requires only `z_range` for membership, accepts
all lateral periodic images, and obtains the sampling cell from the current
substrate when used by the random structure builder.

The legacy class takes `origin` and `cell`, requires zero xy components in
`origin` and the third cell vector, and checks
`origin[2] ≤ z < origin[2] + cell[2][2]` before applying lattice membership.
It inherits periodic lattice membership, including negative lateral images.
The new surface region includes both z boundaries, so account for
that difference when migrating an exact boundary check.
