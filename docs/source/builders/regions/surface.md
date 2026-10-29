(region-surface)=

# surface

This region spans the full periodic xy plane and a finite Cartesian z interval.
Use it for a slab when specifying lateral box boundaries would be redundant:

```yaml
region:
  method: surface
  z_range: [3.0, 8.0]
  wrap: [-0.1, 0.9]
```

The two heights are absolute Cartesian coordinates in Å, with both boundaries
included. Atom membership accepts any lateral periodic image, including negative
x or y. Cartesian z is checked after the optional wrapping step below;
the height interval itself never repeats.
The bounds must be finite and strictly increasing.

The random structure builder takes the xy cell vectors from each substrate and
samples uniformly over one lateral cell and the specified z interval. An explicit
`cell` (a 3 × 3 matrix) can also be supplied. The first two cell vectors must lie
in the xy plane and span a nonzero area; skewed xy cells are supported. Volume is
the xy cell area times `zmax - zmin`. Direct Python use requires
`region.preprocess(atoms)` before sampling or volume evaluation when no cell
was supplied. Membership uses the cell and PBC flags of the structure being
checked, independently of an explicit sampling cell.

## Wrapping

`wrap` is a surface-region option, applied consistently to direct atom membership,
group-expression membership, tagged particle centers, and Thanos checks:

| Value | Coordinate convention |
| --- | --- |
| `true` (default) | Wrap periodic fractional coordinates into `[-0.1, 0.9)`. |
| `[lower, upper]` | Use a custom finite interval spanning exactly one cell. |
| `false` | Check original Cartesian positions. |

The upper endpoint maps to the lower endpoint. Nonperiodic axes remain unchanged.
Wrapping uses the checked structure's simulation cell, never `z_range`, and does
not move the input atoms. Random sampling still uses the specified Cartesian
height interval; `wrap` controls membership, not generated coordinates.

For example, z = 19.5 Å in an orthogonal 20 Å periodic cell is checked as
−0.5 Å with the default interval, but as 19.5 Å with `wrap: [0.0, 1.0]`.
Choose the interval to keep the slab contiguous around its desired origin.
Set `wrap: false` to check an unwrapped height, or use two regions for a
height interval crossing the chosen wrapping boundary.

The string group syntax accepts optional fractional bounds:

```yaml
group: "`region surface -1 3 -0.1 0.9`"
```

Omitting the last two numbers uses the default interval. Tagged-particle
membership averages the wrapped atomic positions to locate each particle center;
as with other regions, a particle straddling the wrapping boundary needs care.
