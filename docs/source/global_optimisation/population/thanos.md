(global-optimisation-thanos)=

# Thanos

Thanos marks evaluated structures as extinct when they violate a configured
rule. Extinct candidates remain in `candidates.db` with `extinct=1`, but cannot
enter the retained population or be selected as parents or chain starts.
GA, BH, and simulated annealing share the `system.thanos` configuration.

Thanos filters evaluated results. It does not add forces, constrain relaxation,
or reject builder proposals before calculation. Apply builder restraints or
executor constraints separately when those stages also need restrictions.

## Configuration

Set `system.thanos` to a single rule mapping or a list of rule mappings. Any
rule returning extinction excludes the structure (logical OR). Omit the field
or set it to `null` to disable extinction checks.

| `name` | Extinction condition |
| --- | --- |
| `region` | Any selected atom lies in a forbidden spatial region. |
| `restraints` | Any geometric restraint fails. |
| `particle_number` | The number of tagged particles of a formula falls outside its bounds. |

The following snippets belong inside an existing exploration configuration.
They illustrate individual rules; multiple rules can be listed under the same
`system.thanos` key.

## Region

Select atoms with `group` and define a forbidden `region`. For example, exclude
any oxygen atom in a Cartesian box:

```yaml
system:
  thanos:
    - name: region
      group: "`symbol O`"
      wrap: true
      region:
        method: cube
        origin: [0.0, 0.0, 0.0]
        boundary: [0.0, 0.0, 0.0, 10.0, 10.0, 3.0]
```

| Field | Meaning |
| --- | --- |
| `group` | Required, non-empty atom selection expression. |
| `region` | Required region configuration, using an existing region type. |
| `wrap` | Boolean, default `true`; wrap positions along periodic axes before checking membership. |

The cube boundary order is `[xmin, ymin, zmin, xmax, ymax, zmax]`, relative to
`origin`, in Å. Cube faces are included. Other region types use their own
membership conventions; see {doc}`../../builders/region`.

Selection uses the original structure and the standard
{doc}`group expressions <../../groups-and-constraints>`. For example,
`` `symbol O` and `tag 1:4` `` selects oxygen atoms in the specified tags.
The rule tests individual selected atoms, not molecular centers. A symbol
selection works even when all atoms share a tag. An empty selection survives.

Positions are checked on a copy; the input structure is unchanged.
`wrap: false` uses the unwrapped Cartesian coordinates. With wrapping enabled,
define the region in the wrapped simulation cell. The box is a preset region;
it does not automatically follow a moving surface or infer atomic layers.

## Geometric restraints

Use `name: restraints` to reject a structure when any entry in `restraints`
fails. For example, forbid O–O contacts closer than 1.6 Å:

```yaml
system:
  thanos:
    - name: restraints
      restraints:
        - type: contact_count
          pair: [O, O]
          max: 0
          distance:
            min: 0.0
            max: 1.6
          scope: all_atoms
```

`contact_count` limits the number of pairs; `coordination` constrains neighbor
counts around selected element types. `scope: all_atoms` checks pairs regardless
of tags. The default `scope: inter_particle` checks only atoms with different
tags, so shared tags can hide contacts you intended to exclude.

Explicit distances are in Å. If omitted, distance bounds use the sum of the
two elements' covalent radii multiplied by `covalent_ratio`, which defaults to
`[0.8, 2.0]`. See the existing
{doc}`O–O extinction guide <../bh/examples/cuox_thanos>` for a worked BH case.

## Particle number

Use `number: [formula, minimum, maximum]` to bound the number of particles with
the given chemical formula. Bounds are inclusive; `null` disables that bound.

```yaml
system:
  thanos:
    - name: particle_number
      number: [CO, 1, 4]
```

This accepts one to four CO particles and extinguishes structures with zero
or more than four. Particles are defined by atom tags: atoms sharing a tag form
one entity whose chemical formula is compared with `CO`. This is not a count
of geometric bonds or individual element occurrences. Give separate particles
distinct tags. If every atom shares a tag, the entire structure counts as one
entity with its combined formula.

## Population behavior and restart

Extinction is independent of energy ranking and duplicate detection. A favorable
energy does not rescue a candidate that fails a rule.

Inspect the database's `extinct` field to distinguish excluded candidates from
eligible ones. The flag combines all rules; it does not identify which rule
failed. Use the same extinction policy when restarting a run. Start a new output
directory when changing it.
