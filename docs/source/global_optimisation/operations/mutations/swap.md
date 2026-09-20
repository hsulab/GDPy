(ga-swap-mutation)=

# Swap

The `swap` mutation exchanges the positions of two differently typed tagged
particles without changing the overall composition. It is most useful for
exploring chemical ordering in alloys or mixtures after suitable geometries
already exist in the population.

A minimal configuration automatically considers every pair of mobile particle
types:

```yaml
operators:
  mutation:
    method: swap
```

Swap requires at least two mobile particle types. The fixed substrate, whose
atoms have tag 0, is never selected.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `particles` | all detected pairs | Particle types that may exchange positions. A flat list permits every pairwise combination; a list of two-item lists permits only those explicit pairs. |
| `swap_ratio` | `0.33` | Fraction used to determine the requested number of exchanges from the average number of particles per mobile type. At least one exchange is requested. |
| `covalent_ratio` | `[0.8, 2.0]` | Lower and upper acceptable distance ratios for validating each exchanged pair against the reference builder's bond distances. |

For example, this configuration permits only Cu/Ni exchanges and requests one
exchange for the Cu7Ni6 example composition:

```yaml
mutation:
  method: swap
  particles: [Cu, Ni]
  swap_ratio: 0.2
  covalent_ratio: [0.7, 2.0]
```

With more than two types, a flat list enables every combination:

```yaml
particles: [Cu, Ni, Pd]
```

Use nested pairs to restrict the allowed exchanges:

```yaml
particles:
  - [Cu, Ni]
  - [Ni, Pd]
```

Particle type names are the chemical formulas of the tagged particles, such as
`Cu`, `Ni`, or `CO`. If no permitted pair is present, or no trial satisfies the
distance checks, the mutation returns no offspring and the population's normal
completion strategy handles the deficit.

## Tags and fragments

Swap identifies particles by positive ASE tags. Each independently tagged atom
is one particle. Atoms sharing a positive tag form one fragment, which is moved
and randomly reoriented as a rigid unit so its internal geometry is preserved.
Both members of a fragment pair must currently contain the same number of atoms
so their coordinate arrays can be exchanged.

The population manages tag use, the bond-distance dictionary, and the random
number stream. Do not set `use_tags`, `bond_distance_dict`, or `rng` in the
mutation configuration. In particular, use the same `covalent_ratio` in the
reference builder and swap configuration when consistent generation and
mutation distance criteria are required.

See the {ref}`Cu₇Ni₆ alloy-cluster example <ga-alloy-cluster-example>` for a
complete swap-only GA search using EMT.
