(ga-rattle-mutation)=

# rattle

The `rattle` mutation gives selected mobile atoms or tagged fragments random
Cartesian translations. It rejects a trial when the moved particles overlap
one another or the fixed substrate, and retries with new displacements.

For most searches, the defaults are sufficient:

```yaml
operators:
  mutation:
    method: rattle
```

When `mutation` is a list, `probability` is the relative selection weight of
this mutation. It defaults to `1.0` and is not part of the displacement itself.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `rattle_strength` | `0.8` | Maximum absolute displacement in Å along each Cartesian direction. Every selected particle receives an independently sampled vector whose components lie between `-rattle_strength` and `+rattle_strength`. |
| `rattle_prop` | `0.4` | Independent probability that each mobile atom or tagged fragment is selected. The mutation retries if no particle is selected. |
| `test_dist_to_slab` | `true` | Reject a trial when a moved particle is too close to the fixed substrate. |

For example, a gentler move affecting more of the mobile structure can be
configured as:

```yaml
mutation:
  method: rattle
  rattle_strength: 0.3
  rattle_prop: 0.7
```

`rattle_strength` limits each Cartesian component, so the length of the full
three-dimensional displacement can be larger than this value. Larger values
explore more distant configurations but also increase the chance that a trial
is rejected by the minimum-distance checks.

## Atoms, fragments, and substrate

With the default `population.preserve_fragments: true`, all atoms sharing one
positive ASE tag receive the same displacement. The mutation therefore moves
that fragment rigidly without changing its internal geometry. An independently
tagged atom behaves as a one-atom fragment. Set fragment preservation to
`false` only when the intended search is atom-wise.

Atoms belonging to the GA substrate are not rattled. GDPy obtains the
substrate, number of mobile atoms, minimum bond distances, fragment mode, and
random-number stream from the population and its reference builder. These are
managed settings and should not be repeated in the mutation configuration.
