(ga-mirror-mutation)=

# mirror

The `mirror` mutation selects approximately half of the mobile atoms of each
element on one side of a randomly oriented cutting plane through their mean
position. It mirrors those atoms to replace the other half, preserving the
number of atoms of each element. The fixed substrate is retained. Trials that
violate minimum distances within the mobile structure or to the substrate are
rejected.

For an atomic cluster search, configure:

```yaml
system:
  periodic: false
  preserve_fragments: false
strategy:
  operators:
    mutation:
      - method: mirror
        probability: 1.0
        reflect: false
```

This is a configuration excerpt; the remaining system and strategy settings
must still be supplied. `probability` is the relative selection weight when
several mutations are configured, and defaults to `1.0`. It does not guarantee
that a valid mirrored offspring will be produced.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `reflect` | `false` | Also reverse the mirrored half's coordinates parallel to the cutting plane, relative to the mobile structure's mean position. |

The mutation makes up to 1000 attempts to find an offspring satisfying the
minimum-distance checks. After more than half the attempts, a trial that passes
the internal mobile-atom distance check triggers a switch of `reflect` for
subsequent attempts. If the attempt limit is exhausted, no offspring is returned.

The population and reference builder supply the minimum bond distances,
number of mobile atoms, and random-number stream automatically.

## Fragment preservation

Mirror offspring retain the parent's tags at the same atom indices, including
on mobile atoms. Retaining these labels does not preserve the geometry of atoms
sharing a tag: the mutation still moves and replaces atoms individually.

`mirror` currently operates on individual atoms and cannot preserve molecular
fragments. With `system.preserve_fragments: true` (the default), the GA rejects
this mutation during operator setup with an error stating that `MirrorMutation`
cannot guarantee fragment preservation. This also applies when every tagged
fragment contains only one atom: the check is based on the operator's declared
capability, rather than the input's fragment sizes.

There is currently no configuration option to force `mirror` while keeping
`system.preserve_fragments: true`. Setting `use_tags: false` on the mutation is
not an override: `use_tags` is a system-owned setting and is rejected in operator
configuration. The correct spelling of the system setting is
`preserve_fragments`.

For an atom-wise search, set `system.preserve_fragments: false` as shown above.
This disables fragment preservation for all genetic operators in the search,
not only `mirror`. For a molecular search that must retain fragment geometry,
keep it enabled and use a compatible mutation such as
{ref}`rattle <ga-rattle-mutation>` or {ref}`swap <ga-swap-mutation>`.
Supporting mirror mutation with fragment preservation requires a
fragment-aware implementation; it cannot be enabled by changing the YAML alone.
