(ga-crossovers)=

# crossovers

Crossovers create an offspring from two selected parents.

| Method | Description |
| --- | --- |
| `cut_and_splice` | Divides two parents with a random plane and joins material from opposite sides. It supports fixed or variable cells and can preserve tagged molecular fragments. |

`cut_and_splice` is the only supported crossover method, including for isolated
clusters, supported structures, and bulk systems.

Mobile atoms are selected from each parent by `tag != 0`, independently of
their positions in the atom array. Tag 0 is reserved for the substrate, which
is added back after crossover. No `n_top` setting is required or accepted.
The same tag-based mobile selection is used by the standard rattle and mirror
mutations. Turning off `system.preserve_fragments` allows independent atomic
operations but does not discard the substrate/mobile tags.

For isolated clusters, always set `system.periodic: true` and
`system.preserve_fragments: true`, using a periodic cell with sufficient
vacuum. Give each independent atom its own positive tag; atoms in a molecular
fragment share a positive tag. This preserves fragment identity without grouping
atoms by their parent of origin.
