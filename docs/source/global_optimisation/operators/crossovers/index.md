(ga-crossovers)=

# crossovers

Crossovers create an offspring from two selected parents.

| Method | Description |
| --- | --- |
| `cut_and_splice` | Divides two parents with a random plane and joins material from opposite sides. It supports fixed or variable cells and can preserve tagged molecular fragments. |

`cut_and_splice` is the only supported crossover method, including for isolated
clusters, supported structures, and bulk systems.

Crossover selects the `active` group from `system.search_space` by default,
excluding frozen atoms. Set the crossover's `group` expression to override it,
for example `group: "active or buffer"`. Atoms outside the selected group are
retained from the first parent. No `n_top` setting is required or accepted.

Without an explicit search space, active atoms have nonzero tags. Fragment
identity remains controlled by `system.preserve_fragments`: when enabled, a
selection must include complete tagged fragments. The standard rattle and mirror
mutations use the same group-selection policy.

For isolated clusters, always set `system.periodic: true` and
`system.preserve_fragments: true`, using a periodic cell with sufficient
vacuum. Give each independent atom its own positive tag; atoms in a molecular
fragment share a positive tag. This preserves fragment identity without grouping
atoms by their parent of origin.
