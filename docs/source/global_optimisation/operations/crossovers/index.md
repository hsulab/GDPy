(ga-crossovers)=

# Crossovers

Crossovers create an offspring from two selected parents.

| Method | Description |
| --- | --- |
| `cut_and_splice` | Divides two parents with a random plane and joins material from opposite sides. It supports fixed or variable cells and can preserve tagged molecular fragments. |

`cut_and_splice` is the only supported crossover method, including for isolated
clusters, supported structures, and bulk systems.

For isolated clusters, always set `population.periodic: true` and
`population.preserve_fragments: true`, using a periodic cell with sufficient
vacuum. Give each independent atom its own positive tag; atoms in a molecular
fragment share a positive tag. This preserves fragment identity without grouping
atoms by their parent of origin.
