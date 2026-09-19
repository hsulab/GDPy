(ga-crossovers)=

# Crossovers

Crossovers create an offspring from two selected parents.

| Method | Description |
| --- | --- |
| `periodic_cut_and_splice` | Divides two parents with a random plane and joins material from opposite sides. It supports fixed or variable cells and can preserve tagged molecular fragments. |
| `cluster_cut_and_splice` | Applies cut-and-splice crossover to isolated particles or clusters. It preserves composition by default and separates halves when atoms would otherwise be too close. |

Use `periodic_cut_and_splice` for supported structures and periodic systems. Use
`cluster_cut_and_splice` for free clusters where there is no substrate.
