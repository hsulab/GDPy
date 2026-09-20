(ga-mutations)=

# Mutations

Mutations modify one candidate to create structural or compositional variation.

| Method | Description |
| --- | --- |
| `mirror` | Keeps one side of a randomly oriented cutting plane and mirrors it to replace the other side. This currently supports atomic structures only. |
| {ref}`rattle <ga-rattle-mutation>` | Randomly displaces a fraction of the optimised atoms or tagged fragments while enforcing minimum distances. |
| `soft` | Displaces the structure along a smooth low-frequency mode generated from its local geometry. |
| `strain` | Applies a random strain to the cell. It is intended for variable-cell searches and respects configured cell bounds. |
| `bounce` | Selects a tagged atom and moves it using neighbour repulsion. The move can be directionally biased and can target either mobile particles or a selected buffer group. |
| `cluster_rattle` | Finds connected clusters using the atomic graph and translates selected clusters as rigid units in random directions. |
| `cluster_rotation` | Finds graph-connected clusters and rotates selected clusters as rigid units about a fixed or random axis. |
| {ref}`exchange <ga-exchange-mutation>` | Inserts or removes atoms or molecular fragments for variable-composition searches. |
| `group_rattle` | Selects atoms from a group expression and randomly displaces either a fixed number or a fraction of them, rejecting overlaps. |
| {ref}`swap <ga-swap-mutation>` | Exchanges the positions of different tagged particle types while preserving the internal geometry of molecular fragments. |

```{toctree}
:maxdepth: 1
:titlesonly:
:hidden:

exchange
rattle
swap
```
