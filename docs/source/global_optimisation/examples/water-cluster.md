(ga-water-cluster-example)=

# water cluster: molecular packing

## system

Search a cluster of four water molecules in a periodic vacuum cell. Each
molecule shares one tag so compatible search operators preserve its internal
geometry; subsequent relaxation can change its bonds.

```{literalinclude} ../../../../examples/global_optimisation/explorations/genetic_algorithm/water4.yaml
:language: yaml
:caption: system
:start-at: system:
:end-before: strategy:
```

## runtime

- {doc}`ReaxFF / xreac <../../potentials/reax>` ({download}`xreac.yaml <../../../../examples/global_optimisation/runtimes/xreac.yaml>`): minimisation with
  `fmax: 0.05` eV/Å and `steps: 20`.

See the linked potential guide for installation and model setup. Check force
convergence rather than assuming the step limit reaches a minimum.

## methods

- {doc}`genetic algorithm <../genetic-algorithm>`: 1 generation after initialization, with 2 offspring; `cut_and_splice` crossover and `rattle` mutation.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/genetic_algorithm/water4.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `genetic_algorithm` | `water4.yaml` | `../../runtimes/xreac.yaml` | `run-water4-reax` |

## results

Inspect `results/all_candidates.xyz` for molecular arrangements and possible
bond changes during relaxation. The small population and short relaxation budget
do not establish converged water-cluster minima.
