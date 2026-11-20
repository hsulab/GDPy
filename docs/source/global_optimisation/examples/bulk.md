(ga-bulk-example)=

# bulk copper: crystal structure search

## system

Search periodic copper cells containing four atoms. The builder varies atomic
positions and cell shape at a fixed volume of 48 Å³.

```{literalinclude} ../../../../examples/global_optimisation/genetic_algorithm/cu4_bulk.yaml
:language: yaml
:caption: system
:start-at: system:
:end-before: strategy:
```

## runtime

- {doc}`EMT <../../potentials/emt>` ({download}`emt.yaml <../../../../examples/global_optimisation/runtimes/emt.yaml>`): minimisation with
  `fmax: 0.05` eV/Å and `steps: 1000`.

See the linked potential guide for installation and model setup. Check force
convergence rather than assuming the step limit reaches a minimum.

## methods

- {doc}`genetic algorithm <../genetic-algorithm>`: 1 generation after initialization, with 2 offspring; `cut_and_splice` crossover and `rattle`, `strain` mutation.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/genetic_algorithm/cu4_bulk.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `genetic_algorithm` | `cu4_bulk.yaml` | `../runtimes/emt.yaml` | `run-cu4-bulk-emt` |

## results

Inspect relaxed cells in `results/all_candidates.xyz` and energies in
`results/pop.png`. The example produces four initial candidates and two
offspring; increase the search budget before drawing structural conclusions.
