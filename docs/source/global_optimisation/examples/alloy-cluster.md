(ga-alloy-cluster-example)=

# copper–nickel cluster: chemical ordering

## system

Search chemical ordering in a Cu₇Ni₆ cluster. Random structures are built in a
spherical region within a 20 Å periodic vacuum cell, with a separate tag for
each atom.

```{literalinclude} ../../../../examples/global_optimisation/explorations/genetic_algorithm/cu7ni6.yaml
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

- {doc}`genetic algorithm <../genetic-algorithm>`: 1 generation after initialization, with 2 offspring; mutation only, using `swap` mutation.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/genetic_algorithm/cu7ni6.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `genetic_algorithm` | `cu7ni6.yaml` | `../../runtimes/emt.yaml` | `run-cu7ni6-emt` |

## results

Compare relaxed chemical orderings in `results/all_candidates.xyz` and the
energy history in `results/pop.png`. The small population and EMT potential
make this a workflow demonstration, not a validated Cu–Ni stability study.
