(bh-seeded-cu8-example)=

# copper cluster: seeded initialization

## system

Start a Cu₈ search from two random structures and two supplied seed structures
in a 20 Å periodic vacuum cell. The seeds are a perturbed cube and square
antiprism; their existing energies are discarded and both are evaluated again.
Keep composition, cell, periodicity, and tags consistent when supplying your own seeds.

```{literalinclude} ../../../../../examples/global_optimisation/explorations/basin_hopping/cu8_seeded.yaml
:language: yaml
:caption: system
:start-at: system:
:end-before: strategy:
```

## runtime

- {doc}`EMT <../../../potentials/emt>` ({download}`emt.yaml <../../../../../examples/global_optimisation/runtimes/emt.yaml>`): minimisation with
  `fmax: 0.05` eV/Å and `steps: 1000`.

See the linked potential guide for installation and model setup. Check force
convergence rather than assuming the step limit reaches a minimum.

## methods

- {doc}`basin hopping <../../basin_hopping>`: 1 generation after initialization, with 2 chains of
  5 attempted steps using `move` at 500 K acceptance temperature.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/basin_hopping/cu8_seeded.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `basin_hopping` | `cu8_seeded.yaml` | `../../runtimes/emt.yaml` | `run-cu8-seeded-bh-emt` |

## results

Generation 0 in `candidates.db` records two candidates from `init` and two from
`seeds`. Similar relaxed structures can be deduplicated, so providing a seed
does not guarantee its selection. Use a fresh output directory when changing
the seed file or its allocations.
