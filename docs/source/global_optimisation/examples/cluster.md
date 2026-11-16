(cu8-example)=

# copper cluster

## system

Search an isolated Cu₈ cluster in a 20 Å periodic vacuum cell. The initial
random structures occupy a sphere of radius 3 Å; each Cu atom has its own tag.
The annealing input instead reads two seed structures and disables fragment
preservation.

```{literalinclude} ../../../../examples/global_optimisation/explorations/genetic_algorithm/cu8.yaml
:language: yaml
:caption: system
:start-at: system:
:end-before: strategy:
```

This block shows the first method's system settings. The linked inputs below
contain each method's builders and population sizes.

## runtime

- {doc}`EMT <../../potentials/emt>` ({download}`emt.yaml <../../../../examples/global_optimisation/runtimes/emt.yaml>`): minimisation with
  `fmax: 0.05` eV/Å and `steps: 1000`.

- {doc}`EMT <../../potentials/emt>` ({download}`emt_sa.yaml <../../../../examples/global_optimisation/runtimes/emt_sa.yaml>`): NVT dynamics,
  `timestep: 1.0` fs and `steps: 100` per trajectory.
  Temperature follows the cooling schedule below; pressure is not controlled.

Annealing endpoints are minimised with the same potential using
`fmax: 0.05` eV/Å and `steps: 1000`.

See the linked potential guide for installation and model setup. Check force
convergence rather than assuming the step limit reaches a minimum.

(ga-cluster-example)=
(bh-cu8-example)=
(sa-cu8-example)=

## methods

- {doc}`genetic algorithm <../genetic-algorithm>`: 1 generation after initialization, with 2 offspring; `cut_and_splice` crossover and `rattle` mutation.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/genetic_algorithm/cu8.yaml).
- {doc}`basin hopping <../basin_hopping>`: 1 generation after initialization, with 2 chains of
  10 attempted steps using `move` at 500 K acceptance temperature.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/basin_hopping/cu8.yaml).
- {doc}`simulated annealing <../simulated-annealing>`: 3 generations with 4 trajectories each, cooling
  through 1200 → 300, 900 → 200, 600 → 100 K; minimise the selected endpoints.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/simulated_annealing/cu8.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `genetic_algorithm` | `cu8.yaml` | `../../runtimes/emt.yaml` | `run-cu8-emt` |
| `basin_hopping` | `cu8.yaml` | `../../runtimes/emt.yaml` | `run-cu8-bh-emt` |
| `simulated_annealing` | `cu8.yaml` | `../../runtimes/emt_sa.yaml` | `run-sa` |

## results

Inspect `results/all_candidates.xyz` for evaluated minima and `results/pop.png`
for the energy history. For hopping chains, `tmp_folder/gen1/rounds/events.jsonl`
records decisions, including rejected trials. These short runs demonstrate the
workflow rather than establishing the global minimum.
