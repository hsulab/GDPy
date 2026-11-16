(ga-supported-nanoparticle-example)=

# copper cluster on alumina

## system

Search Cu₄ clusters on α-Al₂O₃(0001). The 180-atom support has a
14.28 × 16.49 × 34.0 Å cell. Initial Cu atoms occupy a sphere of radius 1.8 Å
centred at (7.1385, 8.2428, 10.0) Å. Support atoms have tag 0; each Cu atom
has a distinct positive tag.

```{literalinclude} ../../../../examples/global_optimisation/explorations/genetic_algorithm/cu4_alumina111.yaml
:language: yaml
:caption: system
:start-at: system:
:end-before: strategy:
```

This block shows the first method's system settings. The linked inputs below
contain each method's builders and population sizes.

## runtime

For local relaxation, choose a suitable {doc}`potential <../../potentials/index>`
and create `runtime.yaml` with an ASE minimizer: `fmax: 0.05` eV/Å and
`steps: 20`, fixing the bottom 60 support atoms (`zbot 60`).

- {doc}`Prophet <../../potentials/prophet>` ({download}`prophet_oame_mbd.yaml <../../../../examples/global_optimisation/runtimes/prophet_oame_mbd.yaml>`): minimisation with
  `fmax: 0.05` eV/Å and `steps: 20`. Fixed atoms: zbot 60.

- {doc}`DPA4C / DeepMD <../../potentials/deepmd>` ({download}`dpa4c_mini_cu4_alumina_sa.yaml <../../../../examples/global_optimisation/runtimes/dpa4c_mini_cu4_alumina_sa.yaml>`): NVT dynamics,
  `timestep: 0.5` fs and `steps: 200` per trajectory.
  Temperature follows the cooling schedule below; pressure is not controlled. Fixed atoms: zbot 60.

Annealing endpoints are minimised with the same potential using
`fmax: 0.1` eV/Å and `steps: 20`.

See the linked potential guide for installation and model setup. Check force
convergence rather than assuming the step limit reaches a minimum.

(bh-supported-nanoparticle-example)=
(sa-supported-nanoparticle-example)=

## methods

- {doc}`genetic algorithm <../genetic-algorithm>`: 1 generation after initialization, with 2 offspring; `cut_and_splice` crossover and `rattle` mutation.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/genetic_algorithm/cu4_alumina111.yaml).
- {doc}`basin hopping <../basin_hopping>`: 1 generation after initialization, with 2 chains of
  8 attempted steps using `move` at 500 K acceptance temperature.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/basin_hopping/cu4_alumina111.yaml).
- {doc}`simulated annealing <../simulated-annealing>`: 2 generations with 4 trajectories each, cooling
  through 600 → 100, 450 → 100 K; minimise the selected endpoints.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/simulated_annealing/cu4_alumina111.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `genetic_algorithm` | `cu4_alumina111.yaml` | `runtime.yaml` | `run-cu4-alumina111-ga` |
| `basin_hopping` | `cu4_alumina111.yaml` | `runtime.yaml` | `run-cu4-alumina111-bh` |
| `basin_hopping` | `cu4_alumina111.yaml` | `../../runtimes/prophet_oame_mbd.yaml` | `run-cu4-alumina111-bh-prophet` |
| `simulated_annealing` | `cu4_alumina111.yaml` | `../../runtimes/dpa4c_mini_cu4_alumina_sa.yaml` | `run-cu4-alumina111-sa` |

## results

Inspect `results/all_candidates.xyz` and `results/pop.png` for candidate
structures and energies. Hopping histories are under `results/history/`;
annealing endpoint provenance is recorded in `gen*/sel/selection.json`.
The 20-step relaxation limits may leave unconverged structures; validate the
potential and increase both relaxation and search budgets for production.
