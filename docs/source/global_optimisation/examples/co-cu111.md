(co-cu111-example)=

# carbon monoxide on copper

## system

Search positions and orientations of one CO molecule on a three-layer
Cu(111)-p(2×2) slab. The 5.11 × 4.43 Å surface cell has a height of 24 Å.
The bottom four Cu atoms are fixed; the remaining atoms and CO are mobile.

```{literalinclude} ../../../../examples/global_optimisation/simulated_annealing/co_cu111.yaml
:language: yaml
:caption: system
:start-at: system:
:end-before: strategy:
```

## runtime

- {doc}`DPA4C / DeepMD <../../potentials/deepmd>` ({download}`dpa4c_mini_sa.yaml <../../../../examples/global_optimisation/runtimes/dpa4c_mini_sa.yaml>`): NVT dynamics,
  `timestep: 0.5` fs and `steps: 1000` per trajectory.
  Temperature follows the cooling schedule below; pressure is not controlled. Fixed atoms: zbot 4.

Annealing endpoints are minimised with the same potential using
`fmax: 0.05` eV/Å and `steps: 1000`.

See the linked potential guide for installation and model setup. Check force
convergence rather than assuming the step limit reaches a minimum.

## methods

- {doc}`simulated annealing <../simulated-annealing>`: 3 generations with 4 trajectories each, cooling
  through 600 → 100, 450 → 100, 300 → 100 K; minimise the selected endpoints.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/simulated_annealing/co_cu111.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `simulated_annealing` | `co_cu111.yaml` | `../runtimes/dpa4c_mini_sa.yaml` | `run-co-cu111-sa` |

## results

Inspect `results/all_candidates.xyz` for adsorption geometry, desorption, or
dissociation, and `results/pop.png` for energy history. CO is flexible during
dynamics and minimisation. Total energies rank these fixed-composition
candidates; they are not adsorption energies relative to a clean slab and gas-phase CO.
