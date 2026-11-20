(ga-surface-oxide-example)=
(ga-variable-surface-oxide-example)=

# copper surface oxide

## system

Search fixed Cu₄O₄ or variable CuₓOᵧ overlayers on a two-layer Cu(111) slab
containing eight substrate atoms. The 5.11 × 4.43 Å surface cell has a height of 22 Å; initial overlayer
atoms occupy z = 4.60–9.10 Å. Substrate atoms have tag 0; added atoms have
positive tags. The variable composition input allows 1–4 atoms of each element. The baseline below fixes
the overlayer at Cu₄O₄.

```{literalinclude} ../../../../examples/global_optimisation/explorations/genetic_algorithm/cu4o4_cu111.yaml
:language: yaml
:caption: examples/global_optimisation/explorations/genetic_algorithm/cu4o4_cu111.yaml
:start-at: system:
:end-before: strategy:
```

## runtime

- {doc}`MatterSim <../../potentials/mattersim>` ({download}`mattersim.yaml <../../../../examples/global_optimisation/runtimes/mattersim.yaml>`): minimisation with
  `fmax: 0.05` eV/Å and `steps: 20`.

The shared runtime relaxes all atoms. For a constrained surface study, add
`zbot 4` as the executor setup constraint to fix the bottom substrate layer.

See the linked potential guide for installation and model setup. Check force
convergence rather than assuming the step limit reaches a minimum.

## methods

Both inputs use a {doc}`genetic algorithm <../genetic-algorithm>` for 1 generation
after initialization:

- **Fixed composition:** 2 offspring using `cut_and_splice` crossover and `rattle` mutation.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/genetic_algorithm/cu4o4_cu111.yaml).
- **Variable composition:** 4 offspring using only mutations: `exchange` (70%)
  and `rattle` (30%).
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/genetic_algorithm/cuxoy_cu111.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `genetic_algorithm` | `cu4o4_cu111.yaml` | `../../runtimes/mattersim.yaml` | `run-cu4o4-cu111` |
| `genetic_algorithm` | `cuxoy_cu111.yaml` | `../../runtimes/mattersim.yaml` | `run-cuxoy-cu111` |

## results

Inspect `results/all_candidates.xyz` and `results/pop.png` for relaxed overlayers
and their ranking. Variable-composition candidates are ranked by
`E − N_Cu μ_Cu − N_O μ_O`; use reservoir references consistent with the potential
instead of the illustrative supplied values. Converge slab thickness, cell size,
relaxation, and search length before interpreting surface stability.
