(ga-variable-surface-oxide-example)=

# copper surface oxide: variable composition

## system

Search CuₓOᵧ overlayers on a two-layer Cu(111) slab, with both x and y between
1 and 4. Substrate atoms have tag 0; added atoms receive distinct positive tags.
The initial search region spans z = 4.60–9.10 Å over the 5.11 × 4.43 Å surface cell.

```{literalinclude} ../../../../examples/global_optimisation/explorations/genetic_algorithm/cuxoy_cu111.yaml
:language: yaml
:caption: system
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

- {doc}`genetic algorithm <../genetic-algorithm>`: 1 generation after initialization, with 4 offspring; mutation only, using `exchange`, `rattle` mutation.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/genetic_algorithm/cuxoy_cu111.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `genetic_algorithm` | `cuxoy_cu111.yaml` | `../../runtimes/mattersim.yaml` | `run-cuxoy-cu111` |

## results

Candidates are ranked by `E − N_Cu μ_Cu − N_O μ_O`; the supplied chemical
potentials are illustrative. Inspect `results/all_candidates.xyz` and
`results/pop.png`, and use reservoir references consistent with your potential
before comparing compositions scientifically.
