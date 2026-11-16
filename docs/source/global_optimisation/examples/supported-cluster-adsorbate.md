(ga-supported-cluster-adsorbate-example)=
(ga-adsorbate-insertion-example)=

# carbon monoxide on a supported copper cluster

## system

Search CO adsorption on a Cu₄ cluster supported by α-Al₂O₃(0001). Four initial
candidates come from random construction and four from adsorption-site insertion
on a prepared Cu₄/support structure. Support atoms have tag 0, Cu atoms have
individual tags, and C and O share the CO fragment tag.

```{literalinclude} ../../../../examples/global_optimisation/explorations/genetic_algorithm/cu4_co_alumina111.yaml
:language: yaml
:caption: system
:start-at: system:
:end-before: strategy:
```

## runtime

- {doc}`MatterSim <../../potentials/mattersim>` ({download}`mattersim.yaml <../../../../examples/global_optimisation/runtimes/mattersim.yaml>`): minimisation with
  `fmax: 0.05` eV/Å and `steps: 20`.

The shared runtime relaxes all atoms. For a constrained surface study, add
`zbot 60` as the executor setup constraint to fix the bottom substrate layer.

See the linked potential guide for installation and model setup. Check force
convergence rather than assuming the step limit reaches a minimum.

## methods

- {doc}`genetic algorithm <../genetic-algorithm>`: 1 generation after initialization, with 2 offspring; `cut_and_splice` crossover and `rattle` mutation.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/genetic_algorithm/cu4_co_alumina111.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `genetic_algorithm` | `cu4_co_alumina111.yaml` | `../../runtimes/mattersim.yaml` | `run-cu4-co-alumina111` |

## results

Compare the initial builder groups in `results/history/family_tree.png`;
`candidates.log` gives their scores and `tmp_folder/gen0/history.log` records
builder provenance. The comparison includes different initial Cu₄ geometries,
and site insertion is not guaranteed to produce lower energies. Check C–O
bond integrity and relaxation convergence before interpreting the ranking.
