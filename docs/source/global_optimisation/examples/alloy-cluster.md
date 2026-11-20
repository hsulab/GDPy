(ga-alloy-cluster-example)=
(bh-composition-broadcast-example)=
(bh-variable-composition-example)=

# copper–nickel cluster

## system

Explore chemical ordering and composition in Cu–Ni clusters within a 20 Å
periodic vacuum cell. The baseline below builds Cu₇Ni₆ in a spherical region,
with a separate tag for each atom. The other inputs use fixed Cu₆Ni₂/Cu₄Ni₄
compositions or vary the number of Ni atoms around six Cu atoms.

```{literalinclude} ../../../../examples/global_optimisation/genetic_algorithm/cu7ni6.yaml
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

- **Chemical ordering — {doc}`genetic algorithm <../genetic-algorithm>`:**
  1 generation with 2 offspring, using only Cu/Ni `swap` mutations at fixed Cu₇Ni₆ composition.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/genetic_algorithm/cu7ni6.yaml).
- **Composition sweep — {doc}`basin hopping <../basin_hopping>`:** `broadcast`
  runs independent fixed-composition Cu₆Ni₂ and Cu₄Ni₄ searches. Each runs
  1 generation with 2 chains of 2 attempted `move` steps at 500 K.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/basin_hopping/cu_ni_compositions.yaml).
- **Variable composition — basin hopping:** 1 generation with 2 chains of
  8 attempted steps at 1000 K, mixing Ni insertion/removal (`exchange`) and
  atomic moves. Initialization uses 1–3 Ni atoms; these bounds do not restrict
  later exchanges, while the six Cu atoms remain fixed in number.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/basin_hopping/cu6_nix.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `genetic_algorithm` | `cu7ni6.yaml` | `../runtimes/emt.yaml` | `run-cu7ni6-emt` |
| `basin_hopping` | `cu_ni_compositions.yaml` | `../runtimes/emt.yaml` | `run-cu-ni-compositions` |
| `basin_hopping` | `cu6_nix.yaml` | `../runtimes/emt.yaml` | `run-cu6-nix` |

## results

Inspect `results/all_candidates.xyz` and the energy history for relaxed
structures. The composition sweep writes separate searches under `expo.00/`
and `expo.01/`, with shared scheduler records under `_meta/`; preserve broadcast
values and order when resuming.

For variable composition, compare candidate formulas and parent IDs in
`candidates.db` with chain decisions in `tmp_folder/gen1/rounds/events.jsonl`.
Formation-energy ranking uses illustrative reservoirs μNi = −0.5 eV and
μCu = 0 eV. These short EMT searches demonstrate workflows rather than establish
Cu–Ni equilibrium stability.
