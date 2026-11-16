(bh-composition-broadcast-example)=

# copper–nickel cluster: composition sweep

## system

Run independent searches for Cu₆Ni₂ and Cu₄Ni₄ clusters using a shared setup
and seed. The top-level `broadcast` mapping in the complete input changes the
whole composition dictionary; each search preserves its assigned composition.
The system block below shows the base composition.

```{literalinclude} ../../../../../examples/global_optimisation/explorations/basin_hopping/cu_ni_compositions.yaml
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
  2 attempted steps using `move` at 500 K acceptance temperature.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/basin_hopping/cu_ni_compositions.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `basin_hopping` | `cu_ni_compositions.yaml` | `../../runtimes/emt.yaml` | `run-cu-ni-compositions` |

## results

The two searches write separate results under `expo.00/` and `expo.01/`, with
shared scheduler records under `_meta/`. Resume with the same broadcast values
and order. For one Slurm allocation, adapt and submit
`examples/global_optimisation/two_explorations.slurm` from the repository root.
