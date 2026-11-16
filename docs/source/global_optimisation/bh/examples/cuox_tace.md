(bh-cuox-tace-example)=

# copper oxide cluster: tace potential

## system

Search Cu₄O₄ clusters in a 20 Å periodic vacuum cell. All eight atoms move
independently, and no O–O extinction filter is applied.

```{literalinclude} ../../../../../examples/global_optimisation/explorations/basin_hopping/cu4o4.yaml
:language: yaml
:caption: system
:start-at: system:
:end-before: strategy:
```

## runtime

- {doc}`TACE <../../../potentials/tace>` ({download}`tace.yaml <../../../../../examples/global_optimisation/runtimes/tace.yaml>`): minimisation with
  `fmax: 0.05` eV/Å and `steps: 20`.

See the linked potential guide for installation and model setup. Check force
convergence rather than assuming the step limit reaches a minimum.

## methods

- {doc}`basin hopping <../../basin_hopping>`: 1 generation after initialization, with 2 chains of
  8 attempted steps using `move` at 500 K acceptance temperature.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/basin_hopping/cu4o4.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `basin_hopping` | `cu4o4.yaml` | `../../runtimes/tace.yaml` | `run-cu4o4-bh-tace` |

## results

Inspect `candidates.db`, `results/history/gen0001.png`, and
`tmp_folder/gen1/rounds/events.jsonl` for minima and chain decisions. The supplied
20-step relaxation limit may not converge; an earlier 300-step validation
completed four initial structures and sixteen trials with finite energies and forces.
