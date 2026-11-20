(bh-cuox-tace-example)=
(bh-cuox-thanos-example)=

# copper oxide cluster

## system

Search Cu₄O₄ clusters in a 20 Å periodic vacuum cell, with all eight atoms
moving independently. The baseline has no extinction filter; a second input
uses `thanos` to exclude relaxed structures with O–O contacts below 1.6 Å.
This filters evaluated results, rather than preventing bonds during construction
or relaxation.

```{literalinclude} ../../../../../examples/global_optimisation/basin_hopping/cu4o4.yaml
:language: yaml
:caption: system
:start-at: system:
:end-before: strategy:
```

## runtime

Both runtimes minimise with `fmax: 0.05` eV/Å and `steps: 20`:

- Baseline: {doc}`TACE <../../../potentials/tace>`
  ({download}`tace.yaml <../../../../../examples/global_optimisation/runtimes/tace.yaml>`).
- Extinction variant: {doc}`MatterSim <../../../potentials/mattersim>`
  ({download}`mattersim.yaml <../../../../../examples/global_optimisation/runtimes/mattersim.yaml>`).

See the potential guides for setup, and check force convergence; 20 steps may
not reach a minimum. The variants use different potentials, so their energies
are not directly comparable.

## methods

Both {doc}`basin-hopping <../../basin_hopping>` inputs run 1 generation after
initialization, with 2 chains of 8 attempted steps:

- **Baseline:** atomic `move` proposals at 500 K acceptance temperature.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/basin_hopping/cu4o4.yaml).
- **Extinction filtering:** `rattle` proposals at 100000 K acceptance temperature
  to demonstrate extinction and chain replacement. This is a demonstration
  setting, not a physical temperature.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/basin_hopping/cu4o4_thanos.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `basin_hopping` | `cu4o4.yaml` | `../runtimes/tace.yaml` | `run-cu4o4-bh-tace` |
| `basin_hopping` | `cu4o4_thanos.yaml` | `../runtimes/mattersim.yaml` | `run-cu4o4-bh-thanos` |

## results

Inspect `candidates.db`, `results/history/gen0001.png`, and
`tmp_folder/gen1/rounds/events.jsonl` for minima and chain decisions.

For the extinction variant, use database flags and chain events to distinguish
excluded initial candidates, rejected trials, and accepted trials that trigger
replacement.
An accepted extinct trial followed by a replacement indicates an extinction
and restart. The previously validated sequence used 150 relaxation steps;
the current 20-step demonstration may not reproduce those events.
