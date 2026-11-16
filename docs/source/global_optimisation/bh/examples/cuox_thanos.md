(bh-cuox-thanos-example)=

# copper oxide cluster: extinction filtering

## system

Search Cu₄O₄ clusters while excluding relaxed structures with O–O contacts
below 1.6 Å. The periodic vacuum cell is 20 Å wide. The `thanos` rule filters
evaluated results; it does not prevent O–O bonds during construction or relaxation.

```{literalinclude} ../../../../../examples/global_optimisation/explorations/basin_hopping/cu4o4_thanos.yaml
:language: yaml
:caption: system
:start-at: system:
:end-before: strategy:
```

## runtime

- {doc}`MatterSim <../../../potentials/mattersim>` ({download}`mattersim.yaml <../../../../../examples/global_optimisation/runtimes/mattersim.yaml>`): minimisation with
  `fmax: 0.05` eV/Å and `steps: 20`.

See the linked potential guide for installation and model setup. Check force
convergence rather than assuming the step limit reaches a minimum.

## methods

- {doc}`basin hopping <../../basin_hopping>`: 1 generation after initialization, with 2 chains of
  8 attempted steps using `rattle` at 100000 K acceptance temperature.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/basin_hopping/cu4o4_thanos.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `basin_hopping` | `cu4o4_thanos.yaml` | `../../runtimes/mattersim.yaml` | `run-cu4o4-bh-thanos` |

## results

Inspect the database extinction flags and chain events to distinguish excluded
initial candidates, rejected trials, and accepted trials that trigger replacement.
The verifier below requires an accepted extinct trial followed by a replacement:

```shell
# From the basin_hopping example folder, after the search.
python ../../verify_cu4o4_bh_thanos.py ./run-cu4o4-bh-thanos
```

It writes `results/oo_extinct.xyz` and `results/thanos_verification.json`. The
previously validated sequence used 150 relaxation steps; the current 20-step
demonstration may not reproduce those events or pass the verifier.
