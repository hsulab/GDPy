(bh-variable-composition-example)=

# copper–nickel cluster: variable composition

## system

Search Cu₆Niₓ clusters with nickel insertion and removal. Initialization uses
one to three Ni atoms in a 20 Å vacuum cell; these initial bounds do not restrict
later moves. Every candidate retains six Cu atoms.

```{literalinclude} ../../../../../examples/global_optimisation/explorations/basin_hopping/cu6_nix.yaml
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
  8 attempted steps using `exchange`, `move` at 1000 K acceptance temperature.
  [Complete input](https://github.com/hsulab/GDPy/blob/dev/examples/global_optimisation/explorations/basin_hopping/cu6_nix.yaml).

Follow {ref}`running the examples <go-running-examples>` with these file settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `basin_hopping` | `cu6_nix.yaml` | `../../runtimes/emt.yaml` | `run-cu6-nix` |

## results

Use candidate formulas and parent IDs in `candidates.db` to distinguish attempted
exchanges from accepted composition changes; decisions are recorded in
`tmp_folder/gen1/rounds/events.jsonl`. The example uses μNi = −0.5 eV and
μCu = 0 eV for formation-energy ranking. These are illustrative reservoir
settings, not a prediction of equilibrium composition.
