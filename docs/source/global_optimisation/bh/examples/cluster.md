(bh-cu8-example)=

# Gas-phase Cu₈

This example searches for low-energy structures of an isolated eight-atom
copper cluster using population-based basin hopping and ASE's EMT potential.
It generates its initial structures automatically; no structure file or model
download is needed.

## Input

The complete input is `examples/global_optimisation/cu8_bh_emt.yaml`:

```{literalinclude} ../../../../../examples/global_optimisation/cu8_bh_emt.yaml
:language: yaml
```

`population.periodic: false` makes the cluster nonperiodic. The 12 Å box supplies a coordinate
frame for generation and move selection; it does not create periodic images.
Four initial Cu₈ candidates are generated in a sphere and relaxed. Individual
atom tags let the move operator select one Cu atom at a time.

Each subsequent generation launches two independent chains from the retained
pool of up to two distinct candidates. Starts are sampled with replacement,
so both chains may start from the same candidate. Each chain runs three displacement proposals, with a maximum displacement of 0.8 Å.
The `mcworker` minimizes each trial before its energy is used for acceptance.
The operator's 500 K temperature controls uphill acceptance during the search;
it is not an MD thermostat or a claim of thermal equilibrium sampling.

The top-level `runtime` relaxes the initial population and the endpoints of
the hopping chains. Both evaluation stages use EMT and a force tolerance of
0.05 eV/Å. The example stops after generation 2; generation 0 is initialization.

## Run

From the repository root:

```shell
gdp -d ./run-cu8-bh-emt explore \
    ./examples/global_optimisation/cu8_bh_emt.yaml
```

Results are written under `run-cu8-bh-emt/expedition-0`:

- `results/all_candidates.xyz`: relaxed population candidates, ordered by score
  with the lowest-energy candidate first for the default energy objective.
- `results/pop.png`: candidate energies by generation.
- `tmp_folder/gen*/mctrajs/mc-*.xyz`: each chain's starting structure and accepted
  hops; rejected trials are not appended.

The small population and short chains keep this example quick to run. Increase
`population.initial.total_size`, `population.retained_size`,
`population.generation.total_size`, `num_mcmoves`, and
`convergence.generation` for a more extensive search. The demonstration does
not establish the global minimum of Cu₈.
