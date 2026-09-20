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
The `random_structure_improved` builder generates four initial Cu₈ candidates
in a sphere, which are then relaxed. Individual
atom tags let the move operator select one Cu atom at a time.

The search selects starts once and launches two independent chains from the retained
pool of up to two distinct candidates. Starts are sampled with replacement,
so both chains may start from the same candidate. Each chain runs ten displacement proposals, with a maximum displacement of 0.8 Å.
Both chains propose once per round, and the calculation worker minimizes the
batch of valid trials before their energies are used for acceptance.
The operator's 500 K temperature controls uphill acceptance during the search;
it is not an MD thermostat or a claim of thermal equilibrium sampling.

The top-level `runtime` relaxes the initial population and every valid trial
using EMT and a force tolerance of 0.05 eV/Å. Every relaxed trial endpoint is stored, whether accepted or rejected,
without another relaxation. The search adds up to twenty trial minima, for at
most 24 evaluated structures including initialization. The omitted `convergence`
uses the default final generation of 1; generation 0 is initialization. Stored
discoveries do not replace chain states: MC acceptance determines each next state.

## Run

From the repository root:

```shell
gdp -d ./run-cu8-bh-emt explore \
    ./examples/global_optimisation/cu8_bh_emt.yaml
```

Results are written under `run-cu8-bh-emt/expedition-0`:

- `results/all_candidates.xyz`: all evaluated minima (including rejected trials), ordered by score
  with the lowest-energy candidate first for the default energy objective.
- `results/pop.png`: candidate energies by generation.
- `tmp_folder/gen*/mctrajs/mc-*.xyz`: each chain's starting structure and accepted
  hops; rejected trials are not appended. Runs with extinction rules also mark
  replacement starts as restart events. This demo has no extinction rules.

The small population and ten-move chains keep this example short. Increase
`num_mcmoves` for longer chains, or adjust the initial population and number of
chains for a broader search. Setting `convergence.generation` above 1 additionally
reselects chain starts from the accumulated minima between search generations.
The demonstration does not establish the global minimum of Cu₈.
