(cu8-example)=

# Cu<sub>8</sub> cluster

Search for low-energy structures of an isolated eight-atom copper cluster with
ASE's Effective Medium Theory (EMT) potential. A 20 × 20 × 20 Å periodic vacuum
cell separates neighbouring cluster images. No model download is needed.

GA and BH generate four initial structures inside a sphere, using matching
builder settings and random seed. Each Cu atom has a distinct positive tag,
with `system.periodic: true` and `system.preserve_fragments: true`, so individual
atoms can move and recombine. SA instead reads two starting structures from the
shared seed file and does not preserve fragments. The search budgets below are
short workflow demonstrations, not converged global-minimum searches.

## Local relaxation runtime for GA and BH

Both methods relax initial structures and new candidates with this runtime:

```{literalinclude} ../../../../examples/global_optimisation/runtimes/emt.yaml
:language: yaml
```

SA uses an MD runtime for cooling, followed by endpoint minimisation, as shown
in its section below.

(ga-cluster-example)=

## Genetic algorithm

The search creates two offspring in one generation using interatomic-distance
comparison, `cut_and_splice` crossover, and rattle mutation.

### Input

The complete example is available at
`examples/global_optimisation/explorations/genetic_algorithm/cu8.yaml`:

```{literalinclude} ../../../../examples/global_optimisation/explorations/genetic_algorithm/cu8.yaml
:language: yaml
```

### Run

Run from the example folder indicated below:

```shell
# Working directory: examples/global_optimisation/explorations/genetic_algorithm
gdp -d ./run-cu8-emt \
    --runtime ../../runtimes/emt.yaml explore \
    ./cu8.yaml
```

The search is stored under `run-cu8-emt`. When it completes,
`results/all_candidates.xyz` contains the relaxed candidates ordered by the GA
score, while `results/pop.png` summarises the population energies by
generation. The example produces six relaxed Cu8 candidates: four initial
structures and two offspring.

:::{note}
EMT is inexpensive and convenient for demonstrating the workflow, but this
small population and single-generation search are not sufficient for a
scientific Cu8 global optimisation.
:::

(bh-cu8-example)=

## Basin hopping

### Input

The exploration input is `examples/global_optimisation/explorations/basin_hopping/cu8.yaml`:

```{literalinclude} ../../../../examples/global_optimisation/explorations/basin_hopping/cu8.yaml
:language: yaml
```

The search selects starts once and launches two independent chains from the retained
pool of up to two distinct candidates. Starts are sampled with replacement,
so both chains may start from the same candidate. Each chain runs ten displacement proposals, with a maximum displacement of 0.8 Å.
Both chains propose once per round, and the calculation worker minimizes the
batch of valid trials before their energies are used for acceptance.
The operator's 500 K temperature controls uphill acceptance during the search;
it is not an MD thermostat or a claim of thermal equilibrium sampling.

The separate runtime relaxes the initial population and every valid trial
using EMT and a force tolerance of 0.05 eV/Å. Every relaxed trial endpoint is stored, whether accepted or rejected,
without another relaxation. The search adds up to twenty trial minima, for at
most 24 evaluated structures including initialization. The omitted `strategy.convergence`
uses the default final generation of 1; generation 0 is initialization. Stored
discoveries do not replace chain states: MC acceptance determines each next state.

### Run

Use the working directory indicated with each command:

```shell
# Working directory: examples/global_optimisation/explorations/basin_hopping
gdp -d ./run-cu8-bh-emt \
    --runtime ../../runtimes/emt.yaml explore \
    ./cu8.yaml
```

Results are written under `run-cu8-bh-emt`:

- `results/all_candidates.xyz`: all evaluated minima (including rejected trials), ordered by score
  with the lowest-energy candidate first for the default energy objective.
- `results/pop.png`: candidate energies by generation.
- `tmp_folder/gen*/rounds/events.jsonl`: chain decisions and references to
  structures in `candidates.db`. Trajectory files are not written automatically.

Export each chain's starting structure and accepted hops when needed:

```python
from gdpx.exploration.basin_hopping import export_trajectories

export_trajectories(
    "run-cu8-bh-emt/tmp_folder/gen1/rounds",
    "cu8-trajectories",
)
```

This writes `mc-0000.xyz`, `mc-0001.xyz`, and so on, overwriting previous exports
of those chains. Rejected trials are omitted. Extinction replacements are marked
as restart events; this demo has no extinction rules. Export works during a run
up to its last committed round and after generation finalization.

The small population and ten-move chains keep this example short. Increase
`steps_per_chain` for longer chains, or adjust the initial population and number of
chains for a broader search. Setting `strategy.convergence.generation` above 1 additionally
reselects chain starts from the accumulated minima between search generations.
The demonstration does not establish the global minimum of Cu₈.

(sa-cu8-example)=

## Simulated annealing

```{literalinclude} ../../../../examples/global_optimisation/explorations/simulated_annealing/cu8.yaml
:language: yaml
```

The calculation runtime is supplied separately:

```{literalinclude} ../../../../examples/global_optimisation/runtimes/emt_sa.yaml
:language: yaml
```

Run from the example folder indicated below:

```shell
# Working directory: examples/global_optimisation/explorations/simulated_annealing
gdp -d run-sa \
  --runtime ../../runtimes/emt_sa.yaml \
  explore ./cu8.yaml
```

The example minimises two initial structures and runs three generations of four
MD trajectories cooling from 1200 to 300 K, 900 to 200 K, and 600 to 100 K
respectively. Each trajectory is
followed by minimisation. It retains at most two distinct minima after each generation.
