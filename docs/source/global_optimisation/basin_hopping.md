# Basin hopping

`method: basin_hopping` runs the population-based search formerly named
`concurrent_hopping`. It evolves a population through short hopping chains,
evaluates candidates using execution workers, and ranks them using the search
objective. It inherits directly from `BaseExpedition`, independently of MC.

```{toctree}
:maxdepth: 1

bh/examples/cluster
```

See the shared {ref}`global-optimisation-population` reference for population
configuration. The recipe uses these settings:

- `population`: `retained_size`, named `builders`, `initial` allocations,
  `generation.total_size`, comparator, and extinction settings.
- `operators`: weighted move configurations shared with MC.
- `num_mcmoves`: number of proposals per candidate chain.
- `convergence.generation`: final generation number.
- `objective`: energy or formation-energy ranking, with chemical potentials
  for the latter.

For example, the following recipe uses an existing tagged Cu cluster file.
The calculation runtime minimizes the initial structures and each valid trial
batch before acceptance.

```yaml
method: basin_hopping
recipe:
  random_seed: 7
  population:
    retained_size: 1
    periodic: false
    builders:
      random:
        method: read_stru
        fname: ./cu-cluster.xyz
    initial:
      total_size: 1
      builder_allocations:
        - builder: random
          size: 1
    generation:
      total_size: 1
  operators:
    - method: move
      particles: [Cu]
      temperature: 500
      max_disp: 0.5
      probability: 1.0
      skip_distance_check: true
  num_mcmoves: 5
  convergence:
    generation: 10
runtime:
  potential:
    provider: emt
    parameters: {}
  executor:
    provider: ase
    method: min
    parameters:
      steps: 200
      fmax: 0.05
```

Each movable atom needs its own tag; atoms sharing a tag are treated as one
particle. For the default automatic region, provide a nonzero simulation cell.

Moves temporarily borrow and edit a candidate. Local moves record only the
affected coordinates or properties for rollback; deletion records the removed
rows and their original indices. Execution owns the relaxed result. A rejection
restores the candidate without trying to reverse its relaxation.

BH owns population selection and search objectives. Shared acceptance formulas
and biased proposals do not imply that its population is an equilibrium sample.

## Batched rounds and execution

BH advances chains in rounds: one proposal per chain, one batch of valid trials,
then acceptance after every trial result is available. Invalid proposals consume
a hop without evaluation. Rejected trials retain the previous accepted minimum.
Every evaluated minimum enters the candidate database, including rejected
trials and accepted intermediate minima. Acceptance controls the chain's next
state; it does not control whether the result is stored. The next generation
selects from all eligible stored minima using the shared population comparator
and ranking. Repeated evaluations remain separate history records, while the
retained population removes similar structures. No extra candidate or relaxation
is created for a final chain endpoint.

Each trial record includes its chain, round, acceptance decision, originating
accepted candidate (`data.parents`), and starting parent. Invalid proposals have
no evaluated result and create no candidate record. Results excluded by extinction
rules remain stored but are ineligible for selection.

Top-level `scheduler` places the exploration loop. Top-level `runtime` defines
its calculation worker; `runtime.scheduler` places the expensive calculations.
Both use the existing direct, queue, and transport configuration. No driver is
accessed by BH. With a minimisation runtime, every valid trial is minimized;
a single-point runtime omits this step and is not conventional basin hopping.

Round checkpoints under `tmp_folder/gen*/rounds` persist pending trial inputs,
acceptance context, accepted structures, and RNG state. Restart with the same
recipe and directory to resume queued work. Results are matched by trial ID,
and acceptance uses chain order independently of result arrival order.

Remove the former `recipe.mcworker` and move its calculation settings into
`runtime`. Older serial-chain and endpoint-only round checkpoints cannot resume an
in-progress generation with this implementation. Completed history remains
readable; minima from older worker outputs are not backfilled automatically. Round ordering also changes trajectories for old
random seeds; new runs are reproducible across restart boundaries.
