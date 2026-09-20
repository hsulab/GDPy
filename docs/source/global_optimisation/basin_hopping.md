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
- `mcworker`: complete runtime for evaluating moves in those chains.
- `convergence.generation`: final generation number.
- `objective`: energy or formation-energy ranking, with chemical potentials
  for the latter.

For example, the following recipe uses an existing tagged Cu cluster file.
Its chain worker minimizes trial structures. Supply the population-evaluation
runtime separately using the usual exploration runtime configuration.

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
  mcworker:
    potential:
      provider: emt
      parameters: {}
    executor:
      provider: ase
      method: min
      parameters:
        steps: 200
        fmax: 0.05
    options:
      worker: single
  convergence:
    generation: 10
```

Each movable atom needs its own tag; atoms sharing a tag are treated as one
particle. For the default automatic region, provide a nonzero simulation cell.

Moves temporarily borrow and edit a candidate. Local moves record only the
affected coordinates or properties for rollback; deletion records the removed
rows and their original indices. Execution owns the relaxed result. A rejection
restores the candidate without trying to reverse its relaxation.

BH owns population selection and search objectives. Shared acceptance formulas
and biased proposals do not imply that its population is an equilibrium sample.
