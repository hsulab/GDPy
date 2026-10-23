(simulated-annealing)=

# simulated annealing (sa)

Simulated annealing is a population exploration run with `gdp explore`.
It combines MD cooling over a user-selected temperature range for each generation with
minimisation and selection of diverse low-energy structures.

## Search cycle

Generation 0 builds and minimises the initial structures. The shared
{doc}`population <population>` ranks them by energy (or formation energy) and
removes duplicates using the configured comparator.

For each subsequent generation:

1. Choose `system.generation.total_size` parents from the retained population
   using fitness weights. Selection is without replacement when enough parents
   exist, and with replacement otherwise.
2. Start independent MD trajectories with fresh velocities. Run all candidates
   as a batch, cooling linearly from start to end in
   `strategy.temperatures[generation - 1]`.
3. Minimise each final structure with the same potential.
4. Store every endpoint minimum, refresh the retained population from all
   generations, and repeat.

The retained population has at most `system.retained_size` distinct candidates.
Older minima remain eligible, so an unsuccessful generation does not discard
the best structures. This initial version uses supplied builders for generation
0 only; later generations restart from selected minima.

## Examples

### Complete minimal input

This self-contained Cu₂ example uses ASE's EMT potential, so no model download
is needed. Save the following three files in the same directory. It minimises
one starting structure, runs one 100-step MD trajectory cooling from 600 to
300 K, and minimises the endpoint. This short run demonstrates the workflow;
it is not a converged global-minimum search.

`input.xyz`:

```text
2
Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 pbc="T T T"
Cu 4.0 5.0 5.0
Cu 6.4 5.0 5.0
```

`sa.yaml`:

```yaml
method: global_optimisation
random_seed: 7
system:
  periodic: true
  preserve_fragments: false
  retained_size: 1
  builders:
    random:
      method: read_stru
      fname: ./input.xyz
  initial:
    total_size: 1
    builder_allocations:
      - builder: random
        size: 1
  generation:
    total_size: 1
strategy:
  method: simulated_annealing
  temperatures:
    - [600, 300]
  convergence:
    generation: 1
  minimisation:
    provider: ase
    method: min
    parameters:
      stop:
        steps: 1000
        fmax: 0.05
```

`runtime.yaml`:

```yaml
potential:
  provider: emt
executor:
  provider: ase
  method: md
  parameters:
    ensemble:
      method: nvt
      temperature: 600.0
    setup:
      timestep: 1.0
      velocities:
        initialize: if_missing
        remove_rotation: false
      regulator:
        name: berendsen
        parameters:
          Tdamp: 100.0
    stop:
      steps: 100
    output:
      trajectory:
        period: 10
```

Run from the directory containing these files:

```shell
gdp -d ./run-sa --runtime ./runtime.yaml explore ./sa.yaml
```

The two evaluated minima are written to `run-sa/results/all_candidates.xyz`,
and the retained population contains at most one distinct minimum. Rerun the
same command to resume the search.

### Larger examples

The central {doc}`examples catalogue <examples/index>` groups recipes by physical
system. See {ref}`Cu₈ annealing <sa-cu8-example>` for an EMT demonstration and
{ref}`CO adsorption on Cu(111) <co-cu111-example>` for a DPA4C surface search.

## Configuration and parallel execution

Use `method: global_optimisation` and `strategy.method: simulated_annealing`.
The `system` configuration is shared with GA and BH.

```yaml
strategy:
  method: simulated_annealing
  temperatures:
    - [1200, 300]
    - [900, 200]
    - [600, 100]
  minimisation:
    provider: ase
    method: min
    parameters:
      stop:
        steps: 1000
        fmax: 0.05
  convergence:
    generation: 3
```

Generation 0 builds and minimises initial structures without MD. Generations
1, 2, and 3 cool from 1200 to 300 K, 900 to 200 K, and 600 to 100 K,
respectively, followed by minimisation. Each entry is a `[start, end]` pair
with finite temperatures satisfying `start > end > 0`. The list must contain
exactly one cooling range per MD generation; the generation limit defaults to
the list length. An empty list with generation 0 performs only initial
minimisation. Ranges may repeat or start hotter than the previous generation;
each generation is an independent annealing attempt from selected minima.

The separate runtime supplies one potential, an ordinary `md` executor, its
scheduler, and dispatch settings. The exploration sets both temperature endpoints
for each generation without modifying the source runtime. Use thermostatted
NVT or NPT dynamics with a driver and thermostat that support temperature ramps.
Omit runtime `tend`: the exploration supplies it from each pair. The MD
`stop.steps` controls the number of integration steps over which cooling occurs.

`strategy.minimisation` defines a complete minimisation executor with its own
provider and parameters, plus an optional scheduler (direct by default).
It inherits the MD runtime's fixed potential and dispatch settings. Potential
and modifier overrides are forbidden. Positive minimisation `stop.steps` and
`stop.fmax` are required. SA retains the existing restriction against modifiers.
There is no `native/simulated_annealing` compute executor.

The exploration owns `convergence.generation`, `objective`, `use_archive`, and
the temperature schedule. The exploration scheduler controls the population
loop, while the MD and minimisation schedulers control their respective jobs.
Every generation starts with fresh velocities and persisted independent seeds
for each candidate. All candidates finish MD and minimisation before selection
for the next generation.

Failed MD or minimisation candidates are recorded and excluded from selection;
they are not retried or switched to another optimiser. Successful candidates in
the same generation still enter the population. If a later generation has no
successful candidates, the retained minima from earlier generations remain
available for the next generation. If no eligible minima exist at all (for
example, every initial minimisation fails), the search stops with an explicit
error. Failure details are saved in `population.json` and, for new failures,
`candN/gdp-failure.json`; restart preserves the skip decision. Incomplete queued
jobs remain pending and are not treated as failed candidates.

## Results and restart

- `candidates.db`: all evaluated minima, including duplicate and unretained
  candidates; records include generation, parent IDs, and replica index.
- `population.json`: generation plans, parent selection, retained IDs,
  configuration fingerprint, and random streams.
- `results/all_candidates.xyz`: all evaluated endpoint minima, including duplicates
  and candidates not retained.
- `results/pop.png`: candidate objective versus generation, with parent–child
  links, the best evaluated objective so far, and the final retained population
  highlighted. Generation 0 contains the minimised initial structures.
- `gen0/min/`: initial evaluations.
- `genN/sim/`: MD cooling trajectories over generation N’s temperature range.
- `genN/min/`: endpoint minimisations on the same potential.
- `genN/inputs.xyz`: an extended XYZ snapshot of the generation's starting
  structures, not its MD trajectory. Generation 0 stores builder outputs; later
  generations store selected parent structures, with old momenta removed so that
  each annealing attempt starts with fresh velocities.

Each generation also snapshots its inputs. Restart with the same configuration
and directory: selected parents and random seeds are reused, queued calculations
are polled, and already recorded endpoints are not duplicated.
Keep `inputs.xyz` when restarting: it avoids rebuilding or reselecting the inputs.
Old population SA inputs and checkpoints require a fresh directory.
Changes to the configuration require a fresh directory.
Rerunning a completed search regenerates the final results from the database
without repeating simulations. Existing `population.xyz` files from older runs
are left untouched; new runs do not export this file.
`get_workers()` exposes cooling and minimisation workers for downstream analysis.

The earlier `method: simulated_annealing` recipe remains available as a
low-level cooling runner, accepting either `annealing` or the explicit
`temperatures` list of constant-temperature holds. Use the population configuration above for structure
search. These outputs are not compatible with the earlier experimental BH-based
SA configuration.
