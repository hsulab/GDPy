(simulated-annealing)=

# simulated annealing (sa)

Population-based MD annealing combines parallel cooling trajectories with
selection of diverse low-energy minima. A single cooling trajectory can be run
with `gdp compute`; this exploration manages the population and repeated cycles.

## Search cycle

Generation 0 builds and minimises the initial structures. The shared
{doc}`population <population>` ranks them by energy (or formation energy) and
removes duplicates using the configured comparator.

For each subsequent generation:

1. Choose `system.generation.total_size` parents from the retained population
   using fitness weights. Selection is without replacement when enough parents
   exist, and with replacement otherwise.
2. Start independent MD trajectories with fresh velocities. Run all candidates
   as a batch, cooling each from the start to the final temperature.
3. Minimise each final structure with the same potential.
4. Store every endpoint minimum, refresh the retained population from all
   generations, and repeat.

The retained population has at most `system.retained_size` distinct candidates.
Older minima remain eligible, so an unsuccessful cooling cycle does not discard
the best structures. This initial version uses supplied builders for generation
0 only; later generations restart from selected minima.

## Cu8 example

```{literalinclude} ../../../examples/global_optimisation/explorations/simulated_annealing/cu8.yaml
:language: yaml
```

Run from the repository root:

```shell
gdp -d run-sa explore examples/global_optimisation/explorations/simulated_annealing/cu8.yaml
```

The example minimises two initial structures and runs three generations of four
annealing trajectories. It retains at most two distinct minima after each cycle.

## Configuration and parallel execution

Use `method: global_optimisation` and `strategy.method: simulated_annealing`.
The `system` configuration is shared with GA and BH.

- `strategy.annealing.start_temperature`: starting temperature in kelvin.
- `strategy.annealing.final_temperature`: final temperature in kelvin, no higher
  than the starting temperature. Both temperatures must be finite and positive.
- `strategy.annealing.schedule`: `linear` (the default and currently supported schedule).
- `strategy.convergence.generation`: final generation number; 0 performs only
  initial minimisation, and the default 1 performs one cooling generation.
- `strategy.minimisation`: executor parameters for endpoint minimisation,
  including positive `stop.steps` and `stop.fmax`.
- `strategy.objective`: shared energy or formation-energy ranking.
- `strategy.use_archive`: archive completed minimisation calculations.

The top-level runtime supplies ASE NVT dynamics with a Berendsen or Langevin
thermostat, the potential, and the scheduler. Its step count applies to the whole
cooling trajectory in each generation. Minimisation reuses the same
potential, scheduler, and dispatch settings, replacing only executor parameters
and method. This version requires an unmodified potential.

Use the runtime scheduler's batch and concurrency settings to execute independent
candidates in parallel. All candidates finish cooling and endpoint minimisation
before the next generation begins. With a direct scheduler's default settings, the batch may
execute serially. The exploration scheduler independently controls where the
population loop runs.

Every cycle starts with fresh velocities and persisted independent random seeds.
The runtime's velocity seed is overridden for those starts. Each generation
repeats the same start-to-final ramp; temperatures do not continue cooling across
generations. The thermostat target varies linearly with MD step, while the actual
kinetic temperature fluctuates and responds to the thermostat. The strategy
overrides the runtime temperature target; do not also specify a runtime end temperature.
For the example, 100 steps at 1 fs give a 100 fs quench from 1200 K to 300 K per
generation. Increase the step count for slower cooling.

## Results and restart

- `candidates.db`: all evaluated minima, including duplicate and unretained
  candidates; records include generation, parent IDs, and replica index.
- `population.json`: generation plans, parent selection, retained IDs,
  configuration fingerprint, and random streams.
- `population.xyz`: retained minima when the requested search completes.
- `gen0/minimisation/`: initial evaluations.
- `genN/cooling/slices/s000/`: the MD cooling trajectory batch.
- `genN/minimisation/`: final endpoint minimisations.

Each generation also snapshots its inputs. Restart with the same configuration
and directory: selected parents and random seeds are reused, queued calculations
are polled, and already recorded endpoints are not duplicated.
Changes to the configuration require a fresh directory.
`get_workers()` exposes cooling and minimisation workers for downstream analysis.

The earlier `method: simulated_annealing` recipe remains available as a
low-level cooling runner, accepting either `annealing` or the explicit
`temperatures` list of constant-temperature holds. Use the population configuration above for structure
search. These outputs are not compatible with the earlier experimental BH-based
SA configuration.
