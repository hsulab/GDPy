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

## Examples

The central {doc}`examples catalogue <examples/index>` groups recipes by physical
system. See {ref}`Cu₈ annealing <sa-cu8-example>` for an EMT demonstration and
{ref}`CO adsorption on Cu(111) <co-cu111-example>` for a DPA4C surface search.

## Configuration and parallel execution

Use `method: global_optimisation` and `strategy.method: simulated_annealing`.
The `system` configuration is shared with GA and BH. The strategy owns only
population concerns: `convergence.generation`, `objective`, and `use_archive`.
Generation 0 performs initial minimisation; each subsequent generation runs one
cooling/minimisation pass per selected candidate.

Supply a `native/simulated_annealing` compute runtime. Its `parameters.setup`
is an ordered list containing an MD executor and a minimisation executor.
Both inherit the outer potential. Each may specify an independent `scheduler`,
defaulting to direct execution. The outer runtime scheduler schedules candidate
annealing tasks; the exploration scheduler schedules the population loop.

The MD child defines the starting temperature through
`setup.regulator.targets.temperature` and the final temperature through
`setup.tend`. Temperatures must be finite and positive, with final no higher
than start. Its `stop.steps` counts integration steps for the full linear ramp.
The native SA executor currently validates ASE Berendsen and Langevin ramps;
unsupported providers/controllers fail during preparation. The minimisation
child requires positive `stop.steps` and `stop.fmax`. SA currently requires an
unmodified potential.

Population SA requires the parent `stop.steps` to equal one. A standalone
compute run defaults to one pass and can repeat passes by increasing this count.
See the {doc}`single-structure annealing task <../computations/tasks/simulated-annealing>`.

Population cycles start with fresh velocities and persisted independent random
seeds. The MD velocity seed is overridden for those starts. Each generation
repeats the same start-to-final ramp. All candidate passes finish before the
next generation begins; scheduler and dispatch settings determine parallelism.

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
- `genN/sim/`: the batch of composite annealing tasks, with per-candidate
  MD and minimisation children under `candN/calculations/`.
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
