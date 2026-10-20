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

The calculation runtime is supplied separately:

```{literalinclude} ../../../examples/global_optimisation/explorations/simulated_annealing/runtime.yaml
:language: yaml
```

Run from the repository root:

```shell
gdp -d run-sa \
  --runtime examples/global_optimisation/explorations/simulated_annealing/runtime.yaml \
  explore examples/global_optimisation/explorations/simulated_annealing/cu8.yaml
```

The example minimises two initial structures and runs three generations of four
annealing trajectories. It retains at most two distinct minima after each cycle.

## CO adsorption on Cu(111) with DPA4C

This example searches positions and orientations of one CO molecule on an
orthogonal Cu(111)-p(2×2) slab. The right-angled surface cell is rectangular,
5.11 × 4.43 Å, not exactly square. Three layers contain twelve Cu atoms.
The bottom four Cu atoms are fixed during both MD and minimisation; the upper
two layers and CO are mobile.

The dedicated substrate `cu111_p2x2_3layer_sa.xyz` preserves the geometry of
`cu111_p2x2_3layer.xyz`, with substrate tags set to zero and all three PBC flags
enabled to match the population interface. Its 24 Å cell height leaves vacuum
between periodic slab images. The original shared substrate is unchanged.

Four starting structures are built by inserting an intact CO molecule with
random orientation and centre of mass 1.5–3.5 Å above the top Cu layer.
The search retains up to four distinct minima and performs three generations
of four quenches. Each quench starts with fresh velocities and cools from 600
to 100 K over 1,000 steps at 0.5 fs (0.5 ps). If fewer than four parents survive
duplicate filtering, selection uses replacement.

```{literalinclude} ../../../examples/global_optimisation/explorations/simulated_annealing/co_cu111.yaml
:language: yaml
```

The separate runtime uses the DeepMD ASE backend and an ASE Langevin thermostat:

```{literalinclude} ../../../examples/global_optimisation/explorations/simulated_annealing/co_cu111_runtime.yaml
:language: yaml
```

Supply a DPA4C foundation model at `models/dpa4c-foundation.pt2`, or edit the
runtime model path. The model must cover Cu, C, and O and be exported for the
execution device. No weights are bundled or downloaded. See
{ref}`DPA4C setup <potential-deepmd>` for export and installation guidance.
Element coverage alone does not establish accuracy for CO adsorption.

Run from the repository root:

```shell
gdp -d run-co-cu111-sa \
  --runtime examples/global_optimisation/explorations/simulated_annealing/co_cu111_runtime.yaml \
  explore examples/global_optimisation/explorations/simulated_annealing/co_cu111.yaml
```

CO starts intact but is flexible during MD and minimisation. Fragment tags
identify the initial molecule; they do not constrain its bond or prevent
desorption or dissociation. Inspect the final structures to check their chemical
state. Candidates are ranked by total energy because their composition is fixed;
these energies are not adsorption energies relative to a clean slab and gas-phase
CO. This small, short search demonstrates the workflow, not convergence or model
accuracy. Results are written to `results/all_candidates.xyz` and `results/pop.png`.

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

The top-level runtime supplies MD dynamics (`executor.method: md`), the
potential, and the scheduler. SA does not restrict the executor provider or
thermostat; the executor validates its own settings. Use thermostatted NVT or
NPT dynamics and an executor that honours the start/end temperature targets
(`temp` and `tend`) for a linear ramp. Accepting an MD runtime does not guarantee
that its backend implements temperature ramps. The Cu8 example uses ASE.
The runtime step count applies to the whole
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
- `results/all_candidates.xyz`: all evaluated endpoint minima, including duplicates
  and candidates not retained.
- `results/pop.png`: candidate objective versus generation, with parent–child
  links, the best evaluated objective so far, and the final retained population
  highlighted. Generation 0 contains the minimised initial structures.
- `gen0/min/`: initial evaluations.
- `genN/sim/slices/s000/`: the MD cooling trajectory batch.
- `genN/min/`: final endpoint minimisations.
- `genN/inputs.xyz`: an extended XYZ snapshot of the generation's starting
  structures, not its MD trajectory. Generation 0 stores builder outputs; later
  generations store selected parent structures, with old momenta removed so that
  each annealing attempt starts with fresh velocities.

Each generation also snapshots its inputs. Restart with the same configuration
and directory: selected parents and random seeds are reused, queued calculations
are polled, and already recorded endpoints are not duplicated.
Keep `inputs.xyz` when restarting: it avoids rebuilding or reselecting the inputs.
The directory names `sim` and `min` replace the earlier `cooling` and
`minimisation` names; existing directories are not automatically renamed.
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
