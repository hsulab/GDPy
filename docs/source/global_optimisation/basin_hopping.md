# basin hopping

Examples keep search recipes in
`examples/global_optimisation/explorations/basin_hopping/` and reusable
calculation settings in `examples/global_optimisation/runtimes/`. Combine them
with `gdp --runtime <runtime.yaml> explore <exploration.yaml>`, placing
`--runtime` before `explore`. To compare potentials, keep the exploration file
and select another suitable runtime in a new output directory.

`method: global_optimisation` with `strategy.method: basin_hopping` runs the population-based search formerly named
`concurrent_hopping`. It selects starting structures for hopping chains,
evaluates candidates using execution workers, and ranks them using the search
objective. It inherits `PopulationBasedExploration`, independently of MC.

Basin hopping reuses the {ref}`sampling-operators` documented under Boltzmann Sampling. Configure those same moves in `strategy.operators`. With a minimization
runtime, BH relaxes valid trials before the chain's acceptance decision.

```{toctree}
:maxdepth: 2

bh/examples/index
```

See the shared {ref}`global-optimisation-population` reference for population
configuration. The recipe uses these settings:

- `system`: `retained_size`, named `builders`, `initial` allocations,
  `generation.total_size`, comparator, and extinction settings.
- `strategy.operators`: weighted moves; see {ref}`sampling-operators` for configuration and {ref}`bh-operator-logs` for move logs.
- `strategy.steps_per_chain`: attempted proposals per chain per generation; a non-negative integer.
  Rejected and invalid proposals count; the starting structure does not. The
  progress header labels this value `steps/chain`.
- `strategy.selection.replace`: sample chain starts with replacement; defaults to `false`.
- `strategy.convergence.generation`: final generation number; defaults to `1`.
- `strategy.objective`: energy or formation-energy ranking, with chemical potentials
  for the latter.

By default, generation 0 generates and minimizes the initial structures, then
selects chain starts once for generation 1. Each chain advances through its own
accept/reject decisions for `steps_per_chain` proposals. Omit
`strategy.convergence` to use this default. Set
`strategy.convergence.generation: 0` for initialization only; larger
values add population reselection between search generations.

This self-contained Cu₈ example generates four random structures using
`random_structure_improved` and launches two chains of ten proposals. Its
calculation runtime minimizes initial structures and every valid trial batch.
It uses only {doc}`../explorations/operators/move` for a simple fixed-composition search.
The {ref}`sampling-operators` reference links to a detailed page for each operator.

```{literalinclude} ../../../examples/global_optimisation/explorations/basin_hopping/cu8.yaml
:language: yaml
```

Pair this exploration with the following runtime (passed with `--runtime`):

```{literalinclude} ../../../examples/global_optimisation/runtimes/emt.yaml
:language: yaml
```

See {ref}`exploration-output-layout` for output directories and restart metadata.

## Chain-start selection

```yaml
strategy:
  method: basin_hopping
  selection:
    replace: false
```

Chain starts use fitness-weighted sampling. By default, selection is without
replacement when the eligible retained population has at least as many candidates
as the requested starts. If fewer candidates are available, that selection call
automatically samples with replacement. Set `replace: true` to always allow
replacement, retaining the previous selection policy.

The same setting applies to generation starts and simultaneous extinction
restarts. Uniqueness applies within each selection call: candidates used by
other running chains remain eligible. An empty population still terminates the
search as extinct. Selection borrows candidate references without copying atoms.

Already checkpointed starts are reused on resume. The new default can change
future selections for the same seed; use `replace: true` to retain the previous
sampling policy, and resume with the same configuration for reproducibility.

## Acceptance and population ranking

Operators use the shared {ref}`sampling-operators` reference. The population's
ranking objective is separate from the chain acceptance rule. Keep exchange
`chempots` and objective `chemical_potentials` consistent when using
formation-energy ranking. BH owns population selection and search objectives;
its population is not an equilibrium sample.

## Batched rounds and execution

BH advances chains in rounds: one proposal per chain, one batch of valid trials,
then acceptance after every trial result is available. Invalid proposals consume
a hop without evaluation. Rejected trials retain the previous accepted minimum.
Every evaluated minimum enters the candidate database, including rejected
trials and accepted intermediate minima. Acceptance controls the chain's next
state; it does not control whether the result is stored. When additional generations are requested, each selects from all eligible stored minima using the shared population comparator
and ranking. Repeated evaluations remain separate history records, while the
retained population removes similar structures. No extra candidate or relaxation
is created for a final chain endpoint.

Each trial record includes its chain, round, acceptance decision, originating
accepted candidate (`data.parents`), and starting parent. Invalid proposals have
no evaluated result and create no candidate record. Results excluded by extinction
rules remain stored but are ineligible for selection.

## Extinction and replacement chains

Configure extinction rules with `system.thanos`. Every evaluated
trial receives an extinction flag after its MC acceptance decision, and remains
in the database regardless of either result.

- A rejected trial leaves the chain unchanged, even if that trial is extinct.
- An accepted, non-extinct trial becomes the next chain state.
- An accepted, extinct trial terminates the current segment. It is stored with
  `accepted: true`, `extinct: 1`, and `data.outcome: extinct`, but is never used
  for another proposal.

After all results in the round are stored, BH refreshes the retained population
from eligible database minima, including discoveries from that round. It selects
replacement starts with the existing fitness weights and `strategy.selection.replace`
policy described above. Earlier
valid states remain eligible, even if a later trial terminated their segment.

Replacement starts reuse their stored energies and begin on the next round.
The terminating trial consumes a move: replacement never resets `steps_per_chain`.
No replacement is selected after the final round. If the replacement pool is
empty, the search terminates as extinct. Invalid proposals do not trigger this
mechanism, and the Cu₈ demo does not enable extinction rules.

Chain IDs identify persistent execution slots. `data.segment` increments on
replacement, and `data.start_parent` identifies the new segment's source.
Exported trajectories include replacement structures marked with `event: restart`,
`segment`, and `source_confid`. Rejected and extinct trial geometries remain
available in the database and worker outputs.

See {ref}`bh-operator-logs` for operator setup output and detailed proposal logs.

## Execution and checkpoints

### Generation output

BH prints a scrolling, bordered block for each generation in the terminal and
`gdp.out`. Every line, including borders, has the standard logging timestamp
and level prefix (omitted in the illustration below). Generation 0 reports initialization; hopping generations show one
row per committed round:

```text
round  eval  accept  reject  invalid  extinct  restart  best energy [eV]
6/8       2       2       0        0        1        1          -36.1077
```

The round column expands to fit the configured move count, including `1000/1000`.
`eval` counts minimized/evaluated trials. `accept` includes accepted extinct
trials; `reject` counts MC rejections. `invalid` counts proposals that were not
evaluated. `extinct` includes both accepted and rejected trials marked extinct,
whereas `restart` counts actual replacement starts. These columns overlap:
extinction and restart are not additional acceptance outcomes.

The best eligible value uses the configured objective, excludes extinct
candidates, and includes discoveries from initialization and earlier generations,
even if a trial was MC-rejected. Eligible candidate totals are database counts,
not the size of the deduplicated retained population. Missing values appear as
`—` (or `-` with ASCII output).

Blocks end with `complete`, `waiting`, `extinct`, or `failed`. Resumed blocks
identify the last committed round and reconstruct cumulative counts without
reprinting previous round rows. Timings cover the current invocation only.
Routine GDP worker and move messages appear with `gdp --debug`; warnings and
errors remain visible normally. Blocks use no cursor control or colour escapes,
so redirected and scheduler logs retain the same readable structure.

(bh-operator-logs)=

### Setup output and move logs

Each BH invocation prints a compact setup box with operator indices and names,
normalized selection probabilities, particles, temperatures, and move-specific
settings.

Detailed move diagnostics are saved automatically in one file per hopping
generation: `tmp_folder/gen1/mcmoves.log`, `tmp_folder/gen2/mcmoves.log`, and so
on, alongside each generation's `rounds/` and `evaluations/` folders. Every line has
a timestamp, level, generation, round, chain, segment, parent candidate, and
operator index/name. Invocation headers contain full resolved operator settings;
fields that do not apply to a header use `-`.

Routine operator messages are written at normal verbosity. Detailed DEBUG
messages are included only when gdpx's DEBUG logging is enabled. Move details
stay out of the normal console, while setup and progress boxes remain visible.

Logs distinguish uncommitted proposal diagnostics from committed outcomes.
Outcome lines include acceptance/rejection, energies, extinction, and restart
candidate IDs where applicable. Invalid proposals are logged without an
evaluation. Resume appends an invocation marker and identifies reused pending
proposals; it does not regenerate them for logging. An interruption can leave
uncommitted or replayed diagnostics, so `events.jsonl` and checkpoints remain the
authoritative scientific history. These logs survive checkpoint cleanup.

Logging uses existing results and scalar metadata: it does not copy atoms,
evaluate calculators, or consume random numbers.

### Runtime and restart files

Top-level `scheduler` places the exploration loop. Top-level `runtime` defines
its calculation worker; `runtime.scheduler` places the expensive calculations.
Both use the existing direct, queue, and transport configuration. No driver is
accessed by BH. With a minimisation runtime, every valid trial is minimized;
a single-point runtime omits this step and is not conventional basin hopping.

Round checkpoints under `tmp_folder/gen*/rounds` persist pending trial inputs,
acceptance context, accepted structures, segment state, replacement IDs, and all
named RNG streams, including population selection. Restart with the same
recipe and directory to resume queued work. Results are matched by trial ID,
and acceptance uses chain order independently of result arrival order.

Checkpoints use versioned JSON metadata and uncompressed NumPy `.npz` arrays,
loaded without pickle. Only the latest and previous committed rounds and the
current pending batch are retained. A damaged latest snapshot falls back to the
previous one. `events.jsonl` records candidate IDs, acceptance decisions, and
segment changes; structures remain in `candidates.db`, including rejected minima.
BH does not write chain XYZ trajectories automatically. Use
`gdpx.exploration.basin_hopping.export_trajectories(rounds_directory, output_directory)`
to export them on demand from committed events and database structures; see the
{doc}`bh/examples/cluster` example. An optional `database` argument specifies a
relocated candidate database. After generation finalization, full round snapshots are removed; the
event journal and `final.json` completion metadata remain.

Remove the former `recipe.mcworker` and move its calculation settings into
`runtime`. Older pickle, serial-chain, and pre-replacement round checkpoints cannot resume an
in-progress generation with this implementation. Completed history remains
readable; minima from older worker outputs are not backfilled automatically. Round ordering also changes trajectories for old
random seeds; new runs are reproducible across restart boundaries.

## Lineage figures

Completed runs write compact 1200 × 600 PNGs to `results/lineage/`.
`gen0001.png` combines all initial (generation 0) candidates with generation 1.
`gen0002.png` combines the retained population at the start of generation 2 with
its trials, and later generations follow the same pattern. Earlier candidates
referenced by parentage or restarts are also included in the left column.
An initialization-only run writes `gen0000.png`.

Retained population IDs are saved in the generation plan without copying atoms.
Older runs without this snapshot show recorded chain starts and referenced
candidates instead of reconstructing an uncertain historical population.

Plots retain a frame, a round x-axis, and an energy colorbar, but omit titles,
legends, and y-axis text. Markers and candidate IDs scale to the available pixel spacing, up to
24-point markers and 18-point IDs. Labels are omitted when they cannot fit
without crowding or crossing the frame. Dense production runs
(for example, 20 chains × 50 rounds) omit IDs and use smaller nodes.
Rounds run left to right, with chains in separate lanes and rejected trials
slightly offset; the left column's vertical positions do not indicate chain
assignment.

The `coolwarm` colormap encodes energy within each figure (blue is lower, red higher);
orange indicates unavailable energy. Filled nodes are initial or accepted
candidates, hollow nodes are rejected trials, and red crosses mark extinct
candidates. Dark-blue arrows show parentage. Purple dashed arrows point from a
terminated trial to the selected restart candidate; these are selection links,
not parentage. Only committed hopping rounds are included, using metadata
without loading atomic structures.
