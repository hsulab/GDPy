# Basin hopping

`method: basin_hopping` runs the population-based search formerly named
`concurrent_hopping`. It selects starting structures for hopping chains,
evaluates candidates using execution workers, and ranks them using the search
objective. It inherits directly from `BaseExpedition`, independently of MC.

```{toctree}
:maxdepth: 1

bh/examples/cluster
bh/examples/cuox_thanos
```

See the shared {ref}`global-optimisation-population` reference for population
configuration. The recipe uses these settings:

- `population`: `retained_size`, named `builders`, `initial` allocations,
  `generation.total_size`, comparator, and extinction settings.
- `operators`: weighted move configurations shared with MC.
- `num_mcmoves`: number of proposals per candidate chain.
- `convergence.generation`: final generation number; defaults to `1`.
- `objective`: energy or formation-energy ranking, with chemical potentials
  for the latter.

By default, generation 0 generates and minimizes the initial structures, then
selects chain starts once for generation 1. Each chain advances through its own
accept/reject decisions for `num_mcmoves` proposals. Omit `convergence` to use
this default. Set `convergence.generation: 0` for initialization only; larger
values add population reselection between search generations.

This self-contained Cu₈ example generates four random structures using
`random_structure_improved` and launches two chains of ten proposals. Its
calculation runtime minimizes initial structures and every valid trial batch.

```{literalinclude} ../../../examples/global_optimisation/cu8_bh_emt.yaml
:language: yaml
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
state; it does not control whether the result is stored. When additional generations are requested, each selects from all eligible stored minima using the shared population comparator
and ranking. Repeated evaluations remain separate history records, while the
retained population removes similar structures. No extra candidate or relaxation
is created for a final chain endpoint.

Each trial record includes its chain, round, acceptance decision, originating
accepted candidate (`data.parents`), and starting parent. Invalid proposals have
no evaluated result and create no candidate record. Results excluded by extinction
rules remain stored but are ineligible for selection.

## Extinction and replacement chains

Configure extinction rules with `recipe.population.thanos`. Every evaluated
trial receives an extinction flag after its MC acceptance decision, and remains
in the database regardless of either result.

- A rejected trial leaves the chain unchanged, even if that trial is extinct.
- An accepted, non-extinct trial becomes the next chain state.
- An accepted, extinct trial terminates the current segment. It is stored with
  `accepted: true`, `extinct: 1`, and `data.outcome: extinct`, but is never used
  for another proposal.

After all results in the round are stored, BH refreshes the retained population
from eligible database minima, including discoveries from that round. It selects
replacement starts with the existing fitness weights, with replacement. Earlier
valid states remain eligible, even if a later trial terminated their segment.

Replacement starts reuse their stored energies and begin on the next round.
The terminating trial consumes a move: replacement never resets `num_mcmoves`.
No replacement is selected after the final round. If the replacement pool is
empty, the search terminates as extinct. Invalid proposals do not trigger this
mechanism, and the Cu₈ demo does not enable extinction rules.

Chain IDs identify persistent execution slots. `data.segment` increments on
replacement, and `data.start_parent` identifies the new segment's source.
Exported trajectories include replacement structures marked with `event: restart`,
`segment`, and `source_confid`. Rejected and extinct trial geometries remain
available in the database and worker outputs.

## Execution and checkpoints

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
