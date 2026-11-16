(bh-composition-broadcast-example)=

# copper–nickel cluster: composition sweep

## system

This example runs independent Cu₆Ni₂ and Cu₄Ni₄ basin-hopping searches with
EMT. Both use seed 7, two initial candidates, and two hopping rounds. The move
operator acts on both species, preserving each search's composition.

## configuration

The `broadcast` field replaces the entire composition mapping for each search.
Keeping the element counts together produces exactly two compositions.

```{literalinclude} ../../../../../examples/global_optimisation/explorations/basin_hopping/cu_ni_compositions.yaml
:language: yaml
```

Pair this exploration with the EMT runtime:

```{literalinclude} ../../../../../examples/global_optimisation/runtimes/emt.yaml
:language: yaml
```

## run

Follow {ref}`running the examples <go-running-examples>` with these settings:

| Method folder | Exploration input | Runtime | Output directory |
| --- | --- | --- | --- |
| `basin_hopping` | `cu_ni_compositions.yaml` | `../../runtimes/emt.yaml` | `run-cu-ni-compositions` |

The searches execute sequentially in one process with independent calculation
workers. To run both within one Slurm allocation, activate your gdpx environment,
adapt account/partition settings in the supplied script, and submit from the
repository root:

```shell
sbatch examples/global_optimisation/two_explorations.slurm
```

## results

- `run-cu-ni-compositions/expo.00/`: Cu₆Ni₂ candidates, results, and checkpoints.
- `run-cu-ni-compositions/expo.01/`: Cu₄Ni₄ candidates, results, and checkpoints.
- `run-cu-ni-compositions/_meta/_scheduler.json`: shared provider, layout, and job records.
- `run-cu-ni-compositions/gdp.out`: progress for both searches.

Rerun the same command to resume. Use a fresh directory when changing broadcast
values, their order, or calculation settings. See {ref}`exploration-output-layout`
for broadcast ordering and metadata details.
