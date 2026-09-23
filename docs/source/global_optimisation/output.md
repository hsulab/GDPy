(exploration-output-layout)=

# exploration output layout

For a single exploration, `gdp -d work explore ...` writes algorithm outputs,
checkpoints, calculation directories, and `gdp.out` directly under `work/`.
There is no enclosing `expedition-0` directory.

When a recipe creates multiple explorations, each has its own `expo.<index>`
directory. Indices start at zero. The number of decimal digits in the exploration
count is rounded up to the next even width:

| Explorations | Output directories |
| ---: | --- |
| 2 | `expo.00`, `expo.01` |
| 10 | `expo.00` through `expo.09` |
| 100 | `expo.0000` through `expo.0099` |
| 10,000 | `expo.000000` through `expo.009999` |

The exploration worker keeps its bookkeeping in one shared `work/_meta/`:

- `_scheduler.json`: scheduler provider, saved exploration count, relative output
  directories, and submission/completion records in one file for every provider.
  Its `scheduler` table records `provider` (for example, `direct` or `slurm`),
  its `layout` table records directories, and TinyDB’s `_default` table holds jobs.
- `exp-<uuid>.json`: generated exploration inputs, including the calculation runtime.
- `run.script-<uuid>`: scripts that launch each exploration in its output directory.

Runs using separate `_meta/layout.json` (or `exploration.json`) and
`_<scheduler>_jobs.json` files, or the interim `_<scheduler>.json` files, remain
resumable. The worker validates and merges them into `_scheduler.json`,
preserving job identifiers and status, then removes
the old files. Read-only layout inspection does not migrate files.

Explorations do not need separate metadata directories for the exploration
worker. Calculation workers inside them retain their own caches and job records.
Default scheduler artifacts are written beside the submission scripts; explicit
scheduler output settings remain in effect. The main `gdp.out` stays in the
working directory, and separately launched explorations log in their output
directories.

Resume with the same working directory, scheduler provider, and exploration count. Changing the count
requires a fresh directory. Existing layouts with root-level job records or
`expedition-*` directories are detected and rejected; they are not automatically
moved or resumed by the new layout.

## Broadcast compositions in one allocation

A top-level `broadcast` mapping expands one recipe into independent searches:

```yaml
method: global_optimisation
broadcast:
  population.builders.random.composition:
    - {Cu: 6, Ni: 2}
    - {Cu: 4, Ni: 4}
strategy:
  method: basin_hopping
  # Full builder, population, operators, and convergence settings.
```

Run the complete supplied Cu–Ni example from the repository root:

```shell
OMP_NUM_THREADS=1 gdp -d run-cu-ni-compositions \
  --runtime examples/global_optimisation/runtimes/emt.yaml \
  explore examples/global_optimisation/explorations/basin_hopping/cu_ni_compositions.yaml
```

This runs two short EMT searches sequentially, using seed 7 for both. Cu₆Ni₂
writes to `expo.00` and Cu₄Ni₄ to `expo.01`. Submit
`sbatch examples/global_optimisation/two_explorations.slurm` to run both within
one allocation; adapt environment/account/partition settings to your cluster.

Broadcast keys are dotted paths into top-level search settings, including numeric
list indices such as `strategy.operators.0.temperature`. Each value is a nonempty list of alternatives;
an alternative replaces the entire target value, including dictionaries or lists.
Parents must exist, but final mapping keys may introduce optional settings.
Invalid paths, empty alternatives, and overlapping paths are rejected.

Multiple fields generate a Cartesian product in field/value order, with the
rightmost field varying fastest. Broadcast whole composition dictionaries to
keep element counts paired; broadcasting Cu and Ni separately generates all
combinations. Values in `broadcast` override the base recipe. Ordinary recipe
lists are unchanged. GA's existing chemical-potential expansion is applied
within each explicit broadcast combination.

Generated exploration inputs contain only resolved recipes. Rerun unchanged
settings to resume; use a fresh directory when changing sweep values or order.
Direct scheduling executes the searches sequentially. Selecting a queue
scheduler retains its usual separate submission per exploration.
