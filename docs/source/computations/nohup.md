(nohup)=

# nohup

The `nohup` scheduler starts detached local jobs immediately, without a queue
manager. It follows the same asynchronous submit, query, and retrieve protocol
as Slurm, making it useful for testing queued workflows on a local POSIX host.

```yaml
scheduler:
  provider: nohup
  parameters:
    environs: |
      export OMP_NUM_THREADS=1
dispatch:
  batch_size: 1
```

Activate an environment containing `gdp` before submission. Jobs execute the
generated script with `bash -l`; scheduler `environs` commands can activate an
environment or set the application `PATH` inside that script.

```shell
gdp -d queued-results -r runtime.yaml compute prepare structures.xyz
gdp -d queued-results compute submit
gdp -d queued-results compute status
gdp -d queued-results compute collect
```

Submission returns after the detached supervisor starts. Jobs survive exit of
the submitting GDPy process, and subsequent invocations can query their state.
`dispatch.batch_size` controls structures per job. Separate submitted jobs can
run simultaneously; `concurrent_tasks` limits concurrent tasks within each job,
using the same bounded task loop as other supported schedulers. Its default is
one. There is no host-wide resource limit or waiting queue.

## Runnable EMT example

From the repository root, use `examples/compute/cu2_emt_nohup/runtime.yaml`
with the existing three Cu dimers. ASE's bundled EMT calculator needs no model
download or external simulator. Activate the environment containing gdpx, then
use a fresh output directory:

```shell
gdp -d cu2-nohup -r examples/compute/cu2_emt_nohup/runtime.yaml \
  compute prepare examples/compute/cu2_emt/structures.xyz
gdp -d cu2-nohup compute submit
gdp queue
gdp -d cu2-nohup compute status
```

The example starts three separate jobs, one per dimer. Its `sleep 5` environment
command gives time to observe active jobs; remove that delay for normal runs.
Repeat `compute status` until it reports `finished`, then collect:

```shell
gdp -d cu2-nohup compute collect
```

The three final structures are written to `cu2-nohup/results/end_frames.xyz`.
Submission, status queries, and collection can be run in separate terminal
sessions. Inspect per-attempt logs under `cu2-nohup/_meta/jobscripts/`.

## List simulations across submissions

```shell
gdp queue                 # this user's simulations on the current host
gdp queue --all-users     # simulations of all visible users
gdp queue --long          # full job IDs, directories, and commands
gdp queue --json          # JSON array for scripts
```

These commands wrap `ps` and do not require a runtime, plan, or calculation
directory. They list nohup jobs across submissions and working directories,
alongside identifiable GDPy compute, exploration, and workflow-run processes
started through other launch paths. Each nohup allocation has one row;
descendant GDPy tasks are suppressed. Other jobs use `pid:<PID>` identifiers.

The compact table shows abbreviated nohup IDs (the first eight UUID characters),
PID, user, OS process state, elapsed time, and the directory shortened to 32
characters (with `…` marking an omitted prefix). Use `--long` for full IDs,
directories, and commands. JSON always retains full values. Sleeping and stopped
simulations remain listed; exited and zombie
processes do not. A directory unavailable to the current user is shown as `—`.
JSON uses `null` for unavailable directories.

Discovery covers the queried host's visible processes. It cannot identify
arbitrary Python scripts using GDPy without a recognizable launcher, inspect
other hosts, or list allocations waiting in a cluster scheduler. Use that
scheduler's queue command for pending allocations.

## Logs, failures, and resubmission

Each submitted attempt writes stdout and stderr to `nohup-<UUID>.out` beside
its job script. Previous attempt logs remain available after resubmission.
The backend creates no status directories or startup/completion JSON files.
It queries live supervisors through `ps`; disappearance from the process table
means scheduler completion, just as disappearance from `squeue` does for the
Slurm backend. GDPy's shared `_meta/scheduler.json` retains submission IDs and
attempt counts for every scheduler.

Generated job commands disable the extra `gdp.out` file log. Child diagnostics
go to the attempt's `nohup-<UUID>.out`, so independent jobs do not append to the
submitting CLI's shared log.

A terminal scheduler process does not imply a converged calculation. Inspect
the attempt log and calculation outputs before explicitly resubmitting:

```shell
gdp -d queued-results compute resubmit --batch 0
```

Running jobs cannot be resubmitted. Existing `.script.nohup/` directories from
older versions are ignored and left intact so their logs remain available.
Dry runs start no processes. The scheduler supports local
transport only; it does not allocate resources, provide cancellation, or wrap
the SSH transport.
