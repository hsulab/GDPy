(scheduler-transport)=

# machine resources

gdpx separates the scheduler that starts work from the transport used to reach
the execution host. Worker batching and metadata behavior belong to the
runtime's separate `dispatch` section.

## Choose a scheduler

| Scheduler | Use it for |
| --- | --- |
| {doc}`direct` | Synchronous execution on the local machine or over SSH. |
| {doc}`nohup` | Detached local jobs for testing queue lifecycles without a queue manager. |
| {doc}`slurm` | Slurm allocations, CPU/GPU resources, and concurrent job steps. |
| {doc}`pbs` | PBS queue submission and resource directives. |
| {doc}`lsf` | LSF queue submission and resource directives. |

If the entire `scheduler` section is omitted, gdpx uses the `direct` scheduler
with the local transport. Third-party plugins may provide other schedulers.

```{toctree}
:hidden:
:maxdepth: 1

direct.md
nohup.md
slurm.md
pbs.md
lsf.md
```

## Scheduler and transport

The scheduler provider selects how work starts. Its optional nested transport
selects where the scheduler command runs.

| Execution path | Scheduler provider | Transport provider |
| --- | --- | --- |
| Run directly on this machine | `direct` | `local` |
| Run directly over SSH | `direct` | `ssh` |
| Submit locally to a queue | `slurm`, `pbs`, or `lsf` | `local` |
| Start detached local jobs | `nohup` | `local` |
| Submit to a remote queue | `slurm`, `pbs`, or `lsf` | `ssh` |

If `transport` is omitted, it defaults to local execution. An explicit local
transport looks like:

```yaml
scheduler:
  provider: slurm
  parameters: {}
  transport:
    provider: local
    parameters: {}
```

For remote execution, install SSH support with `pip install gdpx[remote]` and
configure the same scheduler with an SSH transport:

```yaml
scheduler:
  provider: slurm
  parameters: {}
  transport:
    provider: ssh
    parameters:
      hostname: cluster.example
      remote_wdir: /scratch/user/gdpx
```

gdpx stages the working tree into `remote_wdir/<job-name>`, runs submission and
status commands remotely, and synchronizes completed outputs before checking
convergence. gdpx must be installed and available on the remote `PATH`.

SSH uses Paramiko's username, key-file, and agent discovery. The existing
`<HOSTNAME>_PASSWORD` environment-variable convention may be used for password
authentication. `remote_wdir` must be an absolute POSIX path. OpenSSH aliases,
jump hosts, and extra connection fields are not interpreted by this transport.
Paths outside the staged tree are not transferred or rewritten, so models,
datasets, and executables must already be accessible on the remote host.

Direct SSH jobs remain attached to the SSH command and do not survive a lost
connection. Queue jobs are detached by the remote scheduler after submission.

## Batches and application launchers

`dispatch.batch_size` controls how many structures belong to one scheduler job.
It is not a CPU or GPU count. Calculations in a batch run sequentially unless a
supported scheduler configures `concurrent_tasks`.

For external MPI applications, `machine_prefix` wraps the application command.
For example, use `machine_prefix: srun --exact -n 4 -c 1` with a bare
`command: vasp_std`; do not wrap `gdp` itself with a multi-rank launcher.
Resource syntax and concurrency behavior are documented on each scheduler page.

## Packed training

For training committees, scheduler `parameters.concurrent_tasks` greater than
one packs that many independent models into each job. Set the allocation's
task, CPU, GPU, and memory totals in the scheduler parameters. Its
`machine_prefix` launches each single-model CLI (for Slurm, use a single-task
`srun --exclusive --exact` with the per-model resources). Each model has its
own work directory and log; the job fails if any child fails, and convergence
requires all models in the group. Packing currently supports local scheduler
transports. Existing submitted metadata must not be regrouped while jobs are
active. Set the train step's `auto_submit: false` to prepare without submitting.

Training and drive workers share a compact layout: `_meta/inputs.json` stores
the frozen job inputs, `_meta/scheduler.json` tracks submissions and completion,
and `_meta/jobscripts/run-<uuid>.script` contains each generated job script.
Model configurations, logs, checkpoints, and exports stay in `m0`, `m1`, etc.;
the shared training dataset stays in `shared_dataset`. Resubmission reconstructs
missing scripts and model configurations from the saved inputs, preserving model
seeds. Single-model and packed training use the same lifecycle. Old training
directories with `_<scheduler>_jobs.json` require a new working directory.

Submitting training does not require a persistent workflow controller. A later
workflow invocation can inspect completed jobs and continue with validation.

Before loading a dataset, the common training entry point checks each model's
training convergence. Completed models skip training and only ensure their
export exists; unfinished models use their trainer's normal restart behavior.
This applies to direct, single-model, and packed jobs. Packed resubmissions retain
the configured resource request even when some models are already complete.

## Queue lifecycle

Prepare, review, submit, inspect, and collect a queued calculation using one
output directory:

```shell
gdp -d queued-results -r runtime.yaml compute prepare structures.xyz
gdp -d queued-results compute submit
gdp -d queued-results compute status
gdp -d queued-results compute collect
```

`prepare` writes reviewable job scripts without submitting them. Failed or
unconverged jobs require inspection before an explicit resubmission such as
`gdp -d queued-results compute resubmit --batch 0`.

Use `gdp queue` to list this user's active GDPy simulations across working
directories on the current host. `--all-users` includes all visible users and
`--json` emits a machine-readable array. See {doc}`nohup` for process-discovery
behavior and detached-job logs.
