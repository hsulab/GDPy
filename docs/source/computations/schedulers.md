(scheduler-transport)=

# machine resources

gdpx separates the scheduler that starts work from the transport used to reach
the execution host. Worker batching and metadata behavior belong to the
runtime's separate `dispatch` section.

## Choose a scheduler

| Scheduler | Use it for |
| --- | --- |
| {doc}`direct` | Synchronous execution on the local machine or over SSH. |
| {doc}`slurm` | Slurm allocations, CPU/GPU resources, and concurrent job steps. |
| {doc}`pbs` | PBS queue submission and resource directives. |
| {doc}`lsf` | LSF queue submission and resource directives. |

If the entire `scheduler` section is omitted, gdpx uses the `direct` scheduler
with the local transport. Third-party plugins may provide other schedulers.

```{toctree}
:hidden:
:maxdepth: 1

direct.md
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
