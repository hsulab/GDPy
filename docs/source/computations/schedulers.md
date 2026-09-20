(scheduler-transport)=

# Schedulers and transports

GDPy separates **how** work is dispatched from **where** the dispatch command
runs. The scheduler provider is `direct`, `slurm`, `lsf`, `pbs`, or a
third-party queue scheduler. Its nested transport is `local` or `ssh`.

| Execution path | Scheduler provider | Transport provider |
| --- | --- | --- |
| Run directly on this machine | `direct` | `local` |
| Run directly over SSH | `direct` | `ssh` |
| Submit to a queue on this machine | `slurm`, `lsf`, or `pbs` | `local` |
| Submit to a queue over SSH | `slurm`, `lsf`, or `pbs` | `ssh` |

If the entire `scheduler` section is omitted, GDPy uses direct execution with
the local transport. If only `transport` is omitted, the transport is local.

## Direct execution on this machine

```yaml
scheduler:
  provider: direct
  parameters: {}
  transport:
    provider: local
    parameters: {}
```

GDPy calls the calculation in the current process and waits for it to finish.
Activate the required environment before starting GDPy; `environs` contains
shell commands for generated scripts and does not modify an in-process callback.

## Direct execution over SSH

```yaml
scheduler:
  provider: direct
  parameters:
    environs: |
      conda activate gdpx
  transport:
    provider: ssh
    parameters:
      hostname: cluster.example
      remote_wdir: /scratch/user/gdpx
```

Install SSH support with `pip install gdpx[remote]`. GDPy stages the working
tree, runs the generated script over SSH, and keeps the SSH command open until
the calculation finishes. A later status or collection operation synchronizes
the results back to the initiating machine. GDPy must be installed and
available on the remote command's `PATH`.

SSH uses Paramiko's normal local username, key-file, and agent discovery. The
existing `<HOSTNAME>_PASSWORD` environment-variable convention may be used for
password authentication. `remote_wdir` must be an absolute POSIX path.
`hostname` is passed directly to Paramiko; OpenSSH aliases, jump hosts, and
additional connection fields are not interpreted by this transport.

Staging copies the working tree into `remote_wdir/<job-name>`. Paths in model,
dataset, or executable settings outside that tree are not automatically
rewritten or transferred; arrange for those resources to be accessible on the
remote host. Direct jobs remain attached to the SSH command, so this mode does
not provide a detached job that survives loss of the connection.

## Queue submission on this machine

```yaml
scheduler:
  provider: slurm
  parameters:
    partition: compute
    ntasks: 1
    time: "1:00:00"
  transport:
    provider: local
    parameters: {}
```

The queue command, such as `sbatch`, runs on the current machine.

## Queue submission over SSH

```yaml
scheduler:
  provider: slurm
  parameters:
    partition: compute
    ntasks: 1
    time: "1:00:00"
  transport:
    provider: ssh
    parameters:
      hostname: cluster.example
      remote_wdir: /scratch/user/gdpx
```

GDPy stages the working tree and runs submission and status commands on the
remote host. Completed files are synchronized before convergence is checked.

`options.batch_size` controls how many structures are assigned to a queued
task. Direct execution uses one synchronous batch regardless of transport.
