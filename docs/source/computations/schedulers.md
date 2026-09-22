(scheduler-transport)=

# machine resources

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

## Local CPU and GPU runs

Run a task directly after activating its environment:

```shell
conda activate gdpx
OMP_NUM_THREADS=4 gdp -d local-min -r examples/compute/tasks/relaxation.yaml compute examples/compute/tasks/dimers.xyz
```

Generate the demo structures first as described in {doc}`tasks/index`.
`OMP_NUM_THREADS` limits threads only for libraries that honor it; it does not
make independent structures run in parallel. Direct execution uses one
synchronous batch.

For a GPU model, configure a GPU-capable potential. For example, save as
`gpu.yaml` and use a compatible local TACE checkpoint:

```yaml
potential:
  provider: tace
  parameters:
    model: ./model.pt
    device: cuda
executor:
  provider: ase
  method: spc
  parameters: {}
```

```shell
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 gdp -d gpu-results -r gpu.yaml compute structures.xyz
```

Install the model’s dependencies and a compatible accelerator stack in that
environment. `CUDA_VISIBLE_DEVICES` selects visible GPUs, while the potential’s
`device` setting selects the calculator device. Merely requesting a GPU does
not turn a CPU-only calculator such as EMT into a GPU calculator.

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

## Slurm CPU allocation

Append this scheduler to a task runtime when launching from a cluster login
node with local access to `sbatch`:

```yaml
scheduler:
  provider: slurm
  parameters:
    partition: compute
    nodes: 1
    ntasks: 1
    cpus-per-task: 4
    mem-per-cpu: 2G
    time: "01:00:00"
    environs: |
      source /path/to/miniconda3/etc/profile.d/conda.sh
      conda activate gdpx
      export OMP_NUM_THREADS=4
options:
  batch_size: 10
```

Replace the partition and environment paths with your site settings. Hyphenated
keys such as `cpus-per-task` are scheduler parameters, not executor settings.
This example requests one process with four CPU threads per queued job.
`batch_size: 10` groups up to ten structures into a job; it is not a CPU count.

For an MPI simulator, request the required `ntasks` and configure its launch
command under the potential, such as `command: srun vasp_std`. The scheduler
allocation alone does not add an MPI launcher to the calculator command.

Prepare, submit, inspect, and collect using the same output directory:

```shell
gdp -d queued-results -r queued.yaml compute prepare structures.xyz
gdp -d queued-results compute submit
gdp -d queued-results compute status
gdp -d queued-results compute collect
```

`prepare` writes inputs without submitting a job. Inspect the generated script
before submission. Collect after jobs finish; failed or unconverged jobs need
inspection before an explicit `compute resubmit --batch 0`.

## Slurm GPU allocation

For a GPU-capable potential, use a site-appropriate GPU partition and request
a GPU in the same runtime:

```yaml
scheduler:
  provider: slurm
  parameters:
    partition: gpu
    nodes: 1
    ntasks: 1
    cpus-per-task: 4
    gres: gpu:1
    time: "01:00:00"
    environs: |
      source /path/to/miniconda3/etc/profile.d/conda.sh
      conda activate gdpx-gpu
      export OMP_NUM_THREADS=4
options:
  batch_size: 10
```

The queue normally sets device visibility. Do not override its assigned GPU
IDs with the workstation example’s `CUDA_VISIBLE_DEVICES=0`. Select a suitable
calculator device under the potential parameters. This allocation supplies
one GPU per job; GDPy does not automatically split a model across multiple GPUs.

## PBS and LSF

The resource keys follow the selected scheduler’s own directives. For example:

```yaml
scheduler:
  provider: pbs
  parameters:
    q: workq
    l: "select=1:ncpus=4:mem=8gb,walltime=01:00:00"
```

```yaml
scheduler:
  provider: lsf
  parameters:
    q: normal
    n: 4
    W: "1:00"
    R: "span[hosts=1]"
```

These are scheduler fragments to combine with a potential and executor.
Queue names and resource syntax depend on the site. Add `environs` to activate
the required software and add `transport: {provider: ssh}` with the hostname
and remote working directory as shown above when submission must happen remotely.
