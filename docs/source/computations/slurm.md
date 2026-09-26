(slurm)=

# slurm

The `slurm` scheduler writes `#SBATCH` directives, submits with `sbatch`, and
checks active jobs with `squeue`. Parameter keys are rendered directly as Slurm
options, so use the resource names supported by the target cluster.

## CPU allocation

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
dispatch:
  batch_size: 10
```

This requests one process with four CPU threads. `batch_size: 10` assigns up to
ten structures to the allocation; it does not request ten CPUs. Replace queue,
account, environment, and resource values with site-specific settings.

For an external MPI application, set `machine_prefix` to the Slurm step command
and leave the potential command unwrapped:

```yaml
scheduler:
  provider: slurm
  parameters:
    ntasks: 64
    machine_prefix: srun --exact -n 64 -c 1 --cpu-bind=cores
```

## GPU allocation

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
dispatch:
  batch_size: 10
```

The queue normally sets device visibility. Do not replace that binding with a
fixed workstation value such as `CUDA_VISIBLE_DEVICES=0`. Requesting a GPU does
not make a CPU-only potential GPU-capable.

## Concurrent calculations in one allocation

`concurrent_tasks` is the maximum number of independent calculations running at
once inside one batch. A batch of 100 structures with `concurrent_tasks: 4`
runs in 25 bounded waves. The generated script uses loops, so its size does not
grow with the batch.

For 256 allocated CPUs and four concurrent pure-MPI VASP calculations using 64
ranks each:

```yaml
scheduler:
  provider: slurm
  parameters:
    nodes: 2                  # Adjust for the site's cores per node.
    ntasks: 256
    ntasks-per-node: 128
    cpus-per-task: 1
    time: "03:00:00"
    concurrent_tasks: 4
    machine_prefix: >-
      srun --exact -N 1 -n 64 -c 1 --cpu-bind=cores
    environs: |
      export OMP_NUM_THREADS=1
dispatch:
  batch_size: 100
```

The step size is `ntasks * cpus-per-task`: `srun -n 64 -c 1` consumes 64 CPUs,
while `srun -n 64 -c 4` consumes all 256 CPUs and permits only one such step in
this allocation. gdpx does not derive or validate this arithmetic.

For four external GPU calculations in one allocation:

```yaml
scheduler:
  provider: slurm
  parameters:
    nodes: 1
    ntasks-per-node: 4
    cpus-per-task: 32
    gpus-per-task: 1
    concurrent_tasks: 4
    machine_prefix: >-
      srun --exact -N 1 -n 1 -c 32 --gpus-per-task=1
dispatch:
  batch_size: 8
```

`machine_prefix` wraps VASP, LAMMPS, or another external application. Wrapping
`gdp` with `srun -n 64` would start 64 duplicate controllers. Concurrent shared
working directories are rejected. Python-native GPU calculators do not use the
external command prefix, so this option does not isolate their devices.

## Local and remote submission

With no transport, `sbatch` runs on the current host. Add an SSH transport to
stage the working tree and submit on another host:

```yaml
scheduler:
  provider: slurm
  parameters:
    partition: compute
    ntasks: 1
    time: "01:00:00"
  transport:
    provider: ssh
    parameters:
      hostname: cluster.example
      remote_wdir: /scratch/user/gdpx
```

See {ref}`scheduler-transport` for transport behavior and the common queue
lifecycle.
