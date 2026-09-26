(direct)=

# direct

The `direct` scheduler runs a calculation synchronously without a queue manager.
It is the default when a runtime omits the `scheduler` section.

## Local execution

An explicit local configuration is:

```yaml
scheduler:
  provider: direct
  parameters: {}
  transport:
    provider: local
    parameters: {}
```

Activate the required environment before starting gdpx:

```shell
conda activate gdpx
OMP_NUM_THREADS=4 gdp -d local-min \
  -r examples/compute/tasks/relaxation.yaml \
  compute examples/compute/tasks/dimers.xyz
```

Generate the demo structures first as described in {doc}`tasks/index`.
`OMP_NUM_THREADS` limits threads only for libraries that honor it; it does not
make independent structures run concurrently.

For a GPU model, select a GPU-capable potential and calculator device:

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
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  gdp -d gpu-results -r gpu.yaml compute structures.xyz
```

Install the model dependencies and a compatible accelerator stack in that
environment. GPU visibility does not turn a CPU-only calculator into a GPU
calculator.

With the default `concurrent_tasks: 1`, direct execution calls the calculation
in the current process. Consequently, scheduler `environs` commands do not
modify that already-running process.

## Concurrent direct tasks

Set `concurrent_tasks` to run independent tasks through a generated Bash script:

```yaml
scheduler:
  provider: direct
  parameters:
    concurrent_tasks: 4
    environs: |
      export OMP_NUM_THREADS=2
```

Direct execution groups all input structures into one batch and runs a maximum
of four tasks at once in bounded waves. In this mode, `environs` is evaluated
by the generated script. The host must have enough resources for every
simultaneous application, and shared-workdir dispatch is not supported.

For external applications, `machine_prefix` may contain a local launcher such
as `mpirun`. Python-native GPU calculators are not assigned distinct devices by
this option; use `concurrent_tasks: 1` unless device isolation is handled
outside gdpx.

## Direct execution over SSH

Combine the direct scheduler with the SSH transport:

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

The SSH command remains open until the direct calculation finishes. See
{ref}`scheduler-transport` for staging, authentication, and path behavior.
