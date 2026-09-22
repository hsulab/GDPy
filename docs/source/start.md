# getting started

This section introduces the schema-v3 **potential**, **executor**, and
**scheduler** components used to calculate structures with gdpx.

The related commands are

```shell
# gdp -h for more info
$ gdp -h

# --- run simulations on local nodes or submitted to job queues
$ gdp -r ./runtime.yaml compute ./structures.xyz

# - if -d option is used, results would be written to the folder `./results`
$ gdp -d ./results -r ./runtime.yaml compute ./structures.xyz
```

An example input file (`runtime.yaml`) is organised as follows:

```yaml
potential:
    provider: deepmd
    parameters:
        model: ./graph.pb
executor:
    provider: ase
    method: md
    parameters:
        ensemble: nvt
        temp: 600
        timestep: 1.0
        steps: 100
```

No `scheduler` section is needed for direct execution on the current machine.
Omitting `schema_version` uses the current configuration schema. Explicit
unsupported versions are rejected; serialized runtimes and saved compute plans
still record their schema version.

## Units

We use the following units through all input files:

Time `fs`, Length `AA`, Energy `eV`, Force `eV/AA`.

## Potential

Potential providers describe a model independently of the software that will
execute it. Materialization connects the two at runtime.

The example below shows how to define a **deepmd** potential using the **ase** backend
in a **yaml** file:

```yaml
potential:
    provider: deepmd
    method: default
    parameters:
        model: ./graph.pb
```

See {ref}`potential-examples` section for more details.

## Executor

The **executor** specifies the concrete calculation. Its provider determines
the software, while `method` selects single point, minimization, molecular
dynamics, dimer, or a path method such as NEB.

The example below shows how to define an **executor** in a **yaml** file:

```yaml
executor:
    provider: ase
    method: md
    parameters:
        ensemble: nvt
        temp: 600 # temperature, Kelvin
        timestep: 1.0 # fs
        steps: 100
```

## Scheduler

With **potential** and **executor** defined, we can run simulations on local
machines (directly in the command line). However, simulations, under most
circumstances, would be really heavy even by MLIPs (imagine a 10 ns molecular
dynamics). The simulations would ideally be dispatched to high performace clusters
(HPCs).

The example below shows how to define a **scheduler** in a **yaml** file:

```yaml
scheduler:
    provider: slurm
    parameters:
        partition: k2-hipri
        ntasks: 1
        time: "0:10:00"
        environs: "conda activate py37\n"
```

The scheduler above submits with `sbatch` on the current machine because the
transport defaults to `local`. To stage the calculation and submit over SSH,
add the nested SSH transport (install it with `pip install gdpx[remote]`):

```yaml
scheduler:
    provider: slurm
    parameters:
        partition: compute
        time: "1:00:00"
    transport:
      provider: ssh
      parameters:
        hostname: cluster.example
        remote_wdir: /scratch/user/gdpx
```

Use `provider: direct` with the same SSH transport to run synchronously on a
remote machine without a queue. See {ref}`scheduler-transport` for all four
configurations.

## Runtime

A runtime is the complete executable unit: one potential, one executor,
optional modifiers, and one scheduler. Use explicit lists for independent
runtimes and explicit nested lists for runtime chains; gdpx does not infer a
Cartesian product between components.

:::{note}
If **scheduler** is omitted, gdpx uses direct execution with local transport.
:::
