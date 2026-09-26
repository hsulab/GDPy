(pbs)=

# pbs

The `pbs` scheduler writes `#PBS` directives, submits with `qsub`, and checks
active jobs with `qstat`. Its parameter keys follow PBS's short option syntax.

```yaml
scheduler:
  provider: pbs
  parameters:
    A: project
    q: workq
    l: "select=1:ncpus=4:mem=8gb,walltime=01:00:00"
    environs: |
      source /path/to/environment.sh
dispatch:
  batch_size: 10
```

The `l` value is passed through as one PBS resource expression. Queue names,
select syntax, memory units, and launcher commands vary between PBS sites.
Configure an external MPI launcher through `machine_prefix` when required.

PBS currently supports only `concurrent_tasks: 1`; calculations assigned to a
batch execute sequentially. To run independent calculations concurrently,
submit smaller batches as separate PBS jobs.

Submission is local unless an SSH transport is nested under `scheduler`. See
{ref}`scheduler-transport` for transport behavior and the common queue
lifecycle.
