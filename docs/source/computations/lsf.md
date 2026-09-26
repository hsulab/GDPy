(lsf)=

# lsf

The `lsf` scheduler writes `#BSUB` directives, submits with `bsub`, and checks
active jobs with `bjobs`. Its parameter keys follow LSF's short option syntax.

```yaml
scheduler:
  provider: lsf
  parameters:
    q: normal
    n: 4
    W: "1:00"
    R: "span[hosts=1]"
    M: 8G
    environs: |
      source /path/to/environment.sh
dispatch:
  batch_size: 10
```

Queue names, resource expressions, memory units, and launcher commands vary
between LSF sites. Configure an external MPI launcher through `machine_prefix`
when required.

LSF currently supports only `concurrent_tasks: 1`; calculations assigned to a
batch execute sequentially. To run independent calculations concurrently,
submit smaller batches as separate LSF jobs.

Submission is local unless an SSH transport is nested under `scheduler`. See
{ref}`scheduler-transport` for transport behavior and the common queue
lifecycle.
