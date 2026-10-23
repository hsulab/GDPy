# Batching evaluations

Standard MC, biased MC, and HMC use a Monte Carlo worker to coordinate independent
chains. The worker advances each chain until it needs an energy evaluation or
MD segment, collects compatible requests, and submits them through the ordinary
calculation worker. MC proposals and HMC stage order remain sequential within
each chain.

## Configuration

These fields belong at the top level of the sampling runtime:

```yaml
dispatch:
  batch_size: 4
scheduler:
  provider: slurm
  parameters:
    concurrent_tasks: 2
```

With ten ready, compatible requests, this produces jobs containing four, four,
and two basic calculations, with up to two calculations running concurrently
per job. `batch_size` defaults to one. Local direct execution runs basic
calculations sequentially.

- Standard MC requests single-point (`spc`) calculations.
- Biased MC requests minimisations (`min`).
- HMC requests MD segments and the single-point evaluations of its MC stages.

The top-level scheduler is inherited by evaluations. An explicit
`setup.evaluation.scheduler` overrides it for MC. In HMC, a stage's `scheduler`
overrides the parent, and the MC stage's `evaluation.scheduler` can override
that stage scheduler. Different potentials, modifiers, execution settings,
or scheduler configurations are placed in separate batches. Each chain keeps
its fixed potential and modifiers.

The coordinator runs locally; whole MC chains and native HMC/MC stages are not
submitted to the queue. Runtime variants remain separate Monte Carlo workers.
Batching occurs among chains managed by the same Monte Carlo worker.

## Progress and restart

The coordinator advances ready chains independently. It does not wait for all
chains to reach the same MC step. Queued calculations can leave some chains
pending while other batches complete. Repeating the compute command, or using
`gdp compute resubmit`, polls evaluations and advances ready chains. A status
check reports the chain coordinator's completion state; it does not apply
acceptance decisions or generate new proposals.

Requests have stable identities based on their chain, step, and HMC stage.
Their structures, runtimes, random seeds, and batch assignments are saved
before submission. On resume, the coordinator reuses those assignments and the
calculation worker's saved jobs. Acceptance decisions and chain checkpoints
remain owned by the MC/HMC executors.

Each chain retains `mc.out`, trajectories, and checkpoints in its usual directory.
Single-input runs use the worker directory directly; multiple inputs use `candN`.
Basic calculations write directly to the corresponding `steps/step.NNNN/`
or HMC stage directory. Its `evaluation.json` identifies the request and batch.
There is no second copy of calculation outputs in the coordination directory.
Calculation diagnostics are also appended to the owning chain's `mc.out`.

The worker's `_meta/` directory holds coordinator records. Its sibling
`_requests/` directory stores evaluation records:

- `requests.json`: request identities, runtime references, and batch assignments.
- `runtimes/`: one immutable configuration per distinct evaluation runtime.
- `requests/`: immutable input structures and completed endpoints.
- `batches/`: references to those inputs and runtimes, job records, scripts, and batch logs.

Batch workers read the original request inputs instead of saving additional
structure snapshots. A published endpoint is the completion record; there are
no separate readiness flags or completion lists to keep in sync.

Keep this directory for restart. Chain trajectory cadence can prune step outputs;
the completed endpoints and job records remain available without retaining full
calculation trajectories. Use a fresh
run directory when migrating from the previous nested-worker layout.
