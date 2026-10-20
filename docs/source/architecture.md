# gdpx architecture

gdpx separates integration ownership from runtime responsibility. Dependencies
flow in one direction:

```
core/domain -> providers -> execution -> exploration -> workflow -> cli
```

The data, structures, modifiers, and analysis packages are domain libraries and
must not import workflow or CLI modules. Providers own software-specific
knowledge as one vertical plugin: model specifications, materializers,
executable adapters, trainers, and artifacts. For example, DeepMD owns both
its ASE-calculator and LAMMPS-potential materializations, while ASE and LAMMPS
own the corresponding execution mechanisms. Exploration algorithms may
request calculations from the execution service, but execution must never
depend on exploration.

`gdpx.sampling` contains shared sampling primitives. It owns reusable
Monte Carlo moves, acceptance rules, and geometry preparation; it imports no
providers, execution, exploration modules, or application layers.
Basin hopping lives independently in
`gdpx.exploration.basin_hopping`. MC and hybrid MC are execution methods;
all three reuse these sampling primitives.

Moves borrow an exclusively owned `Atoms` object and edit it in place. A
`MoveProposal` holds independent undo values only for changed or removed rows;
acceptance policies read its metadata and energies without mutating structures.
The caller must commit or roll back every successful proposal. A context manager
rolls back automatically on exit, including exceptions. Never retain an alias
as an unchanged accepted structure while a proposal is active. Insertion and
deletion can reallocate ASE arrays, so array views must not escape a transaction.

Workers capture trial inputs before the caller releases the borrow. Relaxed
results belong to execution and replace the accepted structure only after
acceptance. Pending jobs persist the accepted frame and decision metadata, not
live operator references or undo logs. Transaction and checkpoint I/O belongs to execution.

Providers are stateless capability catalogs. Scientific configuration is held
in immutable specifications; mutable state belongs to executors, runtimes, and
execution services. Cross-provider integration uses consumer-defined
materialization interfaces rather than imports of another provider's concrete
implementation.

## Execution and exploration

Execution performs a concrete request: evaluate these structures, optimize
this geometry, run this trajectory, or search this supplied transition-state
problem. Exploration owns the adaptive policy that decides which request to
make next from previous results. Consequently exploration receives an
`ExecutionService`; execution never imports an exploration strategy.

A dimer search is a local transition-state executor because it runs one
replica. NEB and string methods are path transition-state executors because
their concrete request contains a path of replicas. They are exploration only
when another algorithm adaptively chooses or changes those requests.

## Public boundary

The implementation packages are now `providers`, `execution`,
`exploration`, `sampling`, `workflow`, `structures`, `analysis`, `modifiers`, and
`data`. Version 0.1 removes the former compatibility packages and global
registry catalog; integrations enter through the `gdpx.providers` plugin
group. Runtime input requires explicit `potential`/`executor` component
sections and uses the current schema without a user-authored `schema_version`.
Serialized runtimes retain their version. Calculators are created only after
the executor target is known, and unsupported target/modifier combinations fail
explicitly.

Scheduling has two provider boundaries: a scheduler chooses the direct or
queued submission backend, and its transport chooses the current host or SSH.
This keeps host location independent of queue semantics and permits every
combination.

Monte Carlo and hybrid MC are calculator-independent single-structure executors
under `gdpx.execution.monte_carlo`, registered by the `native` provider. They
inherit one fixed potential and compose ordinary child executors and participate in compute's
persisted lifecycle. `SingleStructureExecutor` defines the worker-facing
contract shared with `BaseDriver`; pending sampling work returns `RunState.PENDING`.
Each compute input owns a separate chain. Proposal transactions, accepted-state
serialization, and checkpoint utilities reside in execution and are shared by
sampling and structure-search algorithms.

Native HMC and simulated annealing use list-based `parameters.setup`; ordinary
MD, minimisation, and MC retain mapping-based setup. `gdpx.execution.composite`
owns ordered child execution, invocation seeds, and durable stage boundaries.
Population SA reuses the native annealing executor for each selected candidate.
