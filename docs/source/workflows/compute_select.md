# compute and select

The `compute` operation consumes structures and one runtime, or an explicit
list of runtimes. The runtime already contains the potential, executor,
modifiers, and scheduler needed to execute the calculation. Its output can be
passed to `extract` and then to `select`.

A typical graph is:

```
structures -> compute(runtime) -> extract -> select
```

For a sequence such as pre-relaxation followed by a higher-accuracy
calculation, define a `runtime_chain`; see the complete
{doc}`runtime-chain guide <../computations/runtime-chain>`. Do not express
alternatives by placing lists inside potential or executor fields; list the
complete runtimes instead.

For an executor-parameter sweep, put an explicit `broadcast` on the executor
resource. The keys are relative to its `parameters` mapping:

```yaml
resources:
  md:
    __type__: executor
    options:
      provider: ase
      method: md
      parameters:
        ensemble: nvt
        steps: 1000
      broadcast:
        temp: [300, 600, 900]
```

A compute step using a runtime built from `md` receives three independent
runtimes. See the {doc}`broadcast guide <../computations/runtime-broadcast>`
for Cartesian expansion and validation rules.

This explicit boundary ensures that every submitted task has exactly one
potential/executor pairing and can be serialized as schema version 4.
