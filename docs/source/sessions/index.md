(sessions)=

# sessions

A session is a declarative graph of variables and operations. Variables hold
configuration or lightweight values; operations perform computation and pass
their results to later nodes.

Run a session with:

```
gdp session workflow.yaml
```

## Runtime variables

Computations use provider components and a complete runtime:

```
variables:
  emt:
    type: potential
    provider: emt
  relax:
    type: executor
    provider: ase
    method: min
    parameters:
      fmax: 0.05
      steps: 100
  local_relaxation:
    type: runtime
    potential: ${vx:emt}
    executor: ${vx:relax}
```

Use `runtime_chain` with an explicit `runtimes` list for ordered stages.
Independent runtimes are also written as explicit lists; component broadcasting
and implicit Cartesian products are not supported.

References such as `${vx:emt}` are resolved by the workflow session. Runtime
configuration may omit `schema_version` to use the current schema, as shown
in {ref}`computations`. Serialized runtimes retain an explicit version.

```{toctree}
:maxdepth: 1

operations
```
