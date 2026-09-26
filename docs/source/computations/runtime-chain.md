# runtime chains

A runtime chain runs several calculations in order, passing each stage's output
structures to the next stage. Chains are workflow resources; standalone
`gdp compute` accepts independent runtimes but not sequential chains.

This complete workflow equilibrates and then continues production MD at 300 K
and 600 K:

```yaml
parameters:
  structures: ./structures.xyz

workflow:
  mode: once
  targets: trajectories

resources:
  potential:
    __type__: potential
    options:
      provider: emt

  equilibrate_md:
    __type__: executor
    options:
      provider: ase
      method: md
      parameters:
        ensemble: nvt
        steps: 1000
      broadcast:
        temp: [300, 600]

  production_md:
    __type__: executor
    options:
      provider: ase
      method: md
      parameters:
        ensemble: nvt
        steps: 10000
      broadcast:
        temp: [300, 600]

  equilibrate:
    __type__: runtime
    inputs:
      potential: potential
      executor: equilibrate_md

  production:
    __type__: runtime
    inputs:
      potential: potential
      executor: production_md

  md_chain:
    __type__: runtime_chain
    inputs:
      runtimes: [equilibrate, production]

steps:
  structures:
    __type__: read_stru
    options:
      fname: {$param: structures}

  trajectories:
    __type__: compute_chain
    inputs:
      structures: structures
      runtime_chain: md_chain
```

Run it with:

```shell
gdp workflow run workflow.yaml
```

The two broadcast stages are paired by index, producing a 300 K chain and a
600 K chain. A scalar stage is repeated across every chain variant. If multiple
stages broadcast, every non-scalar stage must expand to the same number of
variants; mismatched widths are rejected while the workflow is constructed.

Use a chain only when the stages are sequential. To evaluate the same input
structures independently with several settings, use {doc}`runtime-broadcast`
with the ordinary `compute` step instead.
