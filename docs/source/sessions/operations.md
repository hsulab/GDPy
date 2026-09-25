(operations)=

# resources and steps

Resources hold configuration or lightweight values. Built-in resource types
include `potential`, `executor`, `runtime`, `runtime_chain`, `scheduler`,
`builder`, `selector`, `trainer`, and `validator`.

Steps consume resources or earlier step results. Common types include `build`,
`read_stru`, `write_stru`, `compute`, `extract`, `select`, `react`, `train`, and
`validate`. Each step stores its state beneath the workflow run directory so a
run can resume.

Dependencies are ordinary names under `inputs`; there is no interpolation
syntax:

```yaml
steps:
  relax:
    __type__: compute
    inputs:
      structures: structures
      runtime: local_relaxation
    options:
      extract_data: true
```

The input key is the constructor argument and the value is the resource or step
name. Put literals under `options`. This separation makes the dependency graph
available to `gdp workflow plan` and `gdp workflow graph` without constructing
or running nodes.
