(computations)=

# Computations

Every computation is described by one schema-v3 runtime: a potential, an
executor, optional modifiers, and an optional scheduler. The potential defines
the model; the executor defines the software and calculation method.

```yaml
schema_version: 3
potential:
  provider: deepmd
  parameters:
    model: ./graph.pb
    type_list: [H, O]
executor:
  provider: ase
  method: md
  parameters:
    ensemble: nvt
    temp: 300
    timestep: 1.0
    steps: 1000
    dump_period: 10
```

With `scheduler` omitted, this calculation runs directly on the current machine.

Run it with:

```
gdp --runtime runtime.yaml compute structures.xyz
```

## Compute lifecycle

The lifecycle can be controlled explicitly:

```
gdp -d results -r runtime.yaml compute prepare structures.xyz
gdp -d results compute submit
gdp -d results compute status
gdp -d results compute resubmit --batch 0
gdp -d results compute collect
```

`prepare` writes a versioned plan without submitting work. `status` is
read-only, and resubmission is always explicit.

## Executors

Use `spc` for single-point evaluation, `min` for minimization, and `md`
for molecular dynamics. Executor parameters are flat settings such as
`steps`, `fmax`, `constraint`, `ensemble`, and `dump_period`.

Transition-state executors have two shapes. `dimer` is local and consumes
one replica; `neb` is a path executor and consumes endpoints or an ordered
image sequence.

## Schedulers and batching

The scheduler controls how work is dispatched, while its nested transport
controls whether commands run on this machine or over SSH. See
{ref}`scheduler-transport` for the four supported combinations and complete
examples. `options.batch_size` controls how many structures are assigned to a
queued task.

For multiple independent calculations, provide an explicit list of complete
runtimes. For sequential calculations, provide an explicit nested runtime
chain. GDPy does not broadcast components or construct a Cartesian grid.

```{toctree}
:hidden:

schedulers
```
