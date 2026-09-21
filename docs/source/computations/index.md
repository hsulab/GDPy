(computations)=

# Computations

Every computation is described by one schema-v3 runtime: a potential, an
executor, optional modifiers, and an optional scheduler. The potential defines
the model; the executor defines the software and calculation method.

```yaml
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

## Simple example: copper dimers

The repository includes a ready-to-run example in `examples/compute/cu2_emt/`.
It minimizes three Cu2 dimers with ASE's built-in EMT potential, requiring no
model download. The input bond lengths are 2.0, 2.5, and 3.0 Å; all structures use
periodic 20 × 20 × 20 Å cells.

From the repository root:

```sh
gdp -d cu2-compute -r examples/compute/cu2_emt/runtime.yaml compute examples/compute/cu2_emt/structures.xyz
```

The example runs locally in one batch, with a force tolerance of 0.05 eV/Å and a
limit of 100 optimization steps per structure. The output summarizes steps,
energy, and `maxfrc` across the three calculations. Relaxed structures are saved
in `cu2-compute/results/end_frames.xyz`.

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

## Progress output

Workers report one aggregate box for the requested calculations, with counts of
finished, pending, and failed calculations. The same timestamped boxes appear
in `gdp compute` and inside basin hopping generation output. Candidate lists are
omitted, so output stays compact for large batches. Long-running local batches
report progress at most once every 30 seconds, after a calculation completes.

Collected results include minimum, average (`avg`), and maximum simulation steps, energy
`[eV]`, and maximum atomic force (`maxfrc`) `[eV/Å]`. Steps come from the last saved frame's
simulation step, rather than the number of saved frames. Single-point steps and
unavailable metadata display `—` (`-` on ASCII terminals). Statistics describe
the available collected results; the box reports how many results contribute.
Force statistics respect atomic constraints and use cached forces.

A finished calculation is not necessarily converged.
Reporting does not evaluate calculators, copy structures, or read extra
trajectories. Routine worker diagnostics remain available at DEBUG level.

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
