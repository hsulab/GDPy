# demo and lifecycle

`gdp compute` applies a runtime to structures read from an ASE-readable file,
such as a multi-frame extended XYZ file. Put global options before `compute`:

```shell
gdp -d results -r runtime.yaml compute structures.xyz
```

`-r` (`--runtime`) selects the runtime YAML; `-d` selects the output directory.
Use a different directory when changing the structures or runtime: an existing
compute plan rejects conflicting inputs.

(compute-copper-dimers-example)=

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

## Independent runtimes

A runtime file can contain a list of complete runtime mappings to evaluate the
same structures independently with different settings. Each gets its own
worker directory. The current `gdp compute` lifecycle accepts a mapping or a
flat list of mappings; it does not accept nested sequential chains or perform
a Cartesian product of implicit list settings. An explicit
`executor.broadcast` does expand executor parameters as described in
{doc}`runtime`. Each member of a flat runtime list may define its own broadcast,
and the resolved runtimes are flattened in source order. Use the workflow layer
for sequential work.

See {doc}`tasks/index` for task examples and {doc}`schedulers` for queue jobs.
