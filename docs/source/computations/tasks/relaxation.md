# fixed-cell relaxation

Use `min` to relax atomic positions while holding the simulation cell fixed.

Generate the input files as described in {doc}`index`, then use
`examples/compute/tasks/relaxation.yaml`:

```yaml
potential:
  provider: emt
executor:
  provider: ase
  method: min
  parameters:
    output:
      trajectory:
        period: 1
    stop:
      fmax: 0.05
      steps: 100
```

From the repository root:

```shell
gdp -d min-demo -r examples/compute/tasks/relaxation.yaml compute examples/compute/tasks/dimers.xyz
```

`fmax` is the maximum-force tolerance in eV/Å; `steps` limits the optimizer
iterations. The default ASE optimizer is BFGS. `dump_period: 1` saves every
step. The three initial Cu–Cu separations should relax toward the same EMT
minimum. Inspect `min-demo/results/end_frames.xyz` and the convergence report;
reaching the step limit alone does not establish convergence.

To freeze atoms, set `executor.parameters.setup.constraint`. For example,
``constraint: "`index 0`"`` fixes the first atom, while
``constraint: "`zbot 4`"`` fixes the bottom four atoms. See
{doc}`../../groups-and-constraints` for selection syntax and indexing conventions.

To change machine resources, add a scheduler as described in
{doc}`../schedulers`.
