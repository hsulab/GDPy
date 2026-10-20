# Simulated annealing

Run a cooling trajectory followed by minimisation on a single potential surface
with `executor.provider: native` and `executor.method: simulated_annealing`.
Each input structure starts an independent task.

```{literalinclude} ../../../../examples/global_optimisation/runtimes/emt_sa.yaml
:language: yaml
```

Run from the repository root:

```shell
gdp -d run-sa compute examples/global_optimisation/assets/cu8_seeds.xyz --runtime examples/global_optimisation/runtimes/emt_sa.yaml
```

`parameters.setup` contains exactly two child executors: MD and minimisation.
MD's regulator temperature is the start of the linear cooling ramp; `setup.tend`
is its final temperature. MD `stop.steps` counts integration steps. The parent
`stop.steps` counts complete cooling/minimisation passes and defaults to one.
The minimisation child has its own stopping criteria.

One shared potential supplies all energies and forces. Children may have their
own schedulers; omitted child schedulers run directly. This implementation
supports ASE Berendsen and Langevin cooling and requires no modifiers.

Each `candN/annealing.xyz` contains the initial state and completed-pass minima.
Children run under `candN/calculations/step.NNNN/procedure.NNNN/` and retain
individual trajectories. Compute collects final minima in
`results/end_frames.xyz`. Checkpoints commit child boundaries so repeating the
same command resumes pending work. Use a new directory for changed inputs.

For repeated population search and selection of diverse minima, see
{doc}`../../global_optimisation/simulated-annealing`.
