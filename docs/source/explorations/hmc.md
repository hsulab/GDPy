(hybrid-monte-carlo)=

# Hybrid Monte Carlo

`hybrid_monte_carlo` alternates worker calculations and blocks of MC proposals.
These examples combine short NVT molecular-dynamics segments with either Cu
displacements or Cu/Ni identity changes, using EMT for every energy and force
calculation. Both reuse the periodic 32-atom structures from the other MC demos.

Each cycle runs **20 MD steps → 5 MC proposals**. Five cycles produce 100 MD
steps and 25 MC proposals, plus the initial single-point evaluation. The
1200 K temperature, short trajectories, and small cells keep the examples
inexpensive; the step budgets are not equilibration criteria.

GDPy accepts the final structure of each MD segment directly. Only the
subsequent MC proposals undergo a Metropolis test using potential-energy
changes. There is no accept/reject test of the complete MD trajectory based on
total Hamiltonian change, so this implementation is a mixed MD/MC workflow,
not a Hamiltonian Monte Carlo sampler. The {ref}`MC sampling limitations
<monte-carlo>` also apply to its MC operators.

See the shared {ref}`sampling-operators` reference for available moves and
their configuration, particle selection, and acceptance rules.

## Examples

The {doc}`EMT examples <hmc/examples/index>` demonstrate canonical displacement
moves and semi-grand-canonical identity changes interleaved with MD.

```{toctree}
:maxdepth: 2
:hidden:

hmc/examples/index
```

## Configure the procedure

Hybrid MC currently takes its method settings at the top level, without a
`recipe` wrapper. Its three runtime roles are distinct:

| Setting | Role |
| --- | --- |
| `runtime` | Evaluates the starting structure before the first cycle |
| `extra_workers.md` | Runs each MD segment |
| `extra_workers.mc` | Evaluates each proposed MC structure |

The YAML anchor `&single_point` and alias `*single_point` reuse one runtime for
initialization and MC evaluation. Both examples contain every required runtime,
so no `--runtime` argument is needed. If supplied, `--runtime` replaces only the
initialization runtime; it does not replace entries in `extra_workers`. Keep
all three potentials and their energy references consistent.

`procedure` lists operations in execution order:

- `worker_md` runs the runtime stored under `extra_workers.md` once and takes
  its final structure forward.
- `[monte_carlo, worker_mc]` performs `num_mcmoves` proposals using the operators
  and the runtime stored under `extra_workers.mc`.

Each MC block selects operators using their normalized `probability` weights.
`num_mcmoves` applies to each MC block, while `convergence.steps` counts complete
passes through `procedure`. Repeating a procedure entry executes it again in
a separate calculation directory. Use simple worker names such as `md` and
`mc`; the `worker_` prefix refers to the corresponding `extra_workers` key.

The MD executor's `steps` counts integration steps and `timestep` is in fs.
`temp` sets its temperature in kelvin and should agree with the MC operators'
`temperature`. Langevin `friction` is in fs⁻¹. `velocity_seed` controls velocity
initialization and the MD `random_seed` controls its random generator; the
top-level `random_seed` controls MC proposals and acceptance. Existing nonzero
velocities are reused by default. These seeds make the demos reproducible but
do not establish statistical convergence.

The MC operators disable geometry filtering and use one proposal attempt.
An invalid hybrid MC proposal consumes its place in the MC block and keeps
the current state; `should_retry` does not introduce retries inside that block.

## Outputs and restart

For an uninterrupted five-cycle run:

- `mc.xyz` contains **six frames**: the initial state and one state after each
  full MD/MC cycle. Intermediate MD frames and individual MC decisions are not
  separate frames in this file.
- `opstat.txt` contains **25 MC decision rows** plus its header. For these
  single-block procedures, the step column labels MC attempts 0–24, not cycles.
  MD segments do not add decision rows.
- `mc_attempts.xyz` contains the initial structure only in the current hybrid
  implementation; use the per-proposal calculations to inspect MC trials.
- `calculations/step.0000/` stores the initial evaluation.
- `calculations/step.0001/procedure.0000/excurs/` stores the first MD segment.
- `calculations/step.0001/procedure.0001/proposal.0000/` stores the first MC
  evaluation. The remaining cycle and proposal indices follow the same pattern.

The MD runtime's `dump_period: 5` saves frames within MD calculations. Hybrid
MC's top-level `dump_period` currently does not prune procedure directories;
all are retained. `ckpt_period: 1` requests a checkpoint after every full cycle.
A pending procedure also forces a checkpoint when its cycle is committed.

Resume by rerunning the same command with the same output directory and inputs.
A pending calculation retains its cycle, procedure entry, and MC proposal index,
so completed work is not redrawn. Completed cycles resume from their checkpoint.
Use a new output directory when changing the procedure, runtimes, or operators.
