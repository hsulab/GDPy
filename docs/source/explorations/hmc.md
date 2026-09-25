(hybrid-monte-carlo)=

# Hybrid Monte Carlo

`hybrid_monte_carlo` alternates worker calculations and blocks of MC proposals.
The examples combine short NVT molecular-dynamics segments with Cu
displacements or Cu/Ni identity changes using EMT, and oxygen exchange on
Cu(111) using xreac. Their short trajectories and small cells illustrate the
workflow rather than equilibration.

GDPy accepts the final structure of each MD segment directly. Only the
subsequent MC proposals undergo a Metropolis test using potential-energy
changes. There is no accept/reject test of the complete MD trajectory based on
total Hamiltonian change, so this implementation is a mixed MD/MC workflow,
not a Hamiltonian Monte Carlo sampler. The {ref}`MC sampling limitations
<monte-carlo>` also apply to its MC operators.

See the shared {ref}`sampling-operators` reference for available moves and
their configuration, particle selection, and acceptance rules.
Hybrid MC uses the same preset and custom `system.ensemble` configuration as
standard MC. The ensemble controls the MC acceptance rules; the MD runtime has
its own thermodynamic settings.

## Examples

The {doc}`examples <hmc/examples/index>` demonstrate canonical displacement,
semi-grand-canonical identity changes, and Cu(111) oxidation with xreac.

```{toctree}
:maxdepth: 2
:hidden:

hmc/examples/index
```

## Configure the cycle

Hybrid MC places the initial structure and MC ensemble under `system`, and its
cycle budget, operators, and ordered cycle under `strategy`. Its three
runtime roles are distinct:

| Setting | Role |
| --- | --- |
| `runtime` | Evaluates the starting structure before the first cycle |
| A `molecular_dynamics` stage runtime | Runs one MD segment |
| A `monte_carlo` stage runtime | Evaluates that stage's proposed structures |

The YAML anchor `&single_point` and alias `*single_point` reuse one runtime for
initialization and MC evaluation. Both examples contain every required runtime,
so no `--runtime` argument is needed. If supplied, `--runtime` replaces only the
initialization runtime. Keep all stage potentials and energy references
consistent.

`strategy.cycle` lists explicit stages in execution order:

- `method: molecular_dynamics` runs its inline `runtime` once and takes the
  final structure forward. Its executor method must be `md`.
- `method: monte_carlo` performs the stage's `steps` proposals using
  `strategy.operators` and its inline single-point runtime. Its executor method
  must be `spc`.

Each MC block selects operators using their normalized `probability` weights.
`strategy.steps` counts complete passes through the cycle. Each MC stage
has its own proposal count, so repeated MC stages may use different lengths.
Repeating a stage executes it again in a separate calculation directory.
`strategy.ckpt_period` controls checkpoints in completed-cycle units.

The MD executor's `steps` counts integration steps and `timestep` is in fs.
`temp` sets its temperature in kelvin and should agree with the MC operators'
ensemble temperature when using a preset ensemble. Langevin `friction` is in
fs⁻¹. `velocity_seed` controls velocity initialization and the MD `random_seed`
controls its random generator; the top-level `random_seed` controls MC proposals
and acceptance. Existing nonzero velocities are reused by default. These seeds
make the demos reproducible but do not establish statistical convergence.

Preset ensembles automatically disable MC geometry filtering to preserve
detailed balance. The `custom` ensemble may enable it for structure search.
The examples use one proposal attempt. An invalid hybrid MC proposal consumes
its place in the MC block and keeps the current state.

## Outputs and restart

For an uninterrupted five-cycle run:

- `mc.xyz` contains **six frames**: the initial state and one state after each
  full MD/MC cycle. Intermediate MD frames and individual MC decisions are not
  separate frames in this file.
- `mcmoves.log` contains **25 MC decision rows** plus its header. For these
  single-block cycles, the step column labels MC attempts 0–24, not cycles.
  MD segments do not add decision rows.
- `mc_attempts.xyz` contains the initial structure only in the current hybrid
  implementation; use the per-proposal calculations to inspect MC trials.
- `calculations/step.0000/` stores the initial evaluation.
- `calculations/step.0001/procedure.0000/` stores the first MD segment.
- `calculations/step.0001/procedure.0001/proposal.0000/` stores the first MC
  evaluation. The remaining cycle and proposal indices follow the same pattern.

The MD runtime's `dump_period: 5` saves frames within MD calculations. Hybrid
MC retains all stage directories. `strategy.ckpt_period: 1` requests a
checkpoint after every full cycle. A pending stage also forces a checkpoint
when its cycle is committed.

Resume by rerunning the same command with the same output directory and inputs.
A pending calculation retains its cycle, stage, and MC proposal index,
so completed work is not redrawn. Completed cycles resume from their checkpoint.
Use a new output directory when changing the cycle, runtimes, or operators.
Flat hybrid inputs using `builder`, `operators`, `extra_workers`, or
`num_mcmoves` are no longer accepted; migrate them to `system` and `strategy`
and start a new output directory.
