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

See the {ref}`MC operator reference <sampling-operators>` for available moves and
their configuration, particle selection, and acceptance rules.
Hybrid MC uses the same preset and custom `setup.ensemble` configuration as
standard MC. The ensemble controls the MC acceptance rules; the MD runtime has
its own thermodynamic settings.

## Examples

This complete canonical input alternates 20 NVT MD steps and five Cu displacement
proposals per cycle, for five cycles at 1200 K. It uses ASE EMT and the Cu32
structure in `examples/monte_carlo/assets/cu32.xyz`.

```{literalinclude} ../../../examples/monte_carlo/hybrid-canonical.yaml
:language: yaml
:caption: examples/monte_carlo/hybrid-canonical.yaml
```

Run from the repository root:

```shell
gdp -d run-mc-hybrid-canonical compute examples/monte_carlo/assets/cu32.xyz --runtime examples/monte_carlo/hybrid-canonical.yaml
python examples/monte_carlo/inspect_run.py run-mc-hybrid-canonical/cand0
```

The example pages explain {doc}`canonical displacement <hmc/examples/canonical>`,
{doc}`semi-grand-canonical identity changes <hmc/examples/semi-grand-canonical>`,
and {doc}`Cu(111) oxidation with xreac <hmc/examples/cu111-oxidation-xreac>`.

```{toctree}
:maxdepth: 2
:hidden:

hmc/examples/index
```

## Configure the cycle

Run `gdp compute INPUT --runtime hybrid.yaml` with `executor.provider: gdpx`
and `executor.method: hybrid_monte_carlo`. Each input starts an independent chain.
The following paths are relative to `executor.parameters`.

| Setting | Role |
| --- | --- |
| `setup.ensemble` | MC thermodynamic settings |
| `setup.operators` | MC proposal distribution |
| `setup.evaluator.executor` | Single-point initialization using the outer potential and modifiers |
| A `molecular_dynamics` stage runtime | Runs one MD segment |
| A `monte_carlo` stage runtime | Evaluates that stage's proposed structures |

The initial evaluator defaults to direct execution; `setup.evaluator.scheduler`
can schedule its calculations independently. Each stage retains its own full
runtime, including potential, modifiers, and scheduler. Keep stage potentials
and energy references consistent.

`setup.cycle` lists explicit stages in execution order:

- `method: molecular_dynamics` runs its inline `runtime` once and takes the
  final structure forward. Its executor method must be `md`.
- `method: monte_carlo` performs the stage's `steps` proposals using
  `setup.operators` and its inline single-point runtime. Its executor method
  must be `spc`.

Each MC block selects operators using their normalized `probability` weights.
`stop.steps` counts complete passes through the cycle. An optional
`stop.earlystop` applies the shared MC early-stop criteria. Each
MC stage has its own proposal count, so repeated MC stages may use different lengths.
Repeating a stage executes it again in a separate calculation directory.
`output.checkpoint.period` controls checkpoints in completed-cycle units.

The MD runtime's `executor.parameters.stop.steps` counts integration steps;
`setup.timestep` is in fs. `setup.regulator.targets.temperature` is in kelvin
and should agree with the MC ensemble temperature. Langevin `friction` is in
fs⁻¹. `setup.velocities.seed` controls velocity initialization and the MD
`random_seed` controls its random generator. The sampling executor's
`random_seed` controls proposals and acceptance. Existing nonzero velocities are reused by default. These seeds
make the demos reproducible but do not establish statistical convergence.

Preset ensembles automatically disable MC geometry filtering to preserve
detailed balance. The `custom` ensemble may enable it for structure search.
The examples use one proposal attempt. An invalid hybrid MC proposal consumes
its place in the MC block and keeps the current state.

## Outputs and restart

Each chain uses its own `candN` directory. For a five-cycle run with
`output.trajectory.period: 1`:

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

The MD runtime's `output.trajectory.period: 5` saves frames within MD calculations. Hybrid
MC retains all stage directories. `output.checkpoint.period: 1` requests a
checkpoint after every full cycle. A pending stage also forces a checkpoint
when its cycle is committed.

Resume by rerunning the same command with the same output directory and inputs.
A pending calculation retains its cycle, stage, and MC proposal index,
so completed work is not redrawn. Completed cycles resume from their checkpoint.
Use a new output directory when changing the cycle, runtimes, or operators.
Old exploration inputs and checkpoints are not accepted. Move the initial
structure to the compute inputs, use `executor.parameters.setup/output/stop`,
and start a new compute directory. The initial and final states are always
written even with sparse trajectory output. Compute collects chain endpoints
into `results/end_frames.xyz`.
