(monte-carlo)=

# Standard Monte Carlo

Standard Monte Carlo (MC) proposes structural changes and accepts or rejects them using
single-point energies from a runtime. Run it through `gdp compute INPUT --runtime mc.yaml`
with `executor.provider: native` and `executor.method: monte_carlo`.
Each input structure starts an independent chain. Settings below are relative to
`executor.parameters`: `ensemble` defines the thermodynamic settings;
`setup` contains operators and the optional evaluator;
`output` controls output cadence; `stop` controls the step budget and early stopping.
MC defaults to an ASE single-point evaluator using the outer potential and
modifiers. Optional `setup.evaluation` accepts `provider` (default `ase`),
`scheduler`, `output`, `stop`, `random_seed`, and provider-specific setup fields
directly. Standard MC always selects `spc`; omit `method`, `parameters`, and
`setup` wrappers. The evaluator scheduler defaults to direct execution
independently of the outer scheduler. Incompatible potential backends fail
during preparation. For example, to customize only the evaluator scheduler:

```yaml
setup:
  evaluation:
    scheduler:
      provider: pbs
  operators:
  - method: move
    particles: [Cu]
    max_disp: 0.2
```

The {doc}`examples <mc/examples/index>` use ASE EMT, so no model download or
external simulation executable is needed.

| Ensemble | Fixed quantities | What changes | Operators |
| --- | --- | --- | --- |
| `nvt` (canonical) | Species counts, volume, temperature | Positions | `move`, `rotate`; `swap` for alloys |
| `suvt` (semi-grand canonical) | Total atom count, volume, temperature, chemical-potential differences | Species identities and optionally positions | `swap_type`, optionally `move` |
| `uvt` (grand canonical) | Volume, temperature, reservoir chemical potential | Particle count and optionally positions | `biased_volume_exchange`, `cavity_exchange`; optionally `move` |

`swap` exchanges the positions of unlike particles
and preserves composition; `swap_type` changes one atom's element and therefore
changes composition. The latter currently supports individual atoms only.

The {ref}`MC operator reference <sampling-operators>` covers `move`, `rotate`,
`swap`, `swap_type`, `biased_volume_exchange`, and `cavity_exchange`.
Additional structure-search proposals are documented under
{ref}`basin-hopping operators <bh-operators>`. MC and BH share implementations,
but accepting the same configuration does not make their sampling equivalent.

## Detailed balance

Equilibrium MC requires detailed balance for the target ensemble. The proposal
must allow the reverse transition, and acceptance must account for any
asymmetry in proposal probabilities. `swap_type` includes the reverse/forward
proposal-count ratio for its species-first selection. Disable ordinary geometry
filtering, use a single proposal attempt, and retain rejected and invalid steps
in the sampled trajectory. Use single-point energies without relaxation.

The two biased exchange operators remain documented here with explicit
limitations: `biased_volume_exchange` substitutes an estimated empty volume
while drawing positions from the ordinary region generator; `cavity_exchange`
uses cavity-count factors and falls back to an unscreened point when no cavity
qualifies. These implementations need separate detailed-balance validation
before quantitative equilibrium use. Preset ensemble checks alone do not
establish that validation.

Define thermodynamic settings once under `parameters.ensemble`:

```yaml
parameters:
  ensemble:
    method: suvt
    temperature: 600.0
    chemical_potentials:
      Cu: 0.0
      Ni: 0.2
  setup:
    operators:
      - method: swap_type
        particles: [Cu, Ni]
```

MC supports `nvt`, `uvt`, and `suvt`. `nve` and `npt` are MD options;
MC has no microcanonical acceptance or volume-change implementation.
The old `custom` input is removed; use basin hopping for structure-search
proposals with per-operator settings. For independent compute chains with
mandatory minimisation and geometry filters, use {doc}`biased-mc`.

See {doc}`ensembles` for the ensemble fields, units, and HMC inheritance.

## Examples

The {doc}`EMT examples <mc/examples/index>` provide complete inputs and commands
for canonical, semi-grand-canonical, and grand-canonical MC with single-point energies.
The {doc}`hybrid MC examples <hmc/examples/index>` combine short EMT molecular
dynamics segments with displacement or identity-change MC blocks. Their guide
covers cycle configuration, the child executor settings, cycle-level output,
and restart behaviour.

```{toctree}
:maxdepth: 2
:hidden:

mc/examples/index
```

## Proposal settings

`probability` is a relative operator-selection weight; the weights are
normalised automatically. Preset ensembles define temperature and chemical
potentials once under `parameters.ensemble`. Use consistent regions when combining moves and exchange; see
{ref}`region-definitions` for region definitions.

Standard MC automatically disables proposal distance checks for all three
supported ensembles. It rejects `skip_distance_check: false`
because geometry filtering changes the proposal distribution and breaks
detailed balance. The energy evaluation decides whether each trial is
accepted.

`max_random_attempts: 1` avoids retrying until a valid move is found.
Failed proposals count as steps and retain the current state.

MC assigns distinct atomic tags by default. If `setup.ignore_atoms_tags: false`,
provide distinct tags for independent atoms; atoms sharing a tag are treated
as one particle. `swap_type` requires single-atom particles.

## Inspect the results

Each candidate's `candN/mc.out` groups setup, initialization, each MC step, and completion in
boxes. Worker evaluations appear inside the corresponding initialization or
step box. Each step reports its operator, acceptance decision, previous and
trial energies, current energy, and atom count. Waiting evaluations and invalid
proposals are labelled separately. Use `gdp --debug ...` for detailed proposal diagnostics.
Restart messages identify the checkpoint or pending step being resumed. Resuming
appends to this log. The terminal and top-level `gdp.out` show batch status and
result summaries; candidate calculation progress goes to the candidate log.

Each chain directory (`cand0`, `cand1`, … under the compute directory) contains:

- `mc.out`: candidate progress, evaluator diagnostics, and failure details.
- `mc.xyz`: the initial state followed by the current state after each completed
  retained MC step, including repeated states on rejection. The initial and final
  states are always retained. With the default output period of one, keep repetitions when
  computing averages.
- `mc_attempts.xyz`: proposed structures sent for evaluation, including rejected
  trials; this is not the accepted-state trajectory.
- `opstat.txt`: operator, diagnostic, current atom count, acceptance flag, and
  previous/trial energies.
- `calculations/step.NNNN/`: retained runtime calculations.

`stop.steps: 100` is the attempted-step budget. An optional
`stop.earlystop` uses the existing early-stop criteria. The
examples retain every step with `output.trajectory.period: 1`; increasing it
thins `mc.xyz` and prunes intermediate ordinary-MC calculation directories.
`output.checkpoint.period` controls checkpoint frequency independently. Completion
is determined from saved progress, not the number of trajectory frames.
Compute also collects each chain’s final structure into `results/end_frames.xyz`.

Inspect energy and composition from the repository root:

```shell
python examples/monte_carlo/inspect_run.py run-mc-canonical/cand0
python examples/monte_carlo/inspect_run.py run-mc-semi-grand-canonical/cand0
python examples/monte_carlo/inspect_run.py run-mc-grand-canonical/cand0
```

Canonical frames should all contain Cu32; semi-grand-canonical frames should all
contain 32 atoms but may have different Cu/Ni counts; grand-canonical frames
should retain one Au while Cu counts change. The cell stays fixed in all
three. For quantitative work, assess equilibration, autocorrelation, acceptance
rates, and sensitivity to run length after validating the sampling algorithm.

## Checkpoints and restart

Restart with the same configuration and output directory. MC stores versioned
JSON metadata and uncompressed NumPy `.npz` arrays, without pickle. Arrays,
constraints, cached results, operator configuration, and random state are
preserved without copying the entire structure before serialization.

`output.checkpoint.period` controls ordinary checkpoint frequency. `output.checkpoint.keep` defaults to two, retaining the latest and
previous committed checkpoints; a damaged latest snapshot falls
back to the previous one. A queued move also retains its pending state until
its resolution is committed, forcing a checkpoint even between ordinary
checkpoint steps. Hybrid MC commits this state after its complete cycle.
Old `gdp explore` configurations and run directories are not supported. Move
structures to compute inputs, migrate settings to `executor.parameters`, and
start a new compute directory. `gdp compute status`, `resubmit`, and `collect`
use the normal compute lifecycle; status never submits child evaluations.

## Application

1. {ref}`ref-acs-catal-2022-xu`
