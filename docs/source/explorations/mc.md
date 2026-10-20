(monte-carlo)=

# Monte Carlo (MC)

Monte Carlo proposes structural changes and accepts or rejects them using
single-point energies from a runtime. Configure the initial structure and
thermodynamic ensemble under `system`, and run settings and proposals under `strategy`.
The {doc}`examples <mc/examples/index>` use ASE EMT, so no model download or
external simulation executable is needed.

| Ensemble | Fixed quantities | What changes | Operators |
| --- | --- | --- | --- |
| Canonical (NVT) | Species counts, volume, temperature | Positions | `move`, `rotate`; `swap` for alloys |
| Semi-grand-canonical | Total atom count, volume, temperature, chemical-potential differences | Species identities and optionally positions | `swap_type`, optionally `move` |
| Grand-canonical (μVT) | Volume, temperature, reservoir chemical potential | Particle count and optionally positions | `biased_volume_exchange`, `cavity_exchange`; optionally `move` |
| Custom | Defined by each operator | Depends on configured operators | Mixture of the MC operators below |

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

Use `system.ensemble.method: custom` when operators need different temperatures
or chemical potentials. In custom mode, put `temperature` and `chempots`
directly on the relevant operators. GDPy then skips the
cross-operator ensemble checks and does not inject thermodynamic settings. The
resulting mixture is user-defined and may not sample one thermodynamic ensemble.

```yaml
system:
  ensemble:
    method: custom
strategy:
  operators:
    - method: move
      particles: [Cu, Ni]
      temperature: 600.0
    - method: swap_type
      particles: [Cu, Ni]
      temperature: 1200.0
      chempots: [0.0, 0.2]
```

## Examples

The {doc}`EMT examples <mc/examples/index>` provide complete inputs and commands
for canonical, semi-grand-canonical, and grand-canonical MC with single-point energies.
The {doc}`hybrid MC examples <hmc/examples/index>` combine short EMT molecular
dynamics segments with displacement or identity-change MC blocks. Their guide
covers cycle configuration, the three runtime roles, cycle-level output,
and restart behaviour.

```{toctree}
:maxdepth: 2
:hidden:

mc/examples/index
```

## Proposal settings

`probability` is a relative operator-selection weight; the weights are
normalised automatically. Preset ensembles define temperature and chemical
potentials once under `system.ensemble`; custom ensembles define them on each
operator. Use consistent regions when combining moves and exchange; see
{ref}`region-definitions` for region definitions.

Canonical, semi-grand-canonical, and grand-canonical ensembles automatically
disable proposal distance checks. They reject `skip_distance_check: false`
because geometry filtering changes the proposal distribution and breaks
detailed balance. The energy evaluation decides whether each trial is
accepted.

The `custom` ensemble permits distance checks for structure search. There,
`covalent_ratio` sets lower and upper multipliers of covalent bond distances,
and `skip_distance_check` defaults to `false`. These filtered proposals do not
represent equilibrium sampling. `max_random_attempts: 1` avoids retrying until
a valid move is found. Failed proposals count as steps and retain the current
state.

MC assigns distinct atomic tags by default. If `system.ignore_atoms_tags: false`,
provide distinct tags for independent atoms; atoms sharing a tag are treated
as one particle. `swap_type` requires single-atom particles.

## Inspect the results

Terminal output groups setup, initialization, each MC step, and completion in
boxes. Worker evaluations appear inside the corresponding initialization or
step box. Each step reports its operator, acceptance decision, previous and
trial energies, current energy, and atom count. Waiting evaluations and invalid
proposals are labelled separately. Use `gdp --debug ...` for detailed proposal diagnostics.
Restart messages identify the checkpoint or pending step being resumed.

Each output directory contains:

- `mc.xyz`: the initial state followed by the current state after each completed
  MC step, including repeated states on rejection. Keep these repetitions when
  computing averages.
- `mc_attempts.xyz`: proposed structures sent for evaluation, including rejected
  trials; this is not the accepted-state trajectory.
- `opstat.txt`: operator, diagnostic, current atom count, acceptance flag, and
  previous/trial energies.
- `calculations/step.NNNN/`: retained runtime calculations.

`strategy.convergence.steps: 100` is the attempted-step budget. An optional
`strategy.convergence.earlystop` uses the existing early-stop criteria. The
examples retain all calculation steps with `strategy.dump_period: 1`;
increasing it prunes intermediate calculation directories, **not** frames in
`mc.xyz`. `strategy.ckpt_period` controls checkpoint frequency independently.

Inspect energy and composition from the repository root:

```shell
python examples/monte_carlo/inspect_run.py run-mc-canonical
python examples/monte_carlo/inspect_run.py run-mc-semi-grand-canonical
python examples/monte_carlo/inspect_run.py run-mc-grand-canonical
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

`strategy.ckpt_period` controls ordinary checkpoint frequency. Only the latest and
previous committed checkpoints are retained; a damaged latest snapshot falls
back to the previous one. A queued move also retains its pending state until
its resolution is committed, forcing a checkpoint even between ordinary
checkpoint steps. Hybrid MC commits this state after its complete cycle.
Older pickle checkpoints are not loaded; use a new output directory for those
runs.

## Application

1. {ref}`ref-acs-catal-2022-xu`
