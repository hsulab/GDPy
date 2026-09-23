(monte-carlo)=

# Monte Carlo (MC)

Monte Carlo proposes structural changes and accepts or rejects them using
energies from a runtime. Choose the ensemble through `recipe.operators`;
there is no separate `ensemble` key. The {doc}`examples <mc/examples/index>` use ASE EMT and
single-point energies, so no model download or external simulation executable
is needed. Install GDPy as described in {doc}`../installation` first.

| Ensemble | Fixed quantities | What changes | Operators |
| --- | --- | --- | --- |
| Canonical (NVT) | Species counts, volume, temperature | Positions | `move`, `rattle`; `swap` for alloys |
| Semi-grand-canonical | Total atom count, volume, temperature, chemical-potential differences | Species identities and optionally positions | `swap_type`, optionally `move` |
| Grand-canonical (μVT) | Volume, temperature, reservoir chemical potential | Particle count and optionally positions | `exchange`, optionally `move` |

`swap` exchanges the positions of unlike particles
and preserves composition; `swap_type` changes one atom's element and therefore
changes composition. The latter currently supports individual atoms only.

These are short workflow demonstrations, not equilibrated production studies.
The current implementation is an exploration method and its historical
acceptance rules do not establish detailed balance for every proposal. In
particular, `swap_type` chooses a species first and then an atom of that species,
but its acceptance rule omits the resulting proposal-count ratio. `exchange`
forces insertion when no exchangeable particles remain, without correcting the
change in insertion/deletion proposal probabilities at that boundary. Do not
use these semi-grand/grand-canonical demos as validated equilibrium samplers.

## Examples

The {doc}`EMT examples <mc/examples/index>` provide complete inputs and commands
for canonical, semi-grand-canonical, and grand-canonical MC with single-point energies.
The {doc}`hybrid MC examples <hmc/examples/index>` combine short EMT molecular
dynamics segments with displacement or identity-change MC blocks. Their guide
covers procedure configuration, the three runtime roles, cycle-level output,
and restart behaviour.

```{toctree}
:maxdepth: 2
:hidden:

mc/examples/index
```

## Proposal settings

`probability` is a relative operator-selection weight; the weights are
normalised automatically. Keep temperatures consistent across operators in
one simulation. Use consistent regions when combining moves and exchange;
see {ref}`region-definitions` for region definitions.

These demos set `skip_distance_check: true` so the energy evaluation, rather
than repeated geometry filtering, decides whether a trial is acceptable.
`max_random_attempts: 1` and `should_retry: false` avoid retrying failed
proposals until a valid move is found. Failed proposals count as steps and
retain the current state. These choices remove geometry-retry bias but do
not fix the semi-grand/exchange proposal limitations above.

With distance checks enabled, `covalent_ratio` sets lower and upper
multipliers of covalent bond distances. Such filters and retries can change
the proposal distribution and should not be assumed to preserve equilibrium
sampling. They can be useful for structure search.

MC assigns distinct atomic tags by default. If `ignore_atoms_tags: false`,
provide distinct tags for independent atoms; atoms sharing a tag are treated
as one particle. `swap_type` requires single-atom particles.

### Collective rattle moves

For collective displacements, replace a `move` entry with:

```yaml
- method: rattle
  particles: [Cu]
  rattle_strength: 0.1
  rattle_prop: 0.4
  temperature: 1200.0
  probability: 1.0
  skip_distance_check: true
  max_random_attempts: 1
```

Each eligible particle is selected independently with `rattle_prop`. Each
Cartesian displacement component is uniform between `-rattle_strength` and
`+rattle_strength` Å, rather than Gaussian. Tagged molecules translate rigidly.
Empty or invalid proposals exhaust the attempt limit and are handled according
to `should_retry`. MC and BH share these moves and acceptance rules, but BH
does not inherit from MC.

## Inspect the results

Terminal output groups setup, initialization, each MC step, and completion in
boxes. Worker evaluations appear inside the corresponding initialization or
step box. Each step reports its operator, acceptance decision, previous and
trial energies, current energy, and atom count. Waiting evaluations and invalid
proposals are labelled separately; an invalid proposal also reports whether
the step will be retried. Use `gdp --debug ...` for detailed proposal diagnostics.
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

`convergence.steps: 100` is a step budget, not a test of statistical convergence.
The examples retain all calculation steps with `dump_period: 1`; increasing
`dump_period` prunes intermediate calculation directories, **not** frames in
`mc.xyz`. `ckpt_period` controls checkpoint frequency independently.

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

`ckpt_period` controls ordinary checkpoint frequency. Only the latest and
previous committed checkpoints are retained; a damaged latest snapshot falls
back to the previous one. A queued move also retains its pending state until
its resolution is committed, forcing a checkpoint even between ordinary
checkpoint steps. Hybrid MC commits this state after its complete procedure.
Older pickle checkpoints are not loaded; use a new output directory for those
runs.

## Application

1. {ref}`ref-acs-catal-2022-xu`
