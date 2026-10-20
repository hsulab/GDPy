(sampling-operators)=

# MC operators

This MC reference covers `move`, `rotate`, `swap`, `swap_type`,
`biased_volume_exchange`, and `cavity_exchange`. Use them with
{doc}`Monte Carlo <../mc>` or the MC blocks of {doc}`hybrid MC <../hmc>`.
Equilibrium sampling requires detailed balance for the complete proposal and
acceptance rule; inclusion here is not a guarantee for every setting.

For structure search, see {ref}`bh-operators`. BH can reuse these six operators
and provides additional search proposals. Both methods share implementations;
this documentation split describes their intended use, not a parser restriction.

## Configuration

MC, hybrid MC, and basin hopping configure proposals under
`strategy.operators`. Preset MC and hybrid MC ensembles define temperature and
chemical potentials under `system.ensemble`. Custom MC, custom hybrid MC, and
basin hopping keep those acceptance settings on each operator. For example,
this MC fragment displaces a copper atom:

```yaml
strategy:
  operators:
    - method: move
      particles: [Cu]
      max_disp: 0.8
```

See {doc}`move` for proposal details and the {doc}`canonical MC example
<../mc/examples/canonical>` for a complete recipe and runtime. Add more operator mappings to mix proposals.
Each `probability` is a relative selection weight (default 1.0). Weights must
be finite and nonnegative, with a positive total; they need not sum to one.
`probability` controls **operator selection**, not trial acceptance. The former
`prob` key is rejected.

## MC methods

Settings belong directly in each operator mapping beside `method`.

| Method | Proposal | Main settings |
| --- | --- | --- |
| {doc}`move` | Displace one selected particle | `particles`, `max_disp` (default 2.0 Å) |
| {doc}`rotate` | Rotate one tagged molecule about a fixed pivot | `particles`, `max_angle` (180°), `center` (`com`), optional `axis` |
| {doc}`swap` | Exchange positions of two particle types | Two distinct `particles`, `swap_mode` (`atomic` or `cop_z`), `check_used_pairs` |
| {doc}`swap_type` | Change atomic identity | At least two atomic `particles` |
| {doc}`biased_volume_exchange` | Exchange using the region's estimated empty volume | Exchange settings; the region must support `get_empty_volume` |
| {doc}`cavity_exchange` | Propose atomic insertions using cavity trials | Exchange settings, required `num_trials`, optional `cavity_distance` |

Use `swap` to rearrange a fixed composition. Use `swap_type` to change species
counts, or an exchange method to change the number of particles. Exchange
methods take one species per operator; add separate entries to exchange
multiple species. Builder composition ranges only control initial structures;
they do not bound later exchange moves.

```{toctree}
:hidden:
:maxdepth: 1

move
rotate
swap
swap_type
biased_volume_exchange
cavity_exchange
```

(sampling-operator-shared-settings)=

## Shared settings and particle selection

| Setting | Meaning | Default |
| --- | --- | --- |
| `temperature` | Acceptance temperature for custom MC/HMC and BH; preset MC/HMC uses `system.ensemble.temperature` | 300.0 |
| `region` | Region used to select particles and, for exchange, place insertions | Automatic region |
| `covalent_ratio` | Lower/upper distance-check factors relative to covalent radii | `[0.8, 2.0]` |
| `allow_isolated` | Permit isolated particles in proposal distance checks | `false` |
| `skip_distance_check` | Skip proposal distance checks where supported; forced `true` by preset MC ensembles | `false` (`custom`) |
| `max_random_attempts` | Maximum proposal attempts where supported | 1000 |

`particles` contains chemical symbols or supported molecular formulas. Atoms
sharing a tag form one particle; use distinct tags for independent atomic moves.
The automatic region requires a nonzero simulation cell. A proposal can be
invalid when no requested particle is present or geometric attempts fail.

## Acceptance and proposal lifecycle

Standard MC requires a single-point runtime and evaluates proposed states
without relaxation. Basin hopping normally minimizes them first. Ordinary
displacement and positional-swap moves use the energy change and temperature.
Identity changes and exchanges also use chemical potentials and
their corresponding acceptance factors.

Invalid proposals do not receive an energy evaluation. Their handling depends
on the exploration method:

- MC retains the current state and advances.
- Hybrid MC consumes the proposal's place in its MC block and retains the
  current state.
- Basin hopping consumes a hop in that chain. Its {doc}`method guide
  <../../global_optimisation/basin_hopping>` describes candidate storage,
  population ranking, and chain replacement.

Moves temporarily borrow and edit a candidate. Local moves record only the
affected coordinates or properties for rollback; deletion records the removed
rows and their original indices. Execution owns the relaxed result. A rejection
restores the candidate without trying to reverse its relaxation.

For equilibrium MC, use reversible proposals, disable ordinary geometry
filtering, and use `max_random_attempts: 1` so failed proposals retain the
current state. Keep selection weights fixed, and ensure the selection region
allows the reverse move. Asymmetric proposals require their reverse/forward
probability ratio in acceptance. `swap_type` includes its species-selection
correction; the two biased exchange methods have additional limitations detailed
on their pages. A preset ensemble does not validate those corrections. See the {doc}`MC guide <../mc>` for limitations and
the {doc}`hybrid MC guide <../hmc>` for how MD and MC are combined.

For output and diagnostic logs, see the {doc}`MC guide <../mc>` and
{ref}`BH move logs <bh-operator-logs>`.
