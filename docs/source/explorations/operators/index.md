(sampling-operators)=

# operators

These proposal operators are shared by {doc}`Monte Carlo <../mc>`,
{doc}`hybrid Monte Carlo <../hmc>`, and
{doc}`basin hopping <../../global_optimisation/basin_hopping>`. They select
particles and propose structural changes; the exploration method controls
execution, invalid-proposal handling, and result storage.

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

See {doc}`move` for proposal details and {ref}`bh-cu8-example` for
the complete recipe and runtime. Add more operator mappings to mix proposals.
Each `probability` is a relative selection weight (default 1.0). Weights must
be finite and nonnegative, with a positive total; they need not sum to one.
`probability` controls **operator selection**, not trial acceptance. The former
`prob` key is rejected.

## Available methods

All methods below are accepted by the shared MC/HMC/BH operator parser. Their
settings belong directly in each operator mapping beside `method`.

| Method | Proposal | Main settings |
| --- | --- | --- |
| {doc}`move` | Displace one selected particle | `particles`, `max_disp` (default 2.0 Å) |
| {doc}`rattle` | Displace a randomly selected subset of particles together | `particles`, `rattle_strength` (0.8 Å), `rattle_prop` (0.4) |
| {doc}`bounce` | Bias an atomic displacement along an axis | `particles`, required `direction` (`+x`, `-x`, `+y`, `-y`, `+z`, `-z`), `bias_ratio`, `max_disp`, `repulsion_strength` |
| {doc}`swap` | Exchange positions of two particle types | Two distinct `particles`, `swap_mode` (`atomic` or `cop_z`), `check_used_pairs` |
| {doc}`swap_type` | Change atomic identity | At least two atomic `particles` |
| {doc}`exchange` | Insert or remove a particle | One-entry `particles`, `region`, optional `use_ads` |
| {doc}`biased_volume_exchange` | Exchange using the region's estimated empty volume | Exchange settings; the region must support `get_empty_volume` |
| {doc}`cavity_exchange` | Propose atomic insertions using cavity trials | Exchange settings, required `num_trials`, optional `cavity_distance` |
| {doc}`adsorbate_exchange` | Exchange adsorbates at configured sites | Exchange settings, required `anchors`; requires `use_ads: true` |
| {doc}`react` | Propose a reaction between configured species | `reaction` with `particles` and signed `coefficients`; `region`, optional `pressure` and `use_bias` |

Use `swap` to rearrange a fixed composition. Use `swap_type` to change species
counts, or an exchange method to change the number of particles. Exchange
methods take one species per operator; add separate entries to exchange
multiple species. Builder composition ranges only control initial structures;
they do not bound later exchange moves.

```{toctree}
:hidden:
:maxdepth: 1

move
rattle
bounce
swap
swap_type
exchange
biased_volume_exchange
cavity_exchange
adsorbate_exchange
react
```

(sampling-operator-shared-settings)=

## Shared settings and particle selection

| Setting | Meaning | Default |
| --- | --- | --- |
| `temperature` | Acceptance temperature for custom MC/HMC and BH; preset MC/HMC uses `system.ensemble.temperature` | 300.0 |
| `region` | Region used to select particles and, for exchange, place insertions | Automatic region |
| `covalent_ratio` | Lower/upper distance-check factors relative to covalent radii | `[0.8, 2.0]` |
| `allow_isolated` | Permit isolated particles in proposal distance checks | `false` |
| `skip_distance_check` | Skip proposal distance checks where supported | `false` |
| `max_random_attempts` | Maximum proposal attempts where supported | 1000 |

`particles` contains chemical symbols or supported molecular formulas. Atoms
sharing a tag form one particle; use distinct tags for independent atomic moves.
The `rattle` operator translates each selected tagged group without rotation.
The automatic region requires a nonzero simulation cell. A proposal can be
invalid when no requested particle is present or geometric attempts fail.

## Acceptance and proposal lifecycle

Standard MC requires a single-point runtime and evaluates proposed states
without relaxation. Basin hopping normally minimizes them first. Ordinary
displacement and positional-swap moves use the energy change and temperature.
Identity changes, exchanges, and reactions also use chemical potentials and
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

Biased proposals and repeated geometry attempts require separate
detailed-balance analysis. See the {doc}`MC guide <../mc>` for limitations and
the {doc}`hybrid MC guide <../hmc>` for how MD and MC are combined.

For output and diagnostic logs, see the {doc}`MC guide <../mc>` and
{ref}`BH move logs <bh-operator-logs>`.
