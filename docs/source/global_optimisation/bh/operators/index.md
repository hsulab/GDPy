(bh-operators)=

# Operators

Basin hopping uses structural proposals to search for low-energy minima.
Configure them under `strategy.operators`, with `temperature` and any chemical
potentials on each operator. With a minimization runtime, valid trials are
relaxed before chain acceptance. The population is not an equilibrium sample,
so these search recipes do not require detailed balance with respect to a
finite-temperature ensemble.

BH also supports the six {ref}`MC operators <sampling-operators>`: `move`,
`rotate`, `swap`, `swap_type`, `biased_volume_exchange`, and `cavity_exchange`.
Their implementation and parameter names are shared. The methods below are
documented here for structure search; availability in the shared parser does
not establish suitability for equilibrium MC.

## Search operators

| Method | Proposal | Main settings |
| --- | --- | --- |
| {doc}`rattle` | Displace a randomly selected subset of particles together | `particles`, `rattle_strength` (0.8 Å), `rattle_prop` (0.4) |
| {doc}`bond_break` | Break a terminal bond by moving one atom | `particles`, `bond`, `cutoffs`, `covalent_ratio` |
| {doc}`bond_form` | Attach one isolated atom to a stationary acceptor | `particles`, `bond`, optional `acceptor_group`, `cutoffs`, `covalent_ratio` |
| {doc}`bounce` | Bias an atomic displacement along an axis | `particles`, required `direction` (`+x`, `-x`, `+y`, `-y`, `+z`, `-z`), `bias_ratio`, `max_disp`, `repulsion_strength` |
| {doc}`adsorbate_exchange` | Exchange adsorbates at configured sites | Exchange settings, required `anchors`; requires `use_ads: true` |
| {doc}`react` | Propose a reaction between configured species | `reaction` with `particles` and signed `coefficients`; `region`, optional `pressure` and `use_bias` |

See {doc}`bond_change` for combining bond breaking and formation.
Builder composition ranges control initialization; they do not bound later
exchange moves.

```{toctree}
:hidden:
:maxdepth: 1

rattle
bond_change
bond_break
bond_form
bounce
adsorbate_exchange
react
```

## Proposal settings and acceptance

`probability` is a relative operator-selection weight, not an acceptance
probability. Geometry filtering and repeated attempts can help produce useful
search candidates: `covalent_ratio` controls distance checks,
`skip_distance_check` defaults to `false`, and `max_random_attempts` limits
retries. See {ref}`sampling-operator-shared-settings` for shared parameter
definitions and particle tags.

For example, collective displacements before relaxation use:

```yaml
strategy:
  operators:
    - method: rattle
      particles: [Cu]
      temperature: 600.0
      rattle_strength: 0.1
      rattle_prop: 0.4
      probability: 1.0
      max_random_attempts: 100
```

Invalid proposals consume a hop and retain the current minimum. Rejected trials
also retain the current minimum, while evaluated candidates remain available
for population ranking. See {doc}`../../basin_hopping` for chain acceptance,
ranking, restart, and {ref}`move logs <bh-operator-logs>`.
