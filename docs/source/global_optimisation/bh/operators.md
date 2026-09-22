(bh-operators)=

# operators

Basin-hopping operators propose changes to the current accepted structure.
Configure them as a list under `recipe.operators`. Each chain selects one
operator per round, using normalized relative `probability` weights, then
submits a valid trial to the calculation runtime before deciding acceptance.

## Configuration

Start with one operator that displaces a copper atom:

```yaml
recipe:
  operators:
    - method: move
      particles: [Cu]
      max_disp: 0.8
      temperature: 500.0
```

See {doc}`operators/move` for proposal details and {ref}`bh-cu8-example` for
the complete recipe and runtime. Add more operator mappings to mix proposals.
Each `probability` is a relative selection weight (default 1.0). Weights must
be finite and nonnegative, with a positive total; they need not sum to one.
`probability` controls **operator selection**, not trial acceptance. The former
`prob` key is rejected.

## Available methods

All methods below are accepted by the shared BH/MC operator parser. Their
settings belong directly in each operator mapping beside `method`.

| Method | Proposal | Main settings |
| --- | --- | --- |
| {doc}`operators/move` | Displace one selected particle | `particles`, `max_disp` (default 2.0 Å) |
| {doc}`operators/rattle` | Displace a randomly selected subset of particles together | `particles`, `rattle_strength` (0.8 Å), `rattle_prop` (0.4) |
| {doc}`operators/bounce` | Bias an atomic displacement along an axis | `particles`, required `direction` (`+x`, `-x`, `+y`, `-y`, `+z`, `-z`), `bias_ratio`, `max_disp`, `repulsion_strength` |
| {doc}`operators/swap` | Exchange positions of two particle types | Two distinct `particles`, `swap_mode` (`atomic` or `cop_z`), `check_used_pairs` |
| {doc}`operators/swap_type` | Change atomic identity | At least two atomic `particles`, corresponding `chempots` |
| {doc}`operators/exchange` | Insert or remove a particle | One-entry `particles` and `chempots`, `region`, optional `use_ads` |
| {doc}`operators/biased_volume_exchange` | Exchange using the region's estimated empty volume | Exchange settings; the region must support `get_empty_volume` |
| {doc}`operators/cavity_exchange` | Propose atomic insertions using cavity trials | Exchange settings, required `num_trials`, optional `cavity_distance` |
| {doc}`operators/adsorbate_exchange` | Exchange adsorbates at configured sites | Exchange settings, required `anchors`; requires `use_ads: true` |
| {doc}`operators/react` | Propose a reaction between configured species | `reaction` with `particles`, `chempot_0`, and signed `coefficients`; `region`, `temperature`, optional `pressure` and `use_bias` |

Use `swap` to rearrange a fixed composition. Use `swap_type` to change species
counts, or an exchange method to change the number of particles. Exchange
methods take one species per operator; add separate entries to exchange
multiple species. Builder composition ranges only control initial structures;
they do not bound later exchange moves.

```{toctree}
:hidden:
:maxdepth: 1

operators/move
operators/rattle
operators/bounce
operators/swap
operators/swap_type
operators/exchange
operators/biased_volume_exchange
operators/cavity_exchange
operators/adsorbate_exchange
operators/react
```

(bh-operator-shared-settings)=

## Shared settings and particle selection

| Setting | Meaning | Default |
| --- | --- | --- |
| `temperature` | Acceptance temperature, in K | 300.0 |
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

Valid proposals are evaluated by the runtime; a minimization runtime relaxes
them before acceptance. Ordinary displacement and positional-swap moves use
the energy change and temperature. Identity changes, exchanges, and reactions
also use their chemical potentials and the corresponding acceptance factors.
The population's ranking objective is separate from the chain acceptance rule.
Keep exchange `chempots` and objective `chemical_potentials` consistent when
using formation-energy ranking, as in the variable-composition example.

Invalid proposals consume a round without an evaluation. Rejected evaluated
trials remain in the candidate database, while the chain retains its previous
accepted structure. Extinction and chain replacement are described on the
{doc}`../basin_hopping` page.

Moves temporarily borrow and edit a candidate. Local moves record only the
affected coordinates or properties for rollback; deletion records the removed
rows and their original indices. Execution owns the relaxed result. A rejection
restores the candidate without trying to reverse its relaxation.

BH owns population selection and search objectives. Shared acceptance formulas
and biased proposals do not imply that its population is an equilibrium sample.

(bh-operator-logs)=

## Setup output and move logs

Each BH invocation prints a compact setup box with operator indices and names,
normalized selection probabilities, particles, temperatures, and move-specific
settings.

Detailed move diagnostics are saved automatically in one file per hopping
generation: `tmp_folder/gen1/mcmoves.log`, `tmp_folder/gen2/mcmoves.log`, and so
on, alongside each generation's `rounds/` and `evaluations/` folders. Every line has
a timestamp, level, generation, round, chain, segment, parent candidate, and
operator index/name. Invocation headers contain full resolved operator settings;
fields that do not apply to a header use `-`.

Routine operator messages are written at normal verbosity. Detailed DEBUG
messages are included only when GDPy's DEBUG logging is enabled. Move details
stay out of the normal console, while setup and progress boxes remain visible.

Logs distinguish uncommitted proposal diagnostics from committed outcomes.
Outcome lines include acceptance/rejection, energies, extinction, and restart
candidate IDs where applicable. Invalid proposals are logged without an
evaluation. Resume appends an invocation marker and identifies reused pending
proposals; it does not regenerate them for logging. An interruption can leave
uncommitted or replayed diagnostics, so `events.jsonl` and checkpoints remain the
authoritative scientific history. These logs survive checkpoint cleanup.

Logging uses existing results and scalar metadata: it does not copy atoms,
evaluate calculators, or consume random numbers. Canonical MC logging is
unchanged.
