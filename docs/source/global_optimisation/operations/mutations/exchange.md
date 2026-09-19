(ga-exchange-mutation)=

# Exchange

The `exchange` mutation inserts or removes atoms or molecular fragments. Use it
with a variable population and a composition-dependent target when the number
of particles may change during a search.

## Configuration

This example allows one to four independently tagged Cu atoms and one to four
independently tagged O atoms:

```yaml
population:
  name: variable
  # builders, initial population, and generation settings

operators:
  mutation:
    method: exchange
    species: [Cu, O]
    num_min_max:
      - [1.0, 4.0]
      - [1.0, 4.0]
    region:
      method: lattice
      origin: [0.0, 0.0, 4.6]
      cell:
        - [5.1, 0.0, 0.0]
        - [0.0, 4.4, 0.0]
        - [0.0, 0.0, 4.5]

property:
  target: cohesive_energy
  chempot:
    Cu: -3.50
    O: -4.95
```

The entries in `num_min_max` correspond to `species` in the same order. Each
pair gives inclusive lower and upper bounds for the number of particles of
that species. At a lower bound, exchange inserts that species; at an upper
bound, it removes that species; between the bounds, it chooses either action.

The initial and completion builders must also generate compositions within the
intended search space. The {ref}`ga-variable-surface-oxide-example` demonstrates
this with range-valued builder compositions for CuₓOᵧ/Cu(111).

:::{note}
Chemical potentials determine how different compositions are ranked. Calculate
them consistently with the selected potential and physical reservoirs before
interpreting a variable-composition search.
:::

## Insertion modes

Without `anchors`, a new particle is placed randomly inside `region` and
accepted only when its distances satisfy `covalent_ratio`. The default ratio is
`[0.8, 2.0]`. `max_attempts`, which defaults to `1000`, limits the number of
placement trials.

Set `anchors` to place species at graph-derived adsorption sites instead. A
single anchor configuration applies to every species, or a list can provide
one configuration per entry in `species`.

## Fragments and managed settings

Each entry in `species` may identify an atom such as `O` or a supported
molecular formula. Exchange counts and removes particles using their positive
ASE tags, so a tagged molecule is inserted or removed as one fragment. Tag 0
remains reserved for the substrate.

GDPy obtains the bond-distance dictionary and random-number stream from the
population's reference builder. These managed settings should not be repeated
in the mutation configuration.
