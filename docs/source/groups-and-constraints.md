(groups-and-constraints)=

# group and constraints

Groups select atoms within a structure. Components with a `group` parameter,
such as builders, sampling operators, comparators, and restraints, use the same
selection syntax. An executor's `constraint` uses that syntax to select atoms
whose positions should remain fixed during a simulation.

## Selecting a group

Enclose each selector in backticks and quote the complete expression in YAML:

```yaml
group: "`symbol Cu O`"
```

This selects every Cu and O atom in the input structure. When a component uses
the group evaluator's default `group: null`, all atoms are selected.

| Selector | Example | Selected atoms |
| --- | --- | --- |
| `index` | `` `index 0 2 5` `` | Atoms at the listed zero-based ASE indices; list individual indices, without ranges. |
| `id` | `` `id 1:4 7` `` | Atoms 1 through 4 and atom 7, using one-based inclusive ranges. |
| `symbol` | `` `symbol Cu O` `` | Atoms with any of the listed chemical symbols. |
| `tag` | `` `tag 1:3` `` | Atoms whose ASE tags are 1, 2, or 3; tag values are not atom indices. |
| `zbot` | `` `zbot 4` `` | The bottom four atoms along z. |
| `ztop` | `` `ztop 4` `` | The top four atoms along z. |
| `region` | `` `region sphere 0 0 0 5` `` | Atoms inside a sphere centered at the origin with radius 5 Å. |

Combine selectors with `and` (intersection), `or` (union), and parentheses:

```yaml
group: "(`symbol Cu` or `symbol O`) and `tag 2`"
```

The result contains Cu or O atoms that also have tag 2. The current evaluator
does not support `not` or set subtraction. Use `index` or `id` when a component
requires a particular set of atoms, and check that the selection has the atom
count that component expects. For example, a distance restraint requires two
atoms. Group expressions select a set; they do not prescribe its ordering.

Region selectors use a region name followed by its numeric arguments. For
example, `sphere` takes three origin coordinates and a radius. See
{doc}`builders/region` for region geometry and units.

## Selecting bottom or top atoms

`zbot N` and `ztop N` select a number of atoms, rather than a number of layers
or atoms below or above a height threshold. Selection is evaluated on the
structure supplied to the component.

For a nonperiodic z axis, these selectors sort Cartesian z coordinates. For
a periodic z axis, they sort fractional z coordinates with the default wrapping
interval [-0.1, 0.9). You can specify an alternative interval:

```yaml
group: "`zbot 4 0.0 1.0`"
```

The two extra values control fractional-coordinate wrapping; they do not
define a spatial filter. Choose the wrapping interval to match the slab's
placement in the cell, and verify which atoms are selected, especially if the
slab crosses a periodic boundary. For tilted cells, fractional z follows the
cell coordinates rather than Cartesian height.

## Fixing atoms in an executor

Put `constraint` under `executor.parameters.setup` in a structured runtime:

```yaml
potential:
  provider: emt
executor:
  provider: ase
  method: min
  parameters:
    setup:
      constraint: "`zbot 4`"
    stop:
      fmax: 0.05
      steps: 100
```

This fixes the bottom four atoms while the remaining atoms relax. The same
expression can be used during molecular dynamics. `constraint` accepts the
same selectors and combinations as `group`, for example:

```yaml
constraint: "`id 1:4`"
```

This fixes the first four atoms regardless of their positions. In contrast,
`zbot` selects atoms from their coordinates, so different input structures can
select different atom indices. Use explicit indices or IDs when the same atoms
must remain fixed across a sequence of structures.

In existing runtimes with flat executor parameters, place `constraint` directly
under `executor.parameters`. Do not mix flat parameters with the structured
`setup`, `output`, and `stop` sections.

The ASE executor replaces constraints already attached to the input `Atoms`
with the configured fixed-atom selection (`FixAtoms`). An omitted or null
`constraint` gives no frozen atoms. This setting freezes all three position
components of each selected atom; it does not define a harmonic restraint or
freeze the simulation cell. For a distance restraint that allows motion, see
{doc}`computations/ni111-water-restraint`; for cell degrees of freedom, see
{doc}`computations/tasks/cell-relaxation`.
