(bh-operator-adsorbate-exchange)=

# `adsorbate_exchange`

Insert adsorbates at sites identified from a surface graph, or remove an
existing tagged adsorbate.

## Configuration

This is an operator fragment to place in an exploration recipe:

```yaml
recipe:
  operators:
    - method: adsorbate_exchange
      particles: [CO]
      chempots: [-0.5]
      temperature: 500.0
      anchors:
        group: "`symbol Cu`"
        cutoff: 3.0
        max_order: 2
        surf_index: 2
```

## Settings

| Setting | Meaning | Default |
| --- | --- | --- |
| `particles` | One supported adsorbate species | Required |
| `chempots` | One chemical potential in eV per adsorbate | Required |
| `anchors` | Surface-site settings, or a one-entry list of settings | Required |
| `use_ads` | Use an adsorbate representation with binding information; must remain enabled | `true` |
| `anchors.group` | Group expression selecting surface atoms | Supply explicitly |
| `anchors.cutoff` | Surface graph neighbor cutoff in Å | 3.0 |
| `anchors.max_order` | Site orders to include: 0 atop, 1 also bridge, 2 also hollow | 3 |
| `anchors.surf_index` | Surface-normal axis: 0 x, 1 y, 2 z | 2 |

See {ref}`bh-operator-shared-settings` for temperature, relative selection
weights, regions, particle tags, and distance checks.

## Proposal and acceptance

The example assumes a Cu surface with its normal along z, separately tagged
adsorbates, and a runtime that describes Cu/C/O adsorption. Its chemical
potential is illustrative; the simple Cu-cluster EMT example is not a complete
runtime for this system.

Insertion builds a graph from `anchors.group`, finds adsorption sites, and
places an adsorbate using its binding information. Distance checks and retries
apply unless skipped. A lack of suitable sites or failed placement produces an
invalid proposal. `max_order: 2` includes atop, bridge, and hollow sites.

When adsorbates are present, insertion/removal is selected as for
{doc}`exchange`. Removal deletes an eligible tagged adsorbate. `region` determines
eligible particles and the volume for acceptance; the anchor group separately
determines the surface graph used for insertion.

Acceptance uses the exchange energy, chemical-potential, particle-count,
temperature, and region-volume factors. Site-directed placement is a search bias;
it does not imply equilibrium surface coverage sampling.
