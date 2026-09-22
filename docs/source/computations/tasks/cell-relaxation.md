# cell relaxation

Use `cmin` to optimize both atomic positions and a periodic simulation cell.

Generate the input files as described in {doc}`index`, then use
`examples/compute/tasks/cell-relaxation.yaml`:

```yaml
potential:
  provider: emt
  parameters: {}
executor:
  provider: ase
  method: cmin
  parameters:
    fmax: 0.02
    steps: 100
    controller:
      name: bfgs
      params:
        isotropic: true
        pressure: 0.0
```

From the repository root:

```shell
gdp -d cmin-demo -r examples/compute/tasks/cell-relaxation.yaml compute examples/compute/tasks/bulk.xyz
```

This starts from an expanded FCC Cu cell. `isotropic: true` restricts cell
strain to isotropic expansion/contraction; set it to `false` to allow general
strain. `controller.params.pressure` is in bar and defaults to zero for this
cell minimizer. These options are nested under the controller, unlike the
flat `fmax` and `steps` settings.

The ASE driver uses BFGS with `UnitCellFilter`. The potential must implement
stress as well as energy and forces. Do not use a molecule-in-vacuum box as
the cell-relaxation example. Inspect the relaxed lattice and stress in
`cmin-demo/results/end_frames.xyz`.

To change machine resources, add a scheduler as described in
{doc}`../schedulers`.
