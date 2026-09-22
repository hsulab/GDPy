# single-point energy and forces

Use `spc` to evaluate each input structure without moving atoms or changing the cell.

Generate the input files as described in {doc}`index`, then use
`examples/compute/tasks/single-point.yaml`:

```yaml
potential:
  provider: emt
  parameters: {}
executor:
  provider: ase
  method: spc
  parameters: {}
```

From the repository root:

```shell
gdp -d spc-demo -r examples/compute/tasks/single-point.yaml compute examples/compute/tasks/dimers.xyz
```

Read `spc-demo/results/end_frames.xyz` for the three evaluated structures,
including energy and force data. Omit `steps` for `spc`; a positive step count
is rejected by the ASE single-point driver.

For native ABACUS, the equivalent electronic task is named `scf`, not `spc`.
Use the {doc}`ABACUS potential configuration <../../potentials/abacus>` and
replace the executor with:

```yaml
executor:
  provider: abacus
  method: scf
  parameters: {}
```

That variant requires ABACUS and its input data; it does not use the EMT demo.

To change machine resources, add a scheduler as described in
{doc}`../schedulers`.
