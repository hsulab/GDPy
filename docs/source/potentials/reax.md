(potential-reax)=

# ReaxFF: xreac and LAMMPS reax/c

Set `potential.provider: reax` to select the potential family. The executor
determines which implementation is used; no potential backend field is needed:

| Internal implementation | Description | Executor |
| --- | --- | --- |
| `xreac` | [xreac](https://github.com/hsulab/xreac), NumPy/Autograd | `ase` |
| `reax/c` | LAMMPS `reax/c` pair style | `lammps` |

One `ReaxManager` handles both implementations. The provider supplies xreac's
calculator for an ASE executor and the `reax/c` pair style for a LAMMPS executor.

## xreac / ASE

Install from the GDPy checkout:

```shell
python -m pip install -e '.[reax]'
```

The extra pins xreac to revision `6660484bb7c84a5f85bb8219b13086ef7223da45`
(v0.7.0). No LAMMPS executable or neural-network checkpoint is needed.

```yaml
schema_version: 3
potential:
  provider: reax
  parameters:
    model: bundled:ffield.reax.HO.2015
executor:
  provider: ase
  method: min
  parameters:
    fmax: 0.05
    steps: 20
```

`model` is required. Use `bundled:<filename>` for an xreac parameter file,
or a local path such as `./ffield.reax`; local paths are expanded and resolved
to absolute paths. Bundled names are preserved for portability. Available
bundles are `ffield.reax.HO.2015`, `ffield.reax.CHO.2008`, and
`ffield.reax.ZnOH.2010`. These do not contain Cu, Ni, or Al parameters.
Element coverage alone does not establish a force field's suitability for a study.

Optional parameters are forwarded to xreac: `neighbor_skin` (0.3 Å by default;
zero rebuilds the neighbor list every evaluation), `neighbor_backend` (`ase` by
default, or `replicated`), `max_expanded_atoms` (512), `full_derivative`
(false), and `total_charge` (only zero is supported). xreac performs QEq
at each geometry. The default forces follow its fixed-charge convention.
The ASE adapter returns energies in eV, forces in eV/Å, and charges in e.
Neutral molecules, clusters, and fixed-cell periodic systems are supported;
stress and variable-cell relaxation are not supported.

See `examples/global_optimisation/runtimes/xreac.yaml` and the water-cluster
benchmark in that directory for the GA example and timing comparison.

## LAMMPS

Provide a ReaxFF force-field file and a LAMMPS binary that accepts the adapter’s `reax/c` pair style.

```yaml
schema_version: 3
potential:
  provider: reax
  parameters:
    command: lmp
    model: ./ffield.reax
    type_list: [H, O]
executor:
  provider: lammps
  method: min
  parameters:
    steps: 20
```

The adapter resolves `model` to an absolute path and sets
`pair_style reax/c NULL`, `units real`, and `atom_style charge`.
`type_list` identifies the elements in the calculation.

The LAMMPS input writer adds the `qeq/reax` fix for `reax/c`. Check compatibility
with your LAMMPS build: binaries providing only differently named ReaxFF styles
are not compatible with this legacy adapter's hard-coded style.
