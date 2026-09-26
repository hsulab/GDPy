# single runtimes and executors

A single runtime defines exactly one potential/executor pairing. Use `spc` for
single-point evaluation, `min` for minimization, and `md` for molecular
dynamics. Executor parameters are settings such as `steps`, `fmax`,
`constraint`, `ensemble`, and `dump_period`.

The {doc}`potential guides <../potentials/providers>` describe the model-specific
part of the configuration. A registered pairing does not guarantee that every
method is supported by the calculator: cell relaxation, for example, requires
a potential that supplies stress.

## Provider compatibility

| Potential provider | Executor provider |
| --- | --- |
| `emt`, `mattersim`, `tace`, `reann`, `fairchem`, `gp`, `nnp`, `xtb` | `ase` |
| `deepmd`, `eam`, `mace` | `ase` or `lammps` |
| `nequip` | `ase` or `lammps` declared; see its potential guide for the current limitation |
| `reax` | `ase` or `lammps` |
| `vasp` | `vasp` or `ase` |
| `cp2k` | `cp2k` or `ase` |
| `abacus` | `abacus` or `ase` |
| `espresso` | `ase` |
| `lasp` | `lasp` or `ase` |
| `dftd3`, `dftd4`, `bias`, `plumed` | `ase`; these provide only a dispersion or bias contribution |

| Executor provider | Registered methods |
| --- | --- |
| `ase` | `spc`, `min`, `cmin`, `md`, `neb`; `dimer` is registered but its controller is not implemented |
| `lammps` | `spc`, `min`, `md` |
| `vasp` | `spc`, `min`, `cmin`, `md`, `freq`, `neb` |
| `cp2k` | `spc`, `min`, `md`, `freq`, `ts`, `dimer`, `neb` |
| `abacus` | `scf`, `min`, `md` |
| `lasp` | `spc`, `min`, `cmin`, `md` |

Transition-state executors have two shapes: `dimer` consumes one replica,
while `neb` consumes endpoints or an ordered image sequence. See
{doc}`tasks/transition-states` for configurations.

## Native and LAMMPS execution

With `executor.provider: vasp`, gdpx defaults to the native `vasp` backend.
With `executor.provider: ase`, VASP defaults to `potential.backend: interactive`.
External DFT-D3 is a `dftd3` modifier with optional `backend: ase`; see
{doc}`../potentials/vasp`.

Native CP2K execution uses backend `cp2k`. ASE-driven CP2K instead uses
`potential.backend: interactive` with a shell-capable command. ABACUS input
templates must use `calculation scf` even though its native executor declares
additional methods. Quantum ESPRESSO has no native executor in gdpx.

LAMMPS requires a binary containing the selected model's pair style. DeepMD
uses `metal` units and writes model-deviation output for a committee. MACE uses
`mace no_domain_decomposition`, `metal` units, `atomic` atom style, and
`newton on`. NequIP requests `newton off`; Allegro requests `newton on`.
ReaxFF uses `real` units and `charge` atom style, and adds `qeq/reax` charge
equilibration.

## Modifiers

Put a restraint or correction in `modifiers` alongside the physical potential:

```yaml
potential:
  provider: emt
executor:
  provider: ase
  method: min
  parameters:
    fmax: 0.05
    steps: 300
modifiers:
  - provider: builtin
    method: distance_harmonic
    parameters:
      group: "`index 0 1`"
      center: 2.5
      kspring: 0.1
```

Selecting `bias`, `dftd3`, or `dftd4` as the potential evaluates only that
contribution. The `plumed` provider is a potential capability rather than a
general modifier factory; combining it with a host calculator requires an
integration that explicitly couples their energies and forces. The removed
`mixer` potential is not part of the current schema.

See the complete {ref}`H2O/Ni(111) example
<compute-ni-water-restraint-example>` for a ReaxFF molecular-dynamics runtime
with an O-H distance restraint.
