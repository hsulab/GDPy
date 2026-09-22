# runtime and executors

Use `spc` for single-point evaluation, `min` for minimization, and `md`
for molecular dynamics. Executor parameters are flat settings such as
`steps`, `fmax`, `constraint`, `ensemble`, and `dump_period`.

Transition-state executors have two shapes. `dimer` is local and consumes
one replica; `neb` is a path executor and consumes endpoints or an ordered
image sequence.

## Pairing potentials with executors

The {doc}`potential guides <../potentials/providers>` describe only the
`potential` component. In a runtime file, add `executor` alongside it, as in
the relaxation example below. Session and workflow configurations may
place these components inside a `runtime` mapping.

The executor selects the materialization interface. A registered pairing does
not guarantee that every calculation method is supported by the calculator:
for example, cell relaxation needs stress, which the `nnp` calculator does not
provide. Potential guides record model-specific limitations.

| Potential provider | Executor provider |
| --- | --- |
| `emt`, `mattersim`, `tace`, `reann`, `fairchem`, `gp`, `nnp`, `xtb` | `ase` |
| `deepmd`, `eam`, `mace` | `ase` or `lammps` |
| `nequip` | `ase` or `lammps` declared; see its guide for the current implementation limitation |
| `reax` | `lammps` |
| `vasp` | `vasp` or `ase` |
| `cp2k` | `cp2k` or `ase` |
| `abacus` | `abacus` or `ase` |
| `espresso` | `ase` |
| `lasp` | `lasp` or `ase` |
| `dftd3`, `dftd4`, `bias`, `plumed` | `ase`; these supply a dispersion or bias contribution only |

| Executor provider | Registered methods |
| --- | --- |
| `ase` | `spc`, `min`, `cmin`, `md`, `neb`; `dimer` is registered but its controller is not implemented |
| `lammps` | `spc`, `min`, `md` |
| `vasp` | `spc`, `min`, `cmin`, `md`, `freq`, `neb` |
| `cp2k` | `spc`, `min`, `md`, `freq`, `ts`, `dimer`, `neb` |
| `abacus` | `scf`, `min`, `md` |
| `lasp` | `spc`, `min`, `cmin`, `md` |

For a fixed-cell relaxation with ASE, for example:

```yaml
potential:
  provider: emt
  parameters: {}
executor:
  provider: ase
  method: min
  parameters:
    fmax: 0.05
    steps: 300
```

## Native and LAMMPS execution

With `executor.provider: vasp`, GDPy uses the file-based VASP interface.
With `executor.provider: ase`, the default is `vasp_interactive`; the optional
potential `interface` parameter can select `vasp_interactive_disp` to include
DFT-D3. Interactive execution sets defaults for externally driven ionic steps.

Native CP2K execution always uses the `cp2k` interface. ASE-driven CP2K can
instead use `interface: cp2k_shell` with a shell-capable command. ABACUS input
templates must use `calculation scf` even though its native executor declares
additional methods. Quantum ESPRESSO has no native executor in GDPy.

LAMMPS requires a binary with the selected model’s pair style. DeepMD uses
`metal` units and writes model-deviation output for a committee. MACE uses
`mace no_domain_decomposition`, `metal` units, `atomic` atom style, and
`newton on`. The declared NequIP interface requests `newton off` for NequIP
and `newton on` for Allegro. ReaxFF uses `real` units and `charge` atom style;
its legacy adapter does not configure a charge-equilibration fix automatically.

## Biases and modifiers

To add a built-in restraint to a physical potential, put it in `modifiers`
alongside `potential` and `executor`:

```yaml
potential:
  provider: emt
  parameters: {}
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
      group: [0, 1]
      center: 2.5
      kspring: 0.1
```

This example restrains two zero-based atom indices. Selecting the `bias`
provider as the potential instead would evaluate the bias alone. Likewise,
`dftd3` and `dftd4` supply only dispersion, not a full electronic energy.

The `plumed` provider exposes a potential capability, not a general modifier
factory. It cannot simply be placed in `modifiers`; combining it with a host
calculator requires an integration that explicitly couples their energies
and forces. Schema version 3 does not provide the removed `mixer` potential.

