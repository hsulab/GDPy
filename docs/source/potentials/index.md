(potential-examples)=

# potentials

A potential component identifies a model family (`provider`) and model `parameters`.
The executor independently identifies the simulation engine (`provider`) and
task (`method`, such as `min` or `md`).

```yaml
potential:
  provider: deepmd
  method: default
  parameters:
    type_list: [H, O]
    model: ./graph.pb
```

## Materialization

During runtime resolution, the potential provider materializes the interface
required by the executor. The same DeepMD potential can therefore be used with
an ASE executor or a LAMMPS executor without changing its identity:

```
DeepMD potential -> ase.calculator  -> ASE executor
                 -> lammps.potential -> LAMMPS executor
```

An incompatible pairing fails during resolution, before submission.

For ReaxFF, the ASE executor selects xreac:

```yaml
potential:
  provider: reax
  parameters:
    model: bundled:ffield.reax.HO.2015
executor:
  provider: ase
  method: min
  parameters:
    stop:
      fmax: 0.05
      steps: 20
```

Selecting `executor.provider: lammps` instead uses LAMMPS `reax/c` and requires
a local force-field file. To use that implementation with ASE, set
`potential.backend: reax/c`.

## Native software

Providers such as VASP and CP2K can expose native execution as well as an ASE
calculator interface. Native input settings remain potential parameters while
the calculation method is selected by the executor:

```
potential:
  provider: vasp
  parameters:
    pp_path: /path/to/potentials
    incar: ./INCAR
    kpts: [1, 1, 1]
    command: mpirun -n 32 vasp_std
executor:
  provider: vasp
  method: spc
  parameters: {}
```

## Modifiers

Biases and enhanced-sampling forces are explicit runtime `modifiers`. Each
modifier is a provider component with parameters, an optional backend, and an
optional method for named operations. Providers with a default operation, such
as DFT-D3, do not require a method. Runtime
resolution applies compatible modifiers to the materialized potential; the
removed mixer potential is not part of schema version 3.

## Supported families

Built-in providers cover classical models, machine-learning potentials,
electronic-structure software, and lightweight ASE calculators. Most providers
require their corresponding optional scientific package. External provider
distributions can add capabilities through the `gdpx.providers` entry-point
group; see {doc}`../extensions/index`.

## Potential guides

See {doc}`providers` for potential configuration examples and requirements.
Runtime setup and executor compatibility are covered in {doc}`../computations/index`.

## Training

Trainer capabilities are owned by the same provider as the potential they
produce. See {ref}`trainers`.

## Backend defaults and overrides

`potential.provider` names the potential; optional `potential.backend` selects
its implementation. `potential.parameters` contains calculator settings.
Defaults depend on the executor's materialization target, never on installed
packages. Resolved runtime configurations record the selected backend.

| Potential | ASE default | ASE alternatives | Native backend |
| --- | --- | --- | --- |
| reax | xreac | reax/c | LAMMPS: reax/c |
| deepmd, nequip, beann | ase | lammps | LAMMPS: lammps |
| allegro | ase | lammps | LAMMPS: lammps |
| eam, mace | ase | — | LAMMPS: lammps |
| mattersim | ase | graph_pes | — |
| xtb | xtb | tblite | — |
| cp2k | cp2k | interactive | CP2K: cp2k |
| vasp | interactive | — | VASP: vasp |
| deepmd_jax | ase | jax | — |
| abacus | abacus | — | ABACUS: abacus |
| lasp | lasp | — | LASP: lasp |
| espresso | espresso | — | — |
| grid | grid | — | — |
| classic | — | — | LAMMPS: lammps |

Other built-in potentials use `ase`. Only declared target/backend combinations
are supported. Third-party materializers without backend declarations continue
to work without an override; explicit overrides require declarations.

Move `parameters.backend` to `potential.backend`. Replace CP2K/VASP's old
`parameters.interface` with `potential.backend`, using `interactive` for
`cp2k_shell` and `vasp_interactive`. Replace `vasp_interactive_disp` with
`interactive` plus a `dftd3` modifier; see {doc}`vasp`.

(potential-cpu-inference-comparison)=

## CPU inference comparison

The following measurements use the 184-atom Cu₄/alumina fixture
`examples/global_optimisation/assets/cu4_alumina111_supported.xyz`. Each value
is the median of ten uncached energy-and-force evaluations on slightly
perturbed geometries after three warmups, with one CPU thread. Model loading,
export, and compilation are excluded. All evaluations returned finite outputs.

| Model | CPU inference path | Time per evaluation |
| --- | --- | --- |
| DPA4C-mini OMat24, v20260819 | CPU-exported `.pt2`, scalar code generation | **29.2 ms** |
| MACE-MP-0b2 Small | ASE calculator with `compile_mode="default"`, float32 | 286.2 ms |
| MACE-OMAT-0 Small | ASE calculator with `compile_mode="default"`, float32 | 373.9 ms |
| MatterSim 1M | Standard ASE calculator, stress disabled | 421.0 ms |
| DPA4-mini OMat24, v20260805 | CPU-exported `.pt2` | 601.1 ms |
| MACE-OMAT-0 Medium | ASE calculator with `compile_mode="default"`, float32 | 1,117.4 ms |
| TACE-OMat24-7M | Compiled ASE calculator, scalar code generation, energy/forces only | 6,104.7 ms |
| TACE-OMat24-7M | Standard ASE calculator, energy/forces only | 10,312.2 ms |

The environment was macOS ARM, Python 3.12 (`gdp3`), PyTorch 2.11.0,
ASE 3.29.0, DeepMD-kit 3.2.0, MatterSim 1.2.5, and TACE 0.2.2. MACE 0.3.16
with e3nn 0.4.4 used a temporary package overlay to preserve TACE dependencies.
These results compare the stated deployment paths on one geometry, not model
accuracy, GPU performance, or complete exploration times.

See {doc}`deepmd` for DPA4/DPA4C exports, {doc}`mace` for foundation-model
results, {doc}`mattersim` for its calculator settings, and {doc}`tace` for
profiling and the CPU compilation workaround.
