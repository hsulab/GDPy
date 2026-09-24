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
    fmax: 0.05
    steps: 20
```

Selecting `executor.provider: lammps` instead uses LAMMPS `reax/c` and requires
a local force-field file. There is one ReaxFF implementation per executor
target, so no explicit potential backend field is needed.

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
modifier is a provider component with a method and parameters. Runtime
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
