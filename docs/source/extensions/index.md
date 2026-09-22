# provider plugins

GDPy integrations are stateless providers. A provider may expose potentials,
materializers, executors, trainers, dataset codecs, modifiers, collective
variables, schedulers, transports, or exploration strategies through a
capability map. A transport factory receives the resolved scheduler in its
`scheduler` context argument and returns the scheduler or a transport wrapper.

## A minimal potential provider

The potential factory returns a backend-neutral object. A separate
materializer translates that object to the interface required by an executor:

```
from gdpx.providers import CapabilityKind, Provider

provider = Provider(
    name="example",
    version="2",
    capabilities={
        CapabilityKind.POTENTIAL: {"default": potential_factory},
        CapabilityKind.MATERIALIZER: {
            "ase.calculator": ase_materializer,
            "lammps.potential": lammps_materializer,
        },
    },
)
```

Factories implement `create(parameters, **context)`. Materializers implement
`materialize(potential, target, **context)`. Providers must not retain the
created calculator, executor, training run, or working-directory state.

## Plugin discovery

External distributions publish one entry point in `pyproject.toml`:

```
[project.entry-points."gdpx.providers"]
example = "example_gdpx:load_provider"
```

The callable returns a {class}`gdpx.providers.Provider` whose name matches the
entry-point name. Provider modules should import optional scientific packages
only from the factory or materializer that needs them.

## Configuration

Provider configurations use schema version 3:

```
potential:
  provider: example
  method: default
  parameters:
    model: model.bin
executor:
  provider: ase
  method: min
  parameters:
    fmax: 0.05
    steps: 10
```

The omitted scheduler defaults to direct execution on the current machine.

The executor declares the materialization interface it consumes. Resolution
fails before submission when the potential cannot produce that interface.

## Breaking boundary in 0.1

Provider entry points and schema version 3 are the only supported extension
boundary. The former manager registries, `potter` configuration, and
`BasePotentialManager.create_driver` API have been removed. An omitted
`schema_version` selects the current schema. Explicit unsupported versions and
legacy component keys are rejected rather than implicitly migrated.

## Provider boundary

Use one provider package per external software or model family. That package
owns its potential factories, materializers, executor adapters, trainer
factories, parsers, and artifact conventions. A potential provider does not
own every program that can execute it: DeepMD publishes both
`ase.calculator` and `lammps.potential` materializations, which ASE and
LAMMPS consume independently.
