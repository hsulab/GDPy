Provider plugins
================

GDPy integrations are stateless providers. A provider may expose potentials,
materializers, executors, trainers, dataset codecs, modifiers, collective
variables, schedulers, or exploration strategies through a capability map.

A minimal potential provider
----------------------------

The potential factory returns a backend-neutral object. A separate
materializer translates that object to the interface required by an executor::

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

Factories implement ``create(parameters, **context)``. Materializers implement
``materialize(potential, target, **context)``. Providers must not retain the
created calculator, executor, training run, or working-directory state.

Plugin discovery
----------------

External distributions publish one entry point in ``pyproject.toml``::

    [project.entry-points."gdpx.providers"]
    example = "example_gdpx:load_provider"

The callable returns a :class:`gdpx.providers.Provider` whose name matches the
entry-point name. Provider modules should import optional scientific packages
only from the factory or materializer that needs them.

Configuration
-------------

Provider configurations use schema version 2::

    schema_version: 2
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
    scheduler:
      provider: local
      parameters: {}

The executor declares the materialization interface it consumes. Resolution
fails before submission when the potential cannot produce that interface.

Legacy manager plugins
----------------------

The manager registry, ``potter`` configuration, and
``BasePotentialManager.create_driver`` remain available for one compatibility
release. New plugins should use providers; legacy configuration is read but
all new serialization uses schema version 2.

Provider boundary
-----------------

Use one provider package per external software or model family.  That package
owns its potential factories, materializers, executor adapters, trainer
factories, parsers, and artifact conventions.  A potential provider does not
own every program that can execute it: DeepMD publishes both
``ase.calculator`` and ``lammps.potential`` materializations, which ASE and
LAMMPS consume independently.
