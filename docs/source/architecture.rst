GDPy architecture
=================

GDPy separates integration ownership from runtime responsibility.  Dependencies
flow in one direction::

    core/domain -> providers -> execution -> exploration -> workflow -> cli

The data, structures, modifiers, and analysis packages are domain libraries and
must not import workflow or CLI modules.  Providers own software-specific
knowledge as one vertical plugin: model specifications, materializers,
executable adapters, trainers, and artifacts.  For example, DeepMD owns both
its ASE-calculator and LAMMPS-potential materializations, while ASE and LAMMPS
own the corresponding execution mechanisms.  Exploration algorithms may
request calculations from the execution service, but execution must never
depend on exploration.

Providers are stateless capability catalogs.  Scientific configuration is held
in immutable specifications; mutable state belongs to executors, runtimes, and
execution services.  Cross-provider integration uses consumer-defined
materialization interfaces rather than imports of another provider's concrete
implementation.

Execution and exploration
-------------------------

Execution performs a concrete request: evaluate these structures, optimize
this geometry, run this trajectory, or search this supplied transition-state
problem.  Exploration owns the adaptive policy that decides which request to
make next from previous results.  Consequently exploration receives an
``ExecutionService``; execution never imports an exploration strategy.

A dimer search is a local transition-state executor because it runs one
replica.  NEB and string methods are path transition-state executors because
their concrete request contains a path of replicas.  They are exploration only
when another algorithm adaptively chooses or changes those requests.

Public boundary
---------------

The implementation packages are now ``providers``, ``execution``,
``exploration``, ``workflow``, ``structures``, ``analysis``, ``modifiers``, and
``data``.  Version 0.1 removes the former compatibility packages and global
registry catalog; integrations enter through the ``gdpx.providers`` plugin
group. Runtime input requires ``schema_version: 2`` and explicit
``potential``/``executor`` component sections. Calculators are created only
after the executor target is known, and unsupported target/modifier
combinations fail explicitly.
