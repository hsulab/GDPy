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

Migration boundary
------------------

The implementation packages are now ``providers``, ``execution``,
``exploration``, ``workflow``, ``structures``, ``analysis``, ``modifiers``, and
``data``.  The former ``potential``, ``trainer``, ``computation``, ``reactor``,
``worker``, ``scheduler``, ``expedition``, and singular domain packages contain
forwarding imports only and remain for one minor release.

Schema-v1 ``potter``/``driver`` input remains readable through a compatibility
translator.  New serialization is schema version 2.  Calculators are created
only after the executor target is known.  Unsupported target/modifier
combinations fail explicitly.
