GDPy architecture
=================

GDPy separates integration ownership from runtime responsibility.  Dependencies
flow in one direction::

    core -> providers -> execution -> exploration -> workflow -> cli

The data, structures, modifiers, and analysis packages are domain libraries and
must not import workflow or CLI modules.  Providers own software-specific
knowledge (for example, how a DeepMD model is represented in LAMMPS), while
executors own the act of running a calculation.  Exploration algorithms may
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

Legacy potential managers and workers are compatibility adapters for one
release.  Their calculators are now created after an executor target is known,
not while parsing the potential specification.  ASE modifiers are composed at
materialization time.  Other engines must expose a target-specific modifier
materializer; GDPy rejects such combinations rather than silently dropping a
requested bias.
