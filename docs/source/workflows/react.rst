Transition-State and Path Searches
==================================

Transition-state methods are executors inside a runtime:

* Dimer is a local, single-replica transition-state executor. It consumes one
  starting structure.
* NEB and related string methods are path transition-state executors. They
  consume an ordered set of images or endpoint structures.

For NEB, use executor method ``neb`` with the provider that runs the path, for
example::

    schema_version: 2
    potential:
      provider: emt
      parameters: {}
    executor:
      provider: ase
      method: neb
      parameters:
        nimages: 7
        interpolation:
          mic: true
        climb: true
        fmax: 0.05
        steps: 100

The potential remains backend-neutral. Its provider materializes the interface
required by the selected path executor.
