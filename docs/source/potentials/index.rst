.. _Potential Examples:

Potentials
==========

A potential component identifies a model family and its model parameters. It
does not select the software that will execute the calculation. That choice
belongs to the executor component.

.. code-block:: yaml

    potential:
      provider: deepmd
      method: default
      parameters:
        type_list: [H, O]
        model: ./graph.pb

Materialization
---------------

During runtime resolution, the potential provider materializes the interface
required by the executor. The same DeepMD potential can therefore be used with
an ASE executor or a LAMMPS executor without changing its identity::

    DeepMD potential -> ase.calculator  -> ASE executor
                     -> lammps.potential -> LAMMPS executor

An incompatible pairing fails during resolution, before submission.

Native software
---------------

Providers such as VASP and CP2K can expose native execution as well as an ASE
calculator interface. Native input settings remain potential parameters while
the calculation method is selected by the executor::

    schema_version: 2
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

Modifiers
---------

Biases and enhanced-sampling forces are explicit runtime ``modifiers``. Each
modifier is a provider component with a method and parameters. Runtime
resolution applies compatible modifiers to the materialized potential; the
removed mixer potential is not part of schema version 2.

Supported families
------------------

Built-in providers cover classical models, machine-learning potentials,
electronic-structure software, and lightweight ASE calculators. Most providers
require their corresponding optional scientific package. External provider
distributions can add capabilities through the ``gdpx.providers`` entry-point
group; see :doc:`../extensions/index`.

Training
--------

Trainer capabilities are owned by the same provider as the potential they
produce. See :ref:`Trainers`.
