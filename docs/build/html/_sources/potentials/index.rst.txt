.. _Potential Examples:

Potentials
==========

We can define a **potential** in a unified input file (`pot.yaml`) for later 
simulation and training. The MLIP calculations are performed by **ase** ``calculators`` using either 
**python** built-in codes (PyTorch, TensorFlow) or File-IO based external codes 
(e.g. **lammps**). 

Formulations
------------

**Suported MLIPs:**

We have already implemented interfaces to the potentials below:

+------------+-----------------------------------+-----------------+------------------------------+
| Name       | Representation                    | Backend         | Notes                        |
+============+===================================+=================+==============================+
| eann_      | (Recursive) Embedded Atom         | Python, LAMMPS  |                              |
+------------+-----------------------------------+-----------------+------------------------------+
| deepmd_    | Deep Descriptor                   | Python, LAMMPS  | Only potential model.        |
+------------+-----------------------------------+-----------------+------------------------------+
| lasp_      | Atom-Centred Symmetry Function    | LASP, LAMMPS    |                              |
+------------+-----------------------------------+-----------------+------------------------------+
| nequip_    | E(3)-Equivalent Message Passing   | Python, LAMMPS  | Allegro is supported as well.|
+------------+-----------------------------------+-----------------+------------------------------+

.. _eann: https://github.com/zhangylch/EANN
.. _deepmd: https://github.com/deepmodeling/deepmd-kit
.. _lasp: http://www.lasphub.com/#/lasp/laspHome
.. _nequip: https://github.com/mir-group/nequip

.. note:: 

    GDPy does not implement any MLIP but offers a unified interface. 
    Therefore, certain MLIP could not be utilised before 
    corresponding required packages are installed correctly.

**Other Potentials:**

Some potentials besides MLIPs are supported. Force fields or semi-empirical 
potentials are used for pre-sampling to build an initial dataset. *Ab-initio* 
methods are used to label structures with target properties (e.g. total energy, 
forces, and stresses).

+------------+-----------------------------------+-----------------+------------------------------+
| Name       | Description                       | Backend         | Notes                        |
+============+===================================+=================+==============================+
| reax       | Reactive Force Field              | LAMMPS          |                              |
+------------+-----------------------------------+-----------------+------------------------------+
| xtb        | Tight Binding                     | xtb             | Under development.           |
+------------+-----------------------------------+-----------------+------------------------------+
| vasp       | Density Functional Theory         | VASP            |                              |
+------------+-----------------------------------+-----------------+------------------------------+

Simulation
----------

The example below shows how to define a **potential** in a **yaml** file: 

.. code-block:: yaml

    # -- ase interface
    potential:
        name: deepmd # name of the potential
        params: # potential-specifc params
            backend: ase # ase or lammps
            model: ./graph.pb

    # -- lammps interface
    potential:
        name: deepmd
        params:
            backend: lammps
            command: lmp -in in.lammps 2>&1 > lmp.out
            model: ./graph.pb


Training
--------
