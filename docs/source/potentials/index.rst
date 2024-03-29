.. _Potential Examples:

Potentials
==========

We can define a **potential** in a unified input file (`worker.yaml`) for later 
simulation and training. The MLIP calculations are performed by **ase** ``calculators`` 
using either **python** built-in codes (PyTorch, TensorFlow) or File-IO based 
external codes (e.g. **lammps**).

Formulations
------------

We have already implemented interfaces to the potentials below:

**Suported MLIPs:**

MLIPs are the major concern.

+------------+-----------------------------------+-----------------+------------------------------+
| Name       | Representation                    | Backend         | Notes                        |
+============+===================================+=================+==============================+
| eann_      | (Recursive) Embedded Atom         | Python, LAMMPS  |                              |
+------------+-----------------------------------+-----------------+------------------------------+
| lasp_      | Atom-Centred Symmetry Function    | LASP, LAMMPS    |                              |
+------------+-----------------------------------+-----------------+------------------------------+
| schnet_    | Graph Neueral Network             | Python          |                              |
+------------+-----------------------------------+-----------------+------------------------------+
| deepmd_    | Deep Descriptor                   | Python, LAMMPS  | Only potential model.        |
+------------+-----------------------------------+-----------------+------------------------------+
| nequip_    | E(3)-Equivalent Message Passing   | Python, LAMMPS  | Allegro is supported as well.|
+------------+-----------------------------------+-----------------+------------------------------+
| mace_      | Equivalent Graph Neueral Network  | Python          |                              |
+------------+-----------------------------------+-----------------+------------------------------+

.. _eann: https://github.com/zhangylch/EANN
.. _lasp: http://www.lasphub.com/#/lasp/laspHome
.. _schnet: https://github.com/atomistic-machine-learning/schnetpack
.. _deepmd: https://github.com/deepmodeling/deepmd-kit
.. _nequip: https://github.com/mir-group/nequip
.. _mace: https://github.com/ACEsuit/mace

.. note:: 

    GDPy does not implement any MLIP but offers a unified interface. 
    Therefore, certain MLIP could not be utilised before 
    corresponding required packages are installed correctly.

**Other Potentials:**

Some potentials besides MLIPs are supported. Force fields or semi-empirical 
potentials are used for pre-sampling to build an initial dataset. *Ab-initio* 
methods are used to label structures with target properties (e.g. total energy, 
forces, and stresses).

+--------+---------------------------------------+---------+-------+
| Name   | Description                           | Backend | Notes |
+========+=======================================+=========+=======+
| reax   | Reactive Force Field                  | LAMMPS  |       |
+--------+---------------------------------------+---------+-------+
| xtb    | Tight Binding                         | xtb     |       |
+--------+---------------------------------------+---------+-------+
| vasp   | Density Functional Theory             | VASP    |       |
+--------+---------------------------------------+---------+-------+
| cp2k   | Density Functional Theory             | cp2k    |       |
+--------+---------------------------------------+---------+-------+
| plumed | Collective-Variable Enhanced Sampling | plumed  |       |
+--------+---------------------------------------+---------+-------+

Simulation
----------

We should define the potential in `worker.yaml` before running any simulation.

Basic Definition
________________

For most potentials, **type_list** and **model** are two required parameters that 
are used by different backends. If the **lammps** backend is used, **command** 
must be set to specify how to run *lammps*. The example below shows how to define 
a **potential** (eann, deepmd, nequip, reax) in a **yaml** file (`worker.yaml`): 

.. code-block:: yaml

    # -- ase interface
    potential:
      name: deepmd # name of the potential
      params: # potential-specifc params
        backend: ase # ase or lammps
        type_list: ["H", "O"]
        model: ./graph.pb

    # -- lammps interface
    potential:
      name: deepmd
      params:
        backend: lammps
        command: lmp -in in.lammps 2>&1 > lmp.out
        type_list: ["H", "O"]
        model: ./graph.pb

.. note:: 

    Allegro can be accessed through the **nequip** potential but with an extra 
    parameter **flavour: allegro** in the **params** section.

For **vasp**, the input can be much different as:

.. code-block:: yaml

    potential:
      name: vasp
      params:
        # NOTE: below paths should be absolute/resolved
        pp_path: _YOUR-PSEUDOPOTENTIAL-PATH_
        vdw_path: _YOUR-VDWKERNEL-PATH_
        incar: _YOUR-INCAR-PATH_
        # - system depandent
        kpts: [1, 1, 1] # kpoints, mesh [1,1,1] or spacing 30 AA^-1
        # - run vasp
        commad: mpirun -n 32 vasp_std 2>&1 > vasp.out

After setting a **driver** in the input file (`worker.yaml`), we can run simulations 
with the defined potential. See :ref:`driver examples` section for more details.

Mixing Potentials
_____________________

Sometimes the simulation requires several potentials, for example, enhanced sampling. 
A `mixer` potentical can be defined to realise this. Currently, it only supports the 
`ase` backend. The parameter accepts a list of potential definitions.

The example below uses a `deepmd` model in tandem with `plumed` that adds external 
forces defined in an input file `./plumed.inp`.

.. code-block:: yaml

    potential:
      name: mixer
      params:
        backend: ase
        potters:
          - name: deepmd
            params:
              backend: ase
              type_list: ["H", "O"]
              model:
                - ./graph.pb
          - name: plumed
            params:
              backend: ase
              inp: ./plumed.inp

Training
--------

See :ref:`Trainer` for more details.
