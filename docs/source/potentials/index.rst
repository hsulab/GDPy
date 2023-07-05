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
            type_list: ["C", "H", "O"]
            model: ./graph.pb

    # -- lammps interface
    potential:
        name: deepmd
        params:
            backend: lammps
            command: lmp -in in.lammps 2>&1 > lmp.out
            type_list: ["C", "H", "O"]
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

Training (DEPRECATED, SEE TRAINER SECTION)
------------------------------------------

See Trainer_Examples_ in the GDPy repository for prepared input files.

.. _Trainer_Examples: https://github.com/hsulab/GDPy/tree/main/examples/potential/trainer

The related command is:

.. code-block:: shell

    # train models in directory xxx
    $ gdp -d xxx -p ./worker.yaml train

To use the training interface, we need to define several parameters in the **trainer** 
section. The dataset would be automatically converted into required format. Some 
training parameters that are the same for different potentials can be set here 
and would be mapped to the corresponding one in the training configuration file:

- config:     configuration file for the training
- size:       number of models trained at the same time
- epochs:     number of training epcohs
- dataset:    a list of directories containing xyz files
- train:      train command
- freeze:     freeze command
- scheduler:  job setting

For **deepmd**,

.. code-block:: yaml

    potential:
        name: deepmd
        trainer:
            config: ./config.json
            size: 2
            epochs: 500
            dataset:
                - ../set/Cu4-Cu4/
                - ../set/Cu8-Cu8/
            train: "dp train config.json 2>&1 > train.out"
            freeze: "dp freeze -o graph.pb 2>&1 > freeze.out"
            scheduler:
                backend: slurm
                partition: k2-gpu
                time: "00:30:00"
                ntasks: 1
                cpus-per-task: 12
                mem-per-cpu: 4G
                gres: gpu:v100:1
                environs: "conda activate deepmd\n"

.. note:: 

    Deepmd has **numb_steps** instead of epochs. Therefore, **numb_steps** would be
    set as **epochs** times number of batchsizes depends on the dataset.

For **nequip** and **allegro**, 

.. code-block:: yaml

    potential:
        name: nequip
        trainer:
            config: ./config.yaml
            size: 2
            epochs: 500
            dataset:
                - ../set/Cu4-Cu4/
                - ../set/Cu8-Cu8/
            train: "nequip-train config.yaml 2>&1 > train.out"
            freeze: "nequip-deploy build --train-dir ./auto ./deployed_model.pth 2>&1 > freeze.out"
            scheduler:
                backend: slurm
                partition: k2-gpu
                time: "00:30:00"
                ntasks: 1
                cpus-per-task: 12
                mem-per-cpu: 4G
                gres: gpu:v100:1
                environs: "conda activate catorch2\n"

For **eann**,

.. code-block:: yaml

    potential:
        name: eann
        trainer:
            config: ./config.yaml
            size: 2
            epochs: 500
            dataset:
                - ../set/Cu4-Cu4/
                - ../set/Cu8-Cu8/
            train: "eann --config ./config.yaml train 2>&1 > train.out"
            freeze: "eann --config ./config.yaml freeze EANN.pth -o eann_latest_ 2>&1 > freeze.out"
            scheduler:
                backend: slurm
                partition: k2-gpu
                time: "00:30:00"
                ntasks: 1
                cpus-per-task: 12
                mem-per-cpu: 4G
                gres: gpu:v100:1
                environs: "conda activate catorch2\n"

