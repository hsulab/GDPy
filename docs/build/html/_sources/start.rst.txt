Getting Started
===============

Here, we would introduce several basic components of GDPy, namely **potential**, 
**driver**, **worker**. **Worker** is the basic unit that performs any 
computation, which contains a ``AbstractPotentialManager`` and a ``Scheduler``.

Units
-----

We use the following units through all input files:

Time ``fs``, Length ``AA``, Energy ``eV``, Force ``eV/AA``.

Potential
---------

We have supported several MLIP formulations based on an ``AbstractPotentialManager`` 
class to access **driver**, **expedition**, and **training** through workflows. 
The MLIP calculations are performed by **ase** ``calculators`` using either 
**python** built-in codes (PyTorch, TensorFlow) or File-IO based external codes 
(e.g. **lammps**). 

The example below shows how to define a **potential** in a **yaml** file: 

.. code-block:: yaml

    # -- ase interface
    potential:
        name: nequip # name of the potential
        params: # potential-specifc params
            backend: ase # ase or lammps
            file: ./deployed_model.pth

    # -- lammps interface
    potential:
        name: nequip
        params:
            backend: lammps
            command: lmp_cat -in in.lammps 2>&1 > lmp.out
            pair_style: nequip
            pair_coeff: "* * ./deployed_model.pth"

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

Driver
------

After potential is defined, we need to further specify what simulation would be 
perfomed in the **driver** section. A driver (``AbstractDriver``) is the basic 
unit with an attacthed **ase** ``calculators`` for basic dynamics tasks, namely, 
minimisation, molecular dynamics and transition-state search. Through a driver, 
we can reuse the input file to perform the same simulation with several different 
backends.

The example below shows how to define a **driver** in a **yaml** file: 

.. code-block:: yaml

    driver:
        backend: external # this means using the same backend as the calc
        task: md # molecular dynamics (md) or minimisation (min)
        init:
            md_style: nvt # thermostat NVT
            temp: 600 # temperature, Kelvin
            timestep: 1.0 # fs
        run:
            steps: 100

See :ref:`Driver Examples` section for more details. 

Scheduler
---------

With **potential** and **driver** defined, we can run simulations on local 
machines (directly in the command line). However, simulations, under most 
circumstances, would be really heavy even by MLIPs (image a 10 ns molecular 
dynamics). The simulations would ideally be dispatched to high performace clusters
(HPCs).

The example below shows how to define a **scheduler** in a **yaml** file:

.. code-block:: yaml

    scheduler:
        # -- currently, we only have slurm :(
        backend: slurm
        # -- scheduler script parameters
        partition: k2-hipri
        ntasks: 1
        time: "0:10:00"
        # -- environment settings
        environs: "source ~/envs/source_eann.sh\nconda activate catorch2\n"
        # -- users' commands
        # user_commands: # automatically set by tasks

Worker
------

**Worker** that combines the above components is what we use throughout various 
workflows to deal with computations. 

The example below shows how to define a **worker** in a **yaml** file: 

.. code-block:: yaml

    potential:
        name: nequip # name of the potential
        backend: ase # ase or lammps
        params: # potential-specifc params
            file: ./deployed_model.pth
    driver:
        backend: external
        task: md # molecular dynamics (md) or minimisation (min)
        init:
            md_style: nvt # thermostat NVT
            temp: 600 # temperature, Kelvin
            timestep: 1.0 # fs
        run:
            steps: 100
    scheduler:
        backend: slurm
        partition: k2-hipri
        ntasks: 1
        time: "0:10:00"
        environs: "source ~/envs/source_eann.sh\nconda activate catorch2\n"

to run a **nvt** simulation with given structures by **nequip** on a **slurm** 
machine

.. code-block:: shell

    # -- submit jobs...
    #    one structure for one job
    $ gdp -p ./worker.yaml worker ./frames.xyz
    nframes:  2
    @@@DriverBasedWorker+run
    Use attached confids...
    Use attached confids...
    cand100 JOBID: 10206151
    cand96 JOBID: 10206152
    @@@DriverBasedWorker+inspect
    cand100 is running...
    cand96 is running...
    @@@DriverBasedWorker+inspect
    cand100 is running...
    cand96 is running...
    @@@DriverBasedWorker+retrieve

    # -- wait a few minutes...
    #    if jobs are not finished, run the command would retrieve nothing
    $ gdp -p ./worker.yaml worker ./frames.xyz
    nframes:  2
    @@@DriverBasedWorker+run
    Use attached confids...
    Use attached confids...
    @@@DriverBasedWorker+inspect
    cand100 is running...
    cand96 is running...
    @@@DriverBasedWorker+inspect
    cand100 is running...
    cand96 is running...
    @@@DriverBasedWorker+retrieve

    # -- retrieve results...
    $ gdp -p ./worker.yaml worker ./frames.xyz
    nframes:  2
    @@@DriverBasedWorker+run
    Use attached confids...
    Use attached confids...
    @@@DriverBasedWorker+inspect
    cand100 is finished...
    cand96 is finished...
    @@@DriverBasedWorker+inspect
    @@@DriverBasedWorker+retrieve
    *** read-results time:   0.0280 ***
    new_frames: 2 energy of the first: -92.219757
    nframes: 2
    statistics of total energies: min    -108.5682 max     -92.2198 avg    -100.3940

.. note:: 

    If **scheduler** is not set in the **yaml** file, the default 
    ``LocalScheduler`` would be used. In other words, the simulations would be 
    directly run in the command line.