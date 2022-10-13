Computations
============

This section gives more details how to run basic simulations with different potentials 
using a unified input file, which is generally made up of three components. We 
need to define what potential to use in the **potential** section, what simulation to run 
in the **driver** section, and finally what **scheduler** to delegate if necessary. 

An example input file (`pot.yaml`) is organised as follows: 

.. code-block:: yaml

    potential:
        ... # define the backend, the model path and the specific parameters
    driver:
        ... # define the init and the run parameters of a simulation
    scheduler:
        ... # define a scheduler 

The related commands are 

.. code-block:: shell

    # gdp -h for more info
    $ gdp -h

    # - results would be written to current directory
    # --- run simulations on local nodes
    $ gdp -p ./pot.yaml driver ./structures.xyz

    # --- run simulations on local nodes or submitted to job queues
    $ gdp -p ./pot.yaml worker ./structures.xyz

    # - if -d option is used, results would be written to it
    $ gdp -d xxx -p ./pot.yaml worker ./structures.xyz


Potential
---------

Potential is the engine to drive any simulation. See :ref:`Potential Examples` 
section for more details on how to define a potential. 


.. _Driver Examples:

Driver
------

The driver supports **minisation**, **molecular dynamics**, and **transition state search**. 
For the **driver** section, `backend`, `task`, `init`, and `run` should be specified 
for each simulation. If an `external` backend is used, the minimisation would use 
the same backend defined in the **potential** section if it is valid. 

An example input file (`pot.yaml`) is

.. code-block:: yaml

    driver:
        backend: external # options are external, ase, lammps, and lasp
        task: min         # options are min, md, and ts
        init:
            ...           # initialisation parameters
        run:
            ...           # running parameters

Constraints
___________

Constraints are of great help when simulating some systems, for instance, surfaces. 
There are two ways to fix atoms in structures. The constraints could be either stored
in the structure file (e.g. move_mask of xyz and FractionalXYZ of xsd) or specified 
in `run: constraints`. If the latter one is used, the file-attached constraints would 
be overridden. 

Constraints can be specified as:

.. code-block:: yaml

    # 1. similiar to lammps group definition by atom ids
    run:
        # fix atoms with indices 1,2,3,4, 6,7,8, starting from 1
        constraints: "1:4 6:8"

Minimisation
____________

To drive a minisation, the minimal parameetrs are `steps` and `fmax`. Specific 
minisation algorithm can be defined in `init: min_style: ...`. The default `min_style` 
is `BFGS` for the `ase` backend while `fire` for the `lammps` backend.

.. code-block:: yaml

    driver:
        backend: external
        task: min
        init:
            min_style: bfgs
        run:
            steps: 200 # number of steps
            fmax: 0.05 # unit eV/AA, convergence criteria for atomic forces

Molecular Dynamics
__________________

To driver a molecular dynamics, thermostat and related parameters need to set in 
`init: ...`. Three thermostats are supported both by `ase` and `lammps`, 
which are nve, nvt and npt.

.. code-block:: yaml

    driver:
        backend: external
        task: md
        init:
            # 1. NVE
            md_style: nve # options are nve, nvt, and npt
            timestep: 2.0 # fs, verlet integration timestep
            # 2. NVT 
            #md_style: nvt # options are nve, nvt, and npt
            #timestep: 2.0 # fs, verlet integration timestep
            #temp: 300     # Kelvin, temperature
            #Tdamp: 100    # fs, temperature control frequency
            # 3. NPT
            #md_style: nvt # options are nve, nvt, and npt
            #timestep: 2.0 # fs, verlet integration timestep
            #temp: 300     # Kelvin, temperature
            #Tdamp: 100    # fs, Heatbath frequency
            #pres: 1.0     # atm, equilibrium pressure
            #Pdamp: 100    # fs, pressure control frequency
        run:
            steps: 200 # number of steps

Transition-State Search
_______________________

We are working on the interface to methods of Sella_ using the `ase` backend 
and NEB using the `lammps` backend.

.. _Sella: https://github.com/zadorlab/sella


Worker
------

If the **scheduler** section is defined in the input file (`pot.yaml`), a worker 
would be created to delegate simulations to the queue. Instead of using server 
database, we implement a light-weight file-based database using TinyDB_ to manage jobs.

.. _TinyDB: https://tinydb.readthedocs.io

Currently, we only support the **slurm** scheduler. The definition is 

.. code-block:: yaml

    scheduler:
        backend: slurm
        ...
        # SLURM-PARAMETERS
        ntasks: ...
        time: ...
        ...
        environs: "conda activte py37" # working environment setting
        user_commands: "" # would be automatically set by tasks

An additional keyword **batchsize** can be set in the input file as 

.. code-block:: yaml

    batchsize: 3
    potential:
        ...
    driver:
        ...
    scheduler:
        ...

which would split the input structures into groups that run as separate jobs. 
For example, two jobs would be submitted if we set a **batchsize** of 3 and have 
5 input structures. The first job would have 3 structures and the second one would 
have 2 structures. The default **batchsize** is 1 that one structure would occupy 
one job.
