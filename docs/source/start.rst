Getting Started
===============
Here, we would introduce several basic components of GDPy, namely **potential**, 
**driver**, **worker**. This section demonstrates how to use **gdp** to computate a number of structures.

The related commands are 

.. code-block:: shell

    # gdp -h for more info
    $ gdp -h

    # --- run simulations on local nodes or submitted to job queues
    $ gdp -p ./worker.yaml compute ./structures.xyz

    # - if -d option is used, results would be written to the folder `./results`
    $ gdp -d ./results -p ./worker.yaml compute ./structures.xyz

An example input file (`worker.yaml`) is organised as follows: 

.. code-block:: yaml

    potential:
        ... # define the backend, the model path and the specific parameters
    driver:
        ... # define the init and the run parameters of a simulation
    scheduler:
        ... # define a scheduler 

Units
-----
We use the following units through all input files:

Time ``fs``, Length ``AA``, Energy ``eV``, Force ``eV/AA``.

Potential
---------
We have supported several MLIP formulations based on an ``AbstractPotentialManager`` 
class to access **driver**, **expedition**, and **training** through workflows. 

The example below shows how to define a **deepmd** potential using the **ase** backend 
in a **yaml** file: 

.. code-block:: yaml

    # -- ase interface
    potential:
        name: deepmd # name of the potential
        params: # potential-specifc params
            backend: ase # ase or lammps
            model: ./graph.pb

See :ref:`Potential Examples` section for more details. 


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
circumstances, would be really heavy even by MLIPs (imagine a 10 ns molecular 
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
        environs: "conda activate py37\n"

Worker
------
**Worker** that combines the above components is what we use throughout various 
workflows to deal with computations. 

The example below shows how to define a **worker** in a **yaml** file: 

.. code-block:: yaml

    potential:
        name: deepmd # name of the potential
        backend: ase # ase or lammps
        params: # potential-specifc params
            model: ./graph.pb
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
        environs: "conda activate py37\n"

to run a **nvt** simulation with given structures by **deepmd** on a **slurm** 
machine

.. code-block:: shell

    # -- submit jobs...
    #    one structure for one job
    $ gdp -p ./worker.yaml compute ./frames.xyz
    nframes:  2
    @@@DriverBasedWorker+run
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
