Getting Started
===============
This section introduces the schema-v2 **potential**, **executor**, and
**scheduler** components used to calculate structures with GDPy.

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

    schema_version: 2
    potential:
        provider: deepmd
        parameters:
            model: ./graph.pb
    executor:
        provider: ase
        method: md
        parameters:
            ensemble: nvt
            temp: 600
            timestep: 1.0
            steps: 100
    scheduler:
        provider: local
        parameters: {}

Units
-----
We use the following units through all input files:

Time ``fs``, Length ``AA``, Energy ``eV``, Force ``eV/AA``.

Potential
---------
Potential providers describe a model independently of the software that will
execute it. Materialization connects the two at runtime.

The example below shows how to define a **deepmd** potential using the **ase** backend 
in a **yaml** file: 

.. code-block:: yaml

    schema_version: 2
    potential:
        provider: deepmd
        method: default
        parameters:
            model: ./graph.pb

See :ref:`Potential Examples` section for more details. 


Driver
------
The **executor** specifies the concrete calculation. Its provider determines
the software, while ``method`` selects single point, minimization, molecular
dynamics, dimer, or a path method such as NEB.

The example below shows how to define a **driver** in a **yaml** file: 

.. code-block:: yaml

    executor:
        provider: ase
        method: md
        parameters:
            ensemble: nvt
            temp: 600 # temperature, Kelvin
            timestep: 1.0 # fs
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
