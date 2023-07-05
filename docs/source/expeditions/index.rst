Expeditions
===========

This section demonstrates several advanced methods to explore the configuration space, 
which are made up of basic computations. In general, an expedition is made up of three 
components, namely, **builder**, **worker**, and some specific parameters. 

.. |Expedition| image:: ../../images/expedition.svg
    :width: 800
    :align: middle

Also, an expedition progresses iteratively. The figure below demonstrates how we decouple 
the working ingrediants and manage the communication between the expedition and 
the job queue (e.g. SLURM) if minimisations were not directly performed in the command line. 

In every iteration, the expedition will build several new structures (either from the scratch or 
based on previously explored structures), and then evolves these structures into more 
physically reasonable ones by minimisation or molecular dynamics. This procedure produces 
a large number of trajectories. Applying some selections, we can extract local minima 
of interest and some structures from trajectories, which help the MLIP learns a comprehensive configuration space.

   |Expedition|

Related Commands
----------------

.. code-block:: shell

   # - explore configuration space defined by `config.yaml` 
   #   results will be written to the `results` folder
   #   a log file will be written to `results/gdp.out` as well
   $ gdp -d exp explore ./config.yaml

   # - or use the below command if there is `worker` section defined in `config.yaml`
   $ gdp -d exp -p worker.yaml explore ./config.yaml

The `config.yaml` defines the specific expedition. See the page of each documentation 
for more information.
    
The `worker.yaml` defines the potential and the minimisation used through the expedition.
(See :ref:`computations` for more details.)
For example, the below configuration indicates a minisation by `deepmd` using both `lammps` 
backends. Note set `ignore_convergence` to **true** will ignore the convergence check of the 
minimisation. Since the structures from the first few iterations are far away from minima 
i.e. they have very high potential energies, there is no need to minimise them to the full 
convergence. In most cases, 400 `steps` is more than enough. This setting help us reduce 
computation costs.

Besides, a `slurm` scheduler is set with a `batchsize` of 5. When the expedition comes 
across any minimisation, it will automatically submit jobs to the queue and each job 
will contain 5 structures as a group. When run the `gdp ... explore ...` command again, 
the expedition will try to retrieve the minimisation results if they are finished, and 
continues to run the rest procedure.

If no scheduler is set, all minimisation will run in the command line. Therefore, 
it is practical to write a job script with the `gdp ... explore ...` command and 
submit it to the queue if a large number of structures are to explore.


.. code-block:: yaml

   batchsize: 5
   driver:
     backend: lammps
     ignore_convergence: true
     task: min
     run:
       fmax: 0.05 # eV/Ang
       steps: 400
       constraint: lowest 120
   potential:
     name: deepmd
     params:
       backend: lammps
       command: lmp -in in.lammps 2>&1 > lmp.out
       type_list: [Al, Cu, O]
       model:
         - ./graph-0.pb
         - ./graph-1.pb
         - ./graph-2.pb
         - ./graph-3.pb
   scheduler:
     backend: slurm
     ntasks: 1
     cpus-per-task: 4
     time: "00:10:00"
     environs: "conda activate deepmd\n"


List of Expeditions
-------------------

.. toctree::
   :maxdepth: 2

   mc.rst
   ga.rst


