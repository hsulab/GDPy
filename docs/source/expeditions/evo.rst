Explore with Global Optimisation
--------------------------------
In this section, we demonstrate how to define an expedition using global optimisation 
in `exp.yaml`. Here, we define an expedition called **exp** and all results would 
be save in the directory **exp** under where we run the `explore` command, and 
a log file `evo.out` in **exp** can be found as well.

The global optimisation method used here is the genetic-algorithm search.

systems
_______
Since we use global optimisation to explore structures, the initial structures are 
usually generated randomly. Thus, the systems must be defined with a generator.
For example, `Cu4` is a system that contains bulks with 4 Cu atoms and random lattice 
constants. The definition using `structure` keyword is normally not allowed since it 
contains invariant initial structures.

.. code-block:: yaml

    Cu4:
      prefix: Cu4
      generator:
        type: bulk
        composition: {"Cu": 4}
        covalent_ratio: 0.5
        volume: 40
        cell: []
        region:
          phi: [35, 145]
          chi: [35, 145]
          psi: [35, 145]
          a: [3, 50]
          b: [3, 50]
          c: [3, 50]
      composition: {"Cu": 4}
      kpts: 30

create
______
As most global optimisation methods are complex, we need to set **task** to the 
corresponding configuration file. In specific, it uses genetic-algorithm search. 
Moreover, these computations take much time and we set **scheduler** to dispatch 
these tasks to the queue.

collect
_______
The computation results usually consist a large number of minimisation trajectories. 
We would like to select low-energy minima and their trajectories that propbably cover 
most area of the potential energy surface. (see article [1]) 

**boltz** would select converged minima (fmax < 0.05 eV/AA) based on their Boltzmann 
propbabilities. Only these minima's trajectories would be considered. Here, 256 
minima are selected for each explored system.

**traj_period** (that is 10) controls the selection interval of the trajectory 
(e.g. frame 0,10,20 for a 20-step trajectory). 

[1] Bernstein, N.; Csányi, G.; Deringer, V. L. 
    De Novo Exploration and Self-Guided Learning of Potential-Energy Surfaces. 
    npj Comput. Mater. 2019, 5, 99.

select
______
The selection contains a convergence selector, which would split all structures into 
two groups: converged ones and others. The consecutive selections would be performed 
on these two groups, respectively. For example, if we collected 1000 structures, 
there are 400 converged and 600 rest. The following descriptor selection would select 
256 from 400 and 256 from 600. The number of candidates to label would be 512 (256+256) 
for each system then.

.. code-block:: yaml

    explorations:
      exp:
        systems: ["Cu4", "Cu8", "Cu12", "Cu24"]
        create:
          task: ./ga.yaml
          scheduler: ./scheduler.yaml
        collect:
          traj_period: 10 # every 10*dyn_dump_period steps plus last step
          boltz:
            random_seed: 1112
            fmax: 0.05 # eV/AA
            boltzmann: 3 # eV
            number: [256, 0.2] # 128 from 1000 (20*50)
        select: # list of selection procedures
          - method: convergence
            fmax: 0.05
          - method: descriptor
            random_seed: 1112
            descriptor:
              name: soap
              species: ["Cu"]
              rcut : 6.0
              nmax : 12
              lmax : 8
              sigma : 0.3
              average : inner
              periodic : true
            criteria:
              method: cur
              zeta: -1
              strategy: descent
            number: [256, 1.0]
