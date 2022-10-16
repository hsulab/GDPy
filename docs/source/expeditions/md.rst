Explore with Molecular Dynamics (MD)
------------------------------------
In this section, we demonstrate how to define a MD-based expedition in `exp.yaml` 
by explaining each keyword. Here, we define an expedition called **exp** and all 
results would be save in the directory **exp** under where we run the `explore` 
command, and a log file `MD.out` in **exp** can be found as well.

systems
_______
We try to explore a system called `sysName` as in **systems** if it were properly 
defined in the **systems** section. 

create
______
The exploration strategy is defined as a driver in `create-driver`, which has 
the same format as we use in `pot.yaml`. If there were one driver attached with 
potential, it would be overridden as the one here. It should be aware of that the 
parameters would be broadcast here. For example, since **temp** is a list of 
integers (`[150, 300, 600]`), there would be three MD simulations for each structure. 
Each simulation run 12 **steps** with a **timestep** of 2 fs. As **dump_period** 
is 4, structures at step `0,4,8,12` would be dumped for later collection, which 
helps save disk usage when runing a very long simulation.

collect
_______
The keyword **traj_period** helps us further choose structures from dumped trajectories.
As it is 2, frame `4` and `12` would be chosen from `0,4,8,12`. Since three simulations 
with different temperatures were run, there would be 6 (2 times 3) structures collected 
for later analysis.

select
______
This is an optional procedure. However, it is highly recommended since it can reduce 
the dataset size by selecting the most representative structures. We can define 
a series of selections to sift collected structures. Here, candidates would be selected 
from 6 structures if they have prediction deviations in the range (`max_devi_e` 
and `max_devi_f`). If there were any selected after the deviation, they would be 
further sifted by their importance in the feature space spanned by the SOAP descriptor. 
At last, the number of candidates would be 2 if there is enough or 0.2 times the 
number of rest structures from the previous selection (`number: [2, 0.2]`).

.. code-block:: yaml

    explorations:
      exp:
        systems: ["sysName"]
        create:
          driver:
            backend: external
            task: md # md
            init:
              md_style: nvt
              temp: [150, 300, 600] # broadcast params
              Tdamp: 100 # fs
              timestep: 2 # fs
              dump_period: 4
            run:
              steps: 12
        collect:
          traj_period: 2
        select: # list of selection procedures
          - method: deviation # selection based on model deviation
            random_seed: 1112
            criteria:
              max_devi_e: [0.01, 0.25]
              max_devi_f: [0.05, 0.25]
          - method: descriptor
            random_seed: 1112
            descriptor:
              name: soap
              species: ["C", "Pd"]
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
            number: [2, 0.2]