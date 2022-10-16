Explore Chemical Reactions
--------------------------
In this section, we demonstrate how to define an expedition using reaction exploration 
in `exp.yaml`. Here, we define an expedition called **exp** and all 
results would be save in the directory **exp** under where we run the `explore` 
command, and a log file `rxn.out` in **exp** can be found as well.

The reaction exploration method is the Artificial Force Induced Reaction (AFIR).

systems
_______
Standard **structure** definition is enough for this expedition. However, we need to
make sure that possible reactants can be found in these initial structures.

create
______
The setting is different from other expeditions since we are still trying to merge 
AFIR into the **driver** object. Currently, we need to set parameters for the keyword 
**AFIR**.

- target_pairs: Reactants.
- gmax, ginit, gintv: AFIR bias parameters.
- graph_params: Use to generate molecular graph to check whether a reaction happens.
- dyn_params: fmax and steps for minimisation.

collect
_______
**traj_period** is used to further select structures from minimisation trajectories. 
The structures in computation results would be split into three groups, namely 
transition states (TSs), final states (FSs), and trajectories (Traj).

.. note:: 

    The TSs from AFIR trajectories is just the structure with the highest energy among 
    the trajectory rather than the exact ones.

select
______
The selection would be performed on TS, FS and Traj separately.


.. code-block:: yaml

    exp:
      systems: ["reaction"]
      create:
        AFIR: # input structures should be optimised
          target_pairs: ["O", "CO"]
          gmax: 2.5
          ginit: 0.5
          gintv: 0.1
          #seed
          graph_params:
            # - graph
            pbc_grid: [1, 1, 0]
            graph_radius: 0
            # - neigh
            covalent_ratio: 1.1
            skin: 0.00
            # - site
            adsorbate_elements: ["C", "O"]
          dyn_params:
            fmax: 0.10 # eV/AA
            steps: 200
      collect:
        traj_period: 20
      select:
        - method: deviation # selection based on model deviation
          criteria:
            max_devi_e: [0.05, 0.25]
        - method: descriptor
          descriptor:
            name: soap
            species: ["C", "O", "Pt"]
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
          number: [4, 0.2]