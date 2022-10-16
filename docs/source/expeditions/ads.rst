Explore Adsorbate Configurations
--------------------------------
In this section, we demonstrate how to define an expedition to find low-energy adsorbate 
configurations in `exp.yaml`. Here, we define an expedition called **exp** and all 
results would be save in the directory **exp** under where we run the `explore` 
command, and a log file `ads.out` in **exp** can be found as well.

This expedition is very similiar to the MD-based one but differ in the system 
definition.

systems
_______
We define a system that contains a p(3x3) 4-layer Pt(111) surface. The generator 
would enumerate possible O adsorption configurations using a graph-theory approach. 
Here, structures with O on hollow site (`site: 3` indicates site with 3 coordination 
number) would be created. (see article [1])

[1] Deshpande, S.; Maxson, T.; Greeley, J. 
    Graph Theory Approach to Determine Configurations of Multidentate and High Coverage Adsorbates for Heterogeneous Catalysis. 
    npj Comput. Mater. 2020, 6, 79.

.. code-block:: yaml

    surface:
      prefix: Pt111
      generator:
        method: adsorbate
        substrate: ./surfaces.xyz
        composition:
          - species: O
            action: add
            distance_to_site: 1.5
            site: 3 # site preference, coordination number is 3
        graph:
          # - graph
          pbc_grid: [2, 2, 0]
          graph_radius: 2
          # - neigh
          covalent_ratio: 1.1
          skin: 0.25
          adsorbate_elements: ["O"]
          coordination_numbers: [3]
          site_radius: 3
          check_site_unique: true
      composition: {"O": 1, "Pt": 36}
      constraint: "1:18"
      kpts: [4, 4, 1]

create
______
Each generated structures would be minimised. The **driver** can be set according 
to :ref:`driver examples`. Important parameters are **traj_period** and **steps**,
which determine how many structure would be collected for later analysis. Their usage
is the same as the MD-based expedition.

collect
_______
This is the same as the MD-based one.

select
______
This is similiar to the MD-based one. A convergence selection is usually applied to 
split collected structures into two groups for later selection. The converged ones 
are usually stable adsorption configurations that can be used to generate convex hull. 
Meanwhile, the structures from minimisation trajectories can be used for training since 
they cover the configuration space that productions may come across.


.. code-block:: yaml

    explorations:
      exp:
        systems: ["surface"]
        create:
          driver:
            backend: external
            task: min
            init:
              dump_period: 8
            run:
              steps: 12
        collect:
          traj_period: 2
        select: # list of selection procedures
          - method: convergence
            fmax: 0.05 # eV/AA
          - method: deviation # selection based on model deviation
            criteria:
              max_devi_e: [0.01, 0.25]
              max_devi_f: [0.05, 0.25]
          - method: descriptor
            random_seed: 1112
            descriptor:
              name: soap
              species: ["C", "H", "O", "Pt"]
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