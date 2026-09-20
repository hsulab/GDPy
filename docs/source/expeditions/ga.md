(genetic-algorithm)=

# Genetic Algorithm (GA)

## Overview

Genetic algorithm is a popular global optimisation method to find stable structures.
GA in gdpy makes use of functionalities in `ase` package and provides a user-friendly
interface by YAML.

The workflow of a GA-based global optimisation is

```{image} ../../images/ga.png
:align: center
:alt: Genetic algorithm workflow
:width: 800
```

The steps are

1. Initial Population

   > - Generate an initial population of structures.
   > - Relax initial structures.

2. Iterate population until convergence.

   > - **Selection**: Choose the best-*N* structures to form a new population.
   >
   > - **Crossover**: Use the periodic cut-and-splice method to generate a new structure from two structures.
   >   This is critical to the success of GA. See the schema below. (Phys. Rev. Lett. 2012, 108, 126101.)
   >
  >   > ```{image} ../../images/CutAndSplice.png
  >   > :align: center
  >   > :alt: Cut-and-splice crossover
  >   > :width: 400
  >   > ```
   >
   > - **Mutation**: Each new structure (offspring) has a possibility to mutate that part of structures
   >   are modified.
   >
   > - **Minimisation**: Relax new structures.
   >
   > - **Convergence**: If the maximum number of generations (iterations) reaches.

3. Anaylse structures.

   > - Find structures with target properties in all explored structures.

## Example

To use GA, the related commands are

```shell
# - explore configuration space defined by `config.yaml`
#   results will be written to the `results` folder
#   a log file will be written to `results/gdp.out` as well
$ gdp -d exp -r runtime.yaml explore ./config.yaml

# - after GA is converged i.e. reaches the maximum generation,
#   all found minima will be saved to `./resuslts/results/all_candidates.xyz`
```

The GA input file `./config.yaml` uses the common global-optimisation layout:

- method: This must be `genetic_algorithm`.

- recipe:

  > Contains the random seed, population generator, and all GA-specific settings.

- recipe.population.builders:

  > Define named builders that can initialise the population and complete later
  > generations. See {ref}`random-builders` for more details.
  > The example builder will put 4 Cu atoms on the substrate stored in the file
  > `./sub.xyz`. Cu atoms are randomly created in a `lattice region`, details of which
  > can be found in {ref}`region-definitions`. More specific, Cu atoms will have
  > arbitrary x- and y-coordiantes but z-coordinate within the range [7,7+6].

GDPy stores explored structures and restart metadata in `candidates.db` inside
the expedition directory; its name is not configurable. The remaining entries
below are in the `recipe` section:

- objective: Optional search target. It defaults to `energy`, so the section can
  be omitted for ordinary energy minimisation. Composition-dependent targets
  use `chemical_potentials` to rank candidates with different compositions.

- convergence: Convergence criteria, e.g., the maximum number of generation.

- population: Define how to create and organise a population. See the shared
  {ref}`global-optimisation-population` reference, including `retained_size`.

  > - periodic and preserve_fragments:
  >
  >   > Both booleans default to `true`. The default makes the searched system
  >   > periodic in all three directions and requires compatible operations to
  >   > keep atoms sharing an ASE tag together as one fragment. Set either value
  >   > explicitly to `false` for a nonperiodic or atom-wise search.
  >
  > - builders and reference_builder:
  >
  >   > `builders` is a mapping of reusable, named structure builders.
  >   > The `random` builder supplies substrate, tags, cell bounds, and bond-distance
  >   > metadata to the genetic operators by default. Set `reference_builder` only
  >   > when another named builder should supply this metadata.
  >
  > - initial:
  >
  >   > `total_size` is the initial population size. `builder_allocations` selects
  >   > one or more named builders and gives the exact `size` produced by each.
  >
  > - generation:
  >
  >   > `total_size` is the target size. Candidates are attempted in order:
  >   > `reproduction`, direct `mutation`, then `completion`. Completion builders
  >   > fill the actual deficit according to `builder_proportions`.

- operators:

  > - crossover:
  >
  >   > This is the most critical operator in GA.
  >
  > - mutation: List of mutation operators.
  >
  >   > Each operator can be selected based on relative probabilities.

```yaml
method: genetic_algorithm
recipe:
  random_seed: 127
  population:
    comparator:
      dE: 0.015
      method: interatomic_distance
    preserve_fragments: false
    builders:
      surface:
        method: random_surface
        composition:
          Cu: 4
        region:
          method: lattice
          origin: [0., 0., 7.]
          cell: [11.174, 0., 0., 0., 8.413, 0., 0., 0., 6.]
        substrates: ./sub.xyz
        covalent_ratio: [0.8, 2.0]
    reference_builder: surface
    initial:
      total_size: 5
      builder_allocations:
        - builder: surface
          size: 5
    generation:
      total_size: 5
      reproduction:
        size: 3
        mutation_probability: 0.8
      mutation:
        size: 0
      completion:
        builder_proportions:
          - builder: surface
            proportion: 1.0
  operators:
    crossover:
      method: periodic_cut_and_splice
    mutation:
    - method: rattle
      probability: 1.0
    - method: mirror
      probability: 1.0
  convergence:
    generation: 2
```

## Application

1. {ref}`ref-jpcc-2022-xu`
2. {ref}`ref-acs-catal-2022-xu`
3. {ref}`ref-acs-catal-2023-han`
