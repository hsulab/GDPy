% gdpx documentation master file, created by
% sphinx-quickstart on Mon Aug 22 14:06:51 2022.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

# gdpx documentation

gdpx stands for **Generating Deep Potential with Python**, including
a set of tools and Python modules to automate the structure exploration
and the model training for **machine learning interatomic potentials** (MLIPs).
It is developed and maintained by [Jiayan Xu] under supervision of Prof. [P. Hu]
at Queen's University Belfast.

:::{figure} ../../assets/logo.png
:align: center
:alt: gdpx logo
:width: 400
:::

## Supported **Potentials**

Classical models, machine-learning potentials, and electronic-structure
interfaces are listed in the {doc}`potential provider overview <potentials/index>`.

## Supported **Explorations**

`molecular dynamics`, `genetic algorithm`, `grand canonical monte carlo`,
`graph-theory adsorbate configuration`, `artificial force induced reaction`

```{toctree}
:caption: 'Introduction:'
:maxdepth: 2

about.md
installation.md
```

```{toctree}
:caption: 'Basic Guides:'
:maxdepth: 2

start
```

```{toctree}
:caption: 'Define Potentials:'
:maxdepth: 2
:titlesonly:

overview <potentials/index>
potentials <potentials/providers>
```

```{toctree}
:caption: 'Train Potentials:'
:maxdepth: 2
:titlesonly:

command: gdp train <trainers/index>
potentials <trainers/potentials>
```

```{toctree}
:caption: 'Batch Simulations:'
:maxdepth: 2
:titlesonly:

command: gdp compute <computations/index>
demo and lifecycle <computations/compute>
tasks <computations/tasks/index>
runtime and executors <computations/runtime>
machine resources <computations/schedulers>
```

```{toctree}
:caption: 'Building Structures:'
:maxdepth: 1
:titlesonly:

command: gdp build <builders/index>
dimer <builders/dimer>
random <builders/random>
graph <builders/graph>
regions <builders/region>
```

```{toctree}
:caption: 'Selecting Structures:'
:maxdepth: 1
:titlesonly:

command: gdp select <selections/index>
descriptor <selections/descriptor>
```

```{toctree}
:caption: 'Exploring Structures:'
:maxdepth: 1
:titlesonly:

command: gdp explore <explorations/index>
```

```{toctree}
:caption: 'Canonical Sampling:'
:maxdepth: 2
:titlesonly:

monte carlo <explorations/mc>
```

```{toctree}
:caption: 'Global Optimisation:'
:maxdepth: 4
:titlesonly:

global_optimisation/genetic-algorithm
global_optimisation/basin_hopping
global_optimisation/population
global_optimisation/output
```

```{toctree}
:caption: 'Advanced Guides:'
:maxdepth: 2

sessions/index
workflows/index
```

```{toctree}
:caption: 'Developer Guides:'
:maxdepth: 2

extensions/index
architecture
migration-0.1
data/index
```

% modules/modules

```{toctree}
:caption: 'Gallery:'
:maxdepth: 2

applications/index
references
```

% Indices and tables

% ==================

%

% * :ref:`genindex`

% * :ref:`modindex`

% * :ref:`search`

[jiayan xu]: https://scholar.google.com/citations?user=ue5SBQMAAAAJ&hl=en
[p. hu]: https://scholar.google.com/citations?user=GNuXfeQAAAAJ&hl=en
