% GDPy documentation master file, created by
% sphinx-quickstart on Mon Aug 22 14:06:51 2022.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

# GDPy Documentation

GDPy stands for **Generating Deep Potential with Python**, including
a set of tools and Python modules to automate the structure exploration
and the model training for **machine learning interatomic potentials** (MLIPs).
It is developed and maintained by [Jiayan Xu] under supervision of Prof. [P. Hu]
at Queen's University Belfast.

:::{figure} ../../assets/logo.png
:align: center
:alt: GPDy LOGO
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
computations/index
builders/index
selections/index
explorations/index
tutorials/index
```

```{toctree}
:caption: 'Potentials:'
:maxdepth: 2
:titlesonly:

Overview <potentials/index>
Potentials <potentials/providers>
Training <trainers/index>
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
