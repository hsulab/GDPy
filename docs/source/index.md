:::{figure} ../../assets/gdpx-logo.png
:align: center
:alt: GDPx logo
:width: 400
:::

# GDPy

GDPy builds MLIP-driven workflows for atomistic simulation using advanced and
newly developed structure-exploration methods. It integrates sampling, dataset
construction, and model training into adaptive workflows.

## Scope

**Focused systems**, with an emphasis on heterogeneous catalysis:

- metal oxides
- supported clusters
- disorder and amorphous surfaces
- solid–liquid interfaces

**Highlighted methods:**

- molecular dynamics (accelerated dynamics and enhanced sampling)
- monte carlo (canonical, semi-grand canonical, and grand canonical)
- global optimisation (structures and reactions)

## Get started

Install GDPy by following the {doc}`installation guide <installation>`, then
choose a complete workflow from the {doc}`quick examples <quick-examples>`.
The {doc}`potential provider overview <potentials/index>` lists the available
classical, machine-learning, and electronic-structure interfaces.

% Keep the documentation hierarchy in the sidebar without rendering a large
% index tree on the landing page.

```{toctree}
:caption: 'Introduction:'
:maxdepth: 2
:hidden:

about.md
installation.md
```

```{toctree}
:caption: 'Basic Guides:'
:maxdepth: 2
:hidden:

units
quick-examples
```

```{toctree}
:caption: 'Define Potentials:'
:maxdepth: 2
:titlesonly:
:hidden:

overview <potentials/index>
potentials <potentials/providers>
```

```{toctree}
:caption: 'Train Potentials:'
:maxdepth: 2
:titlesonly:
:hidden:

command: gdp train <trainers/index>
potentials <trainers/potentials>
```

```{toctree}
:caption: 'Batch Simulations:'
:maxdepth: 2
:titlesonly:
:hidden:

command: gdp compute <computations/index>
demo and lifecycle <computations/compute>
tasks <computations/tasks/index>
runtime configurations <computations/runtime>
machine resources <computations/schedulers>
```

```{toctree}
:caption: 'Build Structures:'
:maxdepth: 2
:titlesonly:
:hidden:

command: gdp build <builders/index>
methods <builders/methods>
regions <builders/region>
```

```{toctree}
:caption: 'Select Structures:'
:maxdepth: 2
:titlesonly:
:hidden:

command: gdp select <selections/index>
methods <selections/methods>
```

```{toctree}
:caption: 'Explore Structures:'
:maxdepth: 1
:titlesonly:
:hidden:

command: gdp explore <explorations/index>
```

```{toctree}
:caption: 'Boltzmann Sampling:'
:maxdepth: 4
:titlesonly:
:hidden:

overview <explorations/sampling>
monte carlo <explorations/mc>
hybrid monte carlo <explorations/hmc>
operators <explorations/operators/index>
```

```{toctree}
:caption: 'Global Optimisation:'
:maxdepth: 4
:titlesonly:
:hidden:

overview <global_optimisation/index>
global_optimisation/genetic-algorithm
global_optimisation/basin_hopping
global_optimisation/population
global_optimisation/output
```

```{toctree}
:caption: 'Advanced Guides:'
:maxdepth: 2
:hidden:

workflows/index
```

```{toctree}
:caption: 'Developer Guides:'
:maxdepth: 2
:hidden:

extensions/index
architecture
migration-0.1
data/index
```

% modules/modules

```{toctree}
:caption: 'Gallery:'
:maxdepth: 2
:hidden:

applications/index
references
```

% Indices and tables

% ==================

%

% * :ref:`genindex`

% * :ref:`modindex`

% * :ref:`search`
