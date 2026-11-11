:::{figure} ../../assets/gdpx-logo.png
:align: center
:alt: GDPx logo
:width: 400
:::

# gdpy

GDPy builds MLIP-driven workflows for atomistic simulation using advanced and
newly developed structure-exploration methods. It integrates sampling, dataset
construction, and model training into adaptive workflows.

## scope

**Focused systems**, with an emphasis on heterogeneous catalysis:

- metal oxides
- supported clusters
- disorder and amorphous surfaces
- solid–liquid interfaces

**Highlighted methods:**

- molecular dynamics (accelerated dynamics and enhanced sampling)
- monte carlo (canonical, semi-grand canonical, and grand canonical)
- global optimisation (structures and reactions)

## get started

Install GDPy by following the {doc}`installation guide <installation>`, then
choose a complete workflow from the {doc}`quick examples <quick-examples>`.
The {doc}`potential provider overview <potentials/index>` lists the available
classical, machine-learning, and electronic-structure interfaces.

% Keep the documentation hierarchy in the sidebar without rendering a large
% index tree on the landing page.

```{toctree}
:caption: 'introduction:'
:maxdepth: 2
:hidden:

about.md
installation.md
```

```{toctree}
:caption: 'basic guides:'
:maxdepth: 2
:hidden:

units
groups-and-constraints
quick-examples
```

```{toctree}
:caption: 'define potentials:'
:maxdepth: 2
:titlesonly:
:hidden:

overview <potentials/index>
potentials <potentials/providers>
```

```{toctree}
:caption: 'train potentials:'
:maxdepth: 2
:titlesonly:
:hidden:

command: gdp train <trainers/index>
potentials <trainers/potentials>
```

```{toctree}
:caption: 'batch simulations:'
:maxdepth: 2
:titlesonly:
:hidden:

command: gdp compute <computations/index>
demo and lifecycle <computations/compute>
tasks <computations/tasks/index>
runtime configurations <computations/runtime>
machine resources <computations/schedulers>
molecular dynamics <computations/tasks/molecular-dynamics>
monte carlo <computations/monte-carlo/index>
```

```{toctree}
:caption: 'build structures:'
:maxdepth: 2
:titlesonly:
:hidden:

command: gdp build <builders/index>
methods <builders/methods>
regions <builders/region>
```

```{toctree}
:caption: 'select structures:'
:maxdepth: 2
:titlesonly:
:hidden:

command: gdp select <selections/index>
methods <selections/methods>
```

```{toctree}
:caption: 'explore structures:'
:maxdepth: 1
:titlesonly:
:hidden:

command: gdp explore <explorations/index>
```

```{toctree}
:caption: 'global optimisation:'
:maxdepth: 4
:titlesonly:
:hidden:

overview <global_optimisation/index>
examples <global_optimisation/examples/index>
global_optimisation/genetic-algorithm
global_optimisation/basin_hopping
simulated annealing <global_optimisation/simulated-annealing>
global_optimisation/population
global_optimisation/output
```

```{toctree}
:caption: 'advanced guides:'
:maxdepth: 2
:hidden:

workflows/index
```

```{toctree}
:caption: 'developer guides:'
:maxdepth: 2
:hidden:

extensions/index
architecture
migration-0.1
data/index
```

% modules/modules

```{toctree}
:caption: 'gallery:'
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
