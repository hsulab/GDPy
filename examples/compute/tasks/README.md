# Batch simulation task demos

Run from the repository root in the GDPy environment:

```shell
python examples/compute/tasks/generate.py
gdp -d spc-demo -r examples/compute/tasks/single-point.yaml compute examples/compute/tasks/dimers.xyz
gdp -d min-demo -r examples/compute/tasks/relaxation.yaml compute examples/compute/tasks/dimers.xyz
gdp -d cmin-demo -r examples/compute/tasks/cell-relaxation.yaml compute examples/compute/tasks/bulk.xyz
gdp -d md-demo -r examples/compute/tasks/molecular-dynamics.yaml compute examples/compute/tasks/md.xyz
python examples/compute/tasks/run_neb.py
```

These use ASE's bundled EMT calculator and need no model downloads or queue.
The NEB example uses the reactor worker API; the current `gdp compute`
lifecycle handles driver workers only. Use new output directories for changed
inputs. Task explanations are in `docs/source/computations/tasks/`.
