# transition states and paths

(compute-neb-example)=

## NEB: a surface-diffusion path

NEB needs an ordered path or two endpoints with the same atom ordering and
cell. A flat list of unrelated structures is not a reaction path.
The generated demo endpoints describe Au moving between sites on an Al slab.
The bottom eight atoms are fixed, and both endpoints are relaxed first.

Use `examples/compute/tasks/neb.yaml`:

```yaml
potential:
  provider: emt
executor:
  provider: ase
  method: neb
  parameters:
    nimages: 5
    interpolation:
      mic: false
    climb: false
    fmax: 0.08
    steps: 100
    dump_period: 5
    constraint: "1:8"
```

`nimages` sets the total image count, including endpoints; `interpolation` controls endpoint
interpolation. `climb: false` starts with ordinary NEB. Use a sufficiently
converged path before enabling a climbing image. `fmax` and `steps` control
path optimization; `constraint: "1:8"` fixes atoms 1 through 8.

The current `gdp compute` lifecycle accepts driver workers only and rejects
NEB reactor workers. Run this demo through the reactor worker API:

```shell
python examples/compute/tasks/generate.py
python examples/compute/tasks/run_neb.py
```

The script loads the YAML, passes the ordered endpoints as one path, runs it, and
retrieves the result:

```python
worker = create_worker(config, directory="neb-demo")
worker.run(endpoints)
worker.inspect(endpoints)
paths = worker.retrieve(include_retrieved=True)
```

For multiple paths, supply a flat list of images tagged with consecutive
`atoms.info["rxn_grp"]` group numbers. Untagged images form one path; do not
pass a nested Python list to this worker.

Inspect the image trajectories in `neb-demo` and check convergence before
using the energy profile as a barrier.

(compute-dimer-example)=

## Dimer: a local transition-state search

Dimer searches start from one structure near a saddle point. The CP2K executor
supports `dimer` (also represented internally as `ts`). With a prepared CP2K
input template and an initial structure, save this as `dimer.yaml`:

```yaml
potential:
  provider: cp2k
  parameters:
    command: cp2k.psmp
    template: ./cp2k-template.inp
executor:
  provider: cp2k
  method: dimer
  parameters:
    steps: 100
    fmax: 0.05
    controller:
      name: dimer
      params:
        dimer_vector: ./dimer-vector.txt
```

The template must reference the required basis and pseudopotential data.
`dimer-vector.txt` supplies the initial non-mass-weighted Cartesian direction
in CP2K’s DIMER_VECTOR input format.

```shell
gdp -d dimer-results -r dimer.yaml compute saddle-guess.xyz
```

This example requires a CP2K installation and system-specific inputs. Verify
a candidate saddle with vibrational analysis and the intended reaction path.
Although `ase:dimer` is registered, its driver currently lacks the matching
controller, so it is not a runnable alternative in this version.
