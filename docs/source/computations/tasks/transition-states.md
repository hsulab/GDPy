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
    setup:
      nimages: 5
      interpolation:
        mic: false
      climb: false
      constraint: "1:8"
    output:
      trajectory:
        period: 5
    stop:
      fmax: 0.08
      steps: 100
```

`nimages` sets the total image count, including endpoints; `interpolation` controls endpoint
interpolation. `climb: false` starts with ordinary NEB. Use a sufficiently
converged path before enabling a climbing image. `fmax` and `steps` control
path optimization; `constraint: "1:8"` fixes atoms 1 through 8.

Generate the endpoints, then run the path through the same one-shot
`gdp compute` interface used by other tasks:

```shell
python examples/compute/tasks/generate.py
gdp -d neb-demo -r examples/compute/tasks/neb.yaml \
    compute examples/compute/tasks/endpoints.xyz
```

The one-shot interface accepts a reactor runtime, treats the ordered structures
as one path, executes or submits it, and collects the final band. The explicit
`prepare`, `submit`, `status`, and `collect` lifecycle actions remain limited to
ordinary driver tasks.

For multiple paths, supply a flat list of images tagged with consecutive
`atoms.info["rxn_grp"]` group numbers. Untagged images form one path; do not
pass a nested Python list to this worker.

Inspect the image trajectories in `neb-demo` and check convergence before
using the energy profile as a barrier.

## NEB: O-H dissociation on Ni(111)

The repository also includes `examples/compute/ni111_water_neb/`, an
illustrative O-H dissociation path using the same Ni(111)/water structure as
the restraint-window example. Its `structure.xyz` is a relative link to the
canonical two-frame endpoint asset under `examples/compute/assets/`, so the
example remains directly runnable without duplicating structures.

Install the ReaxFF extra and run from the example directory:

```shell
python -m pip install -e '.[reax]'
cd examples/compute/ni111_water_neb
gdp -d _run -r runtime.yaml compute structure.xyz
```

The linked input contains the independently minimized molecular-water IS and
dissociated OH + H FS from xreac's validated Ni(111) seven-image path. Its
validation record reports endpoint forces below 0.02 eV/Angstrom with the
matching bundled 2026 ReaxFF model. The command runs a seven-image ASE NEB while
the lower Ni layer remains fixed. The final band is written to
`_run/results/end_frames.xyz`; detailed trajectory files remain under
`_run/pair0/`.

The supplied endpoints are minimized, but the short NEB settings remain a
mechanics example rather than a converged dissociation barrier. Production work
should converge the path, enable a climbing image when appropriate, and verify
the saddle with vibrational analysis.

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
