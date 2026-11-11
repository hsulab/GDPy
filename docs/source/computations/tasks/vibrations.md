(compute-vibrations-example)=

# vibrational analysis

The `ase`, `vasp`, and `cp2k` executors provide the `freq` task. Start from a
relaxed structure and converged force-evaluation settings.

## ase finite differences

The ASE executor uses [ASE Vibrations](https://ase-lib.org/ase/vibrations/modes.html)
with any potential that provides an ASE calculator. Save as `frequency.yaml`,
using your chosen potential; this example uses EMT:

```yaml
potential:
  provider: emt
executor:
  provider: ase
  method: freq
  parameters:
    setup:
      maxstep: 0.01
```

```shell
gdp -d frequency-results -r frequency.yaml compute relaxed.xyz
```

To calculate modes for a subset of atoms, add `group` under `setup`, using a
GDPy group expression such as ``group: "`index 0 1 2`"``. Alternatively, use
`indices: [0, 1, 2]` (zero-based). Without either option,
all atoms except those fixed by `FixAtoms` are selected. Explicit selections
overlapping fixed atoms are rejected; other constraint types are unsupported.
Use `setup.constraint` to override input constraints when needed.

`maxstep` is the displacement in Å. Central differences use two displacements
per coordinate by default; optionally set `nfree: 4` for a four-displacement
stencil. Only selected atoms are displaced; the remaining environment stays
fixed but participates in force evaluation. Selecting N atoms produces 3N
modes of the partial Hessian, without removing rigid translations or rotations.

Each calculation directory contains `vibrations.txt` (ASE's readable spectrum),
`vibrations.json` (indices, frequencies in cm⁻¹, energies in eV, zero-point
energy, Hessian, and modes), and the ASE force cache in `vibrations/`.
Imaginary components are stored separately in the JSON; `modes` has shape
`(3 * selected_atoms, total_atoms, 3)` with zeros outside the selected group.
The collected trajectory contains the undisplaced structure and is marked
complete only after the spectrum is written.

### resuming interrupted ase vibrations

The default `setup.resume: true` reuses completed force evaluations when
`vibrations.restart.json` matches the input geometry (including cell, species,
masses, charges and magnetic moments), constraints, selected indices,
displacement size, stencil, and calculator parameters. Nested mixer calculators
and their weights are included. Regular input files directly referenced by
calculator parameters are fingerprinted by content. External resources not
represented in those parameters (for example a pseudopotential directory's
contents or an executable replaced in place) must be kept unchanged; use a
fresh run if they change.

Empty, malformed, nonfinite, or incorrectly shaped force-cache entries are
removed and recalculated. Valid entries are retained. Changing the recorded
inputs invalidates the cache and recomputes all displacements.
Set `setup.resume: false` to explicitly recompute everything.

Caches produced before restart metadata was introduced are preserved but
rejected for automatic reuse: their original settings cannot be verified from
the force cache alone. They require a separately verified provenance migration,
or an explicit fresh run. Do not create a manifest from current inputs alone
and assume this validates an older cache.

Only run one executor for a given calculation directory at a time, and wait
for the previous scheduler job to stop before resubmitting. Restart support
does not itself submit or retry scheduler jobs. It resumes complete displacement
force evaluations, not an interrupted electronic SCF calculation. Completion
is still recorded only after every required force and the spectrum are saved.

## vasp finite differences

Save as `frequency.yaml`, using your VASP executable and input files:

```yaml
potential:
  provider: vasp
  parameters:
    command: mpirun -n 16 vasp_std
    pp_path: /path/to/potentials
    incar: ./INCAR
    kpts: [1, 1, 1]
executor:
  provider: vasp
  method: freq
  parameters:
    setup:
      optimizer:
        name: finite_difference
        parameters:
          maxstep: 0.015
```

```shell
gdp -d frequency-results -r frequency.yaml compute relaxed.xyz
```

`maxstep` is the displacement in Å. The VASP controller sets `IBRION=5`,
`NFREE=2`, `POTIM=maxstep`, and `NSW=1`. Inspect VASP’s native output for
frequencies and modes; the generic collected XYZ is not a vibrational spectrum.

## cp2k finite differences

Use a {doc}`CP2K potential <../../potentials/cp2k>` with this executor:

```yaml
executor:
  provider: cp2k
  method: freq
  parameters:
    setup:
      optimizer:
        name: finite_difference
        parameters:
          maxstep: 0.01
          num_cpus_per_replica: 16
```

For CP2K, `maxstep` is in **Bohr**, not Å. The controller selects
`VIBRATIONAL_ANALYSIS`, sets `DX`, and sets `NPROC_REP` from
`num_cpus_per_replica`. Match replica resources to the CP2K launch command and
queue allocation. Inspect the native CP2K output for the resulting modes.
