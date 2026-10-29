(compute-vibrations-example)=

# vibrational analysis

The `ase`, `vasp`, and `cp2k` executors provide the `freq` task. Start from a
relaxed structure and converged force-evaluation settings.

## ASE finite differences

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
complete only after the spectrum is written. An explicit executor rerun
recomputes all displacements; interrupted force caches are not resumed.

## VASP finite differences

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

## CP2K finite differences

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
