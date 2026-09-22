# vibrational analysis

Native `vasp` and `cp2k` executors provide the `freq` task. Start from a relaxed
structure and converged electronic settings; the EMT demos do not provide a
native frequency executor.

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
    controller:
      name: finite_difference
      params:
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
    controller:
      name: finite_difference
      params:
        maxstep: 0.01
        num_cpus_per_replica: 16
```

For CP2K, `maxstep` is in **Bohr**, not Å. The controller selects
`VIBRATIONAL_ANALYSIS`, sets `DX`, and sets `NPROC_REP` from
`num_cpus_per_replica`. Match replica resources to the CP2K launch command and
queue allocation. Inspect the native CP2K output for the resulting modes.
